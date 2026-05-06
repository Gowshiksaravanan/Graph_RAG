import re
import pandas as pd
from loguru import logger
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LCOpenAIEmbeddings
from langchain_core.documents import Document as LCDocument

from config import OPENAI_API_KEY
from graph import Neo4jClient, query_graph_rag, has_any_entities


def _extract_hops(context_items: list[dict]) -> int:
    max_hops = 0
    for item in context_items:
        text = item.get("text", "")
        arrows = text.count("]->")
        if arrows > max_hops:
            max_hops = arrows
    return max_hops


def _parse_context_items(result: dict) -> tuple[list[str], list[dict]]:
    contexts = []
    raw_items = []
    if result["context"] and result["context"].items:
        for i, item in enumerate(result["context"].items, 1):
            content_str = str(item.content)
            contexts.append(content_str[:2000])
            score_match = re.search(r"score=([\d.]+)", content_str)
            score_val = f"{float(score_match.group(1)):.4f}" if score_match else "N/A"
            raw_items.append({"index": i, "score": score_val, "text": content_str[:500]})
    return contexts, raw_items


def generate_testset(texts: list[str], model: str = "gpt-4o", testset_size: int = 10) -> pd.DataFrame:
    llm = LangchainLLMWrapper(ChatOpenAI(model=model, api_key=OPENAI_API_KEY))
    embeddings = LangchainEmbeddingsWrapper(LCOpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY))

    lc_docs = [
        LCDocument(page_content=t, metadata={"source": f"doc_{i}"})
        for i, t in enumerate(texts)
    ]

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
    dataset = generator.generate_with_langchain_docs(lc_docs, testset_size=testset_size)
    df = dataset.to_pandas()

    result = pd.DataFrame()
    result["question"] = df.get("user_input", df.get("question", pd.Series()))
    result["ground_truth"] = df.get("reference", df.get("ground_truth", pd.Series()))
    return result


def run_evaluation(
    driver,
    qa_pairs: pd.DataFrame,
    model: str = "gpt-4o",
    max_hops: int = 3,
) -> pd.DataFrame:
    if not has_any_entities(driver):
        raise ValueError("No knowledge graph found. Build one first.")

    rows = []
    for _, row in qa_pairs.iterrows():
        question = str(row["question"])
        ground_truth = str(row["ground_truth"])

        try:
            result = query_graph_rag(
                driver, question, model,
                hops=max_hops,
                weight_threshold=0.1,
                confidence_threshold=0.0,
            )
            answer = result["answer"]
            contexts, raw_items = _parse_context_items(result)
            hops_used = _extract_hops(raw_items)
        except Exception as e:
            logger.error(f"Query failed for: {question} — {e}")
            answer = f"Error: {e}"
            contexts = []
            hops_used = 0

        rows.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts,
            "hops_used": hops_used,
        })

    results_df = pd.DataFrame(rows)

    eval_data = []
    for _, r in results_df.iterrows():
        eval_data.append({
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r["contexts"] if r["contexts"] else ["No context retrieved."],
            "reference": r["ground_truth"],
        })

    dataset = EvaluationDataset.from_list(eval_data)
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=model, api_key=OPENAI_API_KEY))

    ragas_results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=evaluator_llm,
    )

    ragas_df = ragas_results.to_pandas()

    results_df["faithfulness"] = ragas_df.get("faithfulness", 0.0)
    results_df["answer_relevancy"] = ragas_df.get("answer_relevancy", 0.0)
    results_df["context_precision"] = ragas_df.get("context_precision", 0.0)
    results_df["context_recall"] = ragas_df.get("context_recall", 0.0)

    avg_ragas = results_df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean(axis=1)
    results_df["ges"] = avg_ragas * (max_hops - results_df["hops_used"].clip(upper=max_hops) + 1) / max_hops

    return results_df


def compute_summary(results_df: pd.DataFrame, max_hops: int) -> dict:
    return {
        "avg_faithfulness": results_df["faithfulness"].mean(),
        "avg_answer_relevancy": results_df["answer_relevancy"].mean(),
        "avg_context_precision": results_df["context_precision"].mean(),
        "avg_context_recall": results_df["context_recall"].mean(),
        "avg_hops_used": results_df["hops_used"].mean(),
        "avg_ges": results_df["ges"].mean(),
        "max_hops": max_hops,
        "num_questions": len(results_df),
    }
