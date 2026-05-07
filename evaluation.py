import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
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
from graph import Neo4jClient, query_graph_rag, has_any_entities, measure_answer_hops


def _run_in_thread(fn, *args, **kwargs):
    import asyncio

    def _target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return fn(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_target)
        return future.result()


def _parse_context_items(result: dict) -> tuple[list[str], list[dict]]:
    contexts = []
    raw_items = []
    if result["context"] and result["context"].items:
        for i, item in enumerate(result["context"].items, 1):
            content_str = str(item.content)
            contexts.append(content_str[:4000])
            meta = item.metadata if isinstance(item.metadata, dict) else {}
            score_val = f"{meta['score']:.4f}" if "score" in meta else "N/A"
            raw_items.append({"index": i, "score": score_val, "text": content_str[:2000]})
    return contexts, raw_items


def generate_testset(texts: list[str], model: str = "gpt-4o", testset_size: int = 10) -> pd.DataFrame:
    import httpx

    MAX_TOTAL_CHARS = 100_000
    MAX_CHARS_PER_DOC = 50_000

    trimmed = [t[:MAX_CHARS_PER_DOC] for t in texts]
    total = sum(len(t) for t in trimmed)
    if total > MAX_TOTAL_CHARS:
        ratio = MAX_TOTAL_CHARS / total
        trimmed = [t[:int(len(t) * ratio)] for t in trimmed]

    lc_docs = [
        LCDocument(page_content=t, metadata={"source": f"doc_{i}"})
        for i, t in enumerate(trimmed) if t.strip()
    ]

    def _generate():
        import os
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        http_client = httpx.Client(
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            timeout=httpx.Timeout(120.0, connect=30.0),
        )
        async_http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            timeout=httpx.Timeout(120.0, connect=30.0),
        )
        llm = LangchainLLMWrapper(ChatOpenAI(
            model=model,
            openai_api_key=OPENAI_API_KEY,
            timeout=120,
            max_retries=3,
            http_client=http_client,
            http_async_client=async_http_client,
        ))
        embeddings = LangchainEmbeddingsWrapper(
            LCOpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=OPENAI_API_KEY,
                timeout=60,
                max_retries=3,
            )
        )
        generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
        return generator.generate_with_langchain_docs(lc_docs, testset_size=testset_size)

    for attempt in range(3):
        try:
            dataset = _run_in_thread(_generate)
            break
        except Exception as e:
            if attempt < 2:
                wait = 10 * (attempt + 1)
                logger.warning(f"Testset generation attempt {attempt+1} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    df = dataset.to_pandas()

    logger.info(f"RAGAS testset columns: {list(df.columns)}")
    logger.info(f"RAGAS testset shape: {df.shape}")
    if not df.empty:
        logger.info(f"RAGAS testset first row: {df.iloc[0].to_dict()}")

    q_col = next((c for c in ["user_input", "question"] if c in df.columns), None)
    a_col = next((c for c in ["reference", "ground_truth", "reference_answer", "answer"] if c in df.columns), None)

    if q_col is None:
        raise ValueError(f"RAGAS output has no question column. Columns found: {list(df.columns)}")

    result = pd.DataFrame()
    result["question"] = df[q_col]
    result["ground_truth"] = df[a_col] if a_col else "N/A"
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
    for idx, row in qa_pairs.iterrows():
        if len(rows) > 0:
            time.sleep(1)
        question = str(row["question"])
        ground_truth = str(row["ground_truth"])

        answer = None
        for attempt in range(3):
            try:
                result = query_graph_rag(
                    driver, question, model,
                    hops=max_hops,
                    weight_threshold=0.1,
                    confidence_threshold=0.0,
                )
                answer = result["answer"]
                contexts, raw_items = _parse_context_items(result)
                hops_used = result.get("hops_used", 0)
                if hops_used == 0:
                    hops_used = measure_answer_hops(
                        driver, question, answer, max_hops=max_hops,
                    )
                logger.info(f"Q: {question[:80]}... → hops_used={hops_used}")
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Query attempt {attempt+1} failed ({e}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"Query failed after 3 attempts: {question} — {e}")
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

    def _evaluate():
        import os
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=model, openai_api_key=OPENAI_API_KEY))
        return evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
            llm=evaluator_llm,
        )

    for attempt in range(3):
        try:
            ragas_results = _run_in_thread(_evaluate)
            break
        except Exception as e:
            if attempt < 2:
                wait = 5 * (attempt + 1)
                logger.warning(f"RAGAS evaluation attempt {attempt+1} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"RAGAS evaluation failed after 3 attempts: {e}")
                raise

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
