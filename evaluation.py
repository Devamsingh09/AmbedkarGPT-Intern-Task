import os
import json
import time
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# LangChain new imports (your requirement)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

# NLTK data
nltk.download("punkt", quiet=True)


# ----------------------------- CONFIG -----------------------------
CORPUS_DIR = "corpus"
TEST_DATA = "test_dataset.json"
PERSIST_ROOT = "./eval_chroma"
OUTPUT_JSON = "test_results.json"
OUTPUT_MD = "results_analysis.md"

CHUNK_STRATEGIES = {
    "small": (250, 40),
    "medium": (550, 60),
    "large": (900, 100)
}

TOP_K = 5
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ----------------------------- HELPERS -----------------------------

def load_test_questions():
    with open(TEST_DATA, "r", encoding="utf-8") as f:
        return json.load(f)["test_questions"]


def load_corpus():
    corpus = {}
    for fname in os.listdir(CORPUS_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(CORPUS_DIR, fname), "r", encoding="utf-8") as f:
                corpus[fname] = f.read()
    return corpus


def split_text(text, chunk_size, overlap):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return [Document(page_content=c) for c in splitter.split_text(text)]


def build_chroma(corpus, strategy_name, chunk_size, overlap):
    persist_dir = os.path.join(PERSIST_ROOT, strategy_name)
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    docs, metas = [], []
    for fname, text in corpus.items():
        chunks = split_text(text, chunk_size, overlap)
        for i, c in enumerate(chunks):
            docs.append(c)
            metas.append({"source": fname, "chunk": i})

    vectordb = Chroma.from_documents(
        docs,
        embeddings,
        collection_name="ambedkar_eval",
        persist_directory=persist_dir,
    )

    return vectordb


# ------------------- RETRIEVAL + LCEL QA CHAIN -------------------

def build_lcel_chain(retriever):
    prompt = ChatPromptTemplate.from_template(
        """You are an expert assistant. Use ONLY the given retrieved context.
If answer cannot be found, say "I don't know".

Context:
{context}

Question: {question}

Answer:"""
    )

    model = ChatOllama(model="mistral")
    parser = StrOutputParser()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    return chain


# ----------------------------- METRICS -----------------------------

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
bleu_smooth = SmoothingFunction().method1
st_embedder = SentenceTransformer(EMBED_MODEL)


def compute_answer_metrics(pred, truth):
    if not truth:
        return {"rougeL": 0, "bleu": 0, "cosine": 0, "relevance": 0, "faithfulness": 0}

    # ROUGE-L
    r = rouge.score(truth, pred)["rougeL"].fmeasure

    # BLEU
    try:
        b = sentence_bleu(
            [nltk.word_tokenize(truth.lower())],
            nltk.word_tokenize(pred.lower()),
            smoothing_function=bleu_smooth,
        )
    except:
        b = 0.0

    # COSINE
    emb_t = st_embedder.encode([truth])
    emb_p = st_embedder.encode([pred])
    cos = float(cosine_similarity(emb_t, emb_p)[0][0])

    # Relevance (mix semantic + lexical)
    relevance = (cos + r) / 2

    # Faithfulness (approx)
    faith = 1 if (cos > 0.6 and r > 0.2) else 0

    return {
        "rougeL": r,
        "bleu": b,
        "cosine": cos,
        "relevance": relevance,
        "faithfulness": faith,
    }


def compute_retrieval_metrics(retrieved_sources, gold_sources):
    hit = 1 if any(src in gold_sources for src in retrieved_sources) else 0
    rr = 0.0
    for i, src in enumerate(retrieved_sources, 1):
        if src in gold_sources:
            rr = 1 / i
            break
    p_at_k = sum(1 for s in retrieved_sources[:TOP_K] if s in gold_sources) / TOP_K
    return hit, rr, p_at_k


# ----------------------------- EVALUATION -----------------------------

def evaluate_strategy(name, vectordb, questions):
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    chain = build_lcel_chain(retriever)

    results = []
    agg = {
        "hit": 0, "mrr": 0, "p_at_k": 0,
        "rouge": 0, "bleu": 0, "cosine": 0,
        "relevance": 0, "faithfulness": 0
    }
    count = len(questions)

    for q in questions:
        qid = q["id"]
        text = q["question"]
        gold = q["ground_truth"]
        gold_sources = q["source_documents"]

        # Retrieve docs manually
        docs = retriever.invoke(text)
        retrieved_sources = [d.metadata.get("source") for d in docs]

        # LCEL answer
        answer = chain.invoke(text)

        # Retrieval metrics
        hit, rr, p_at_k = compute_retrieval_metrics(retrieved_sources, gold_sources)

        # Answer quality
        met = compute_answer_metrics(answer, gold)

        results.append({
            "id": qid,
            "question": text,
            "gold_answer": gold,
            "generated_answer": answer,
            "retrieved_sources": retrieved_sources,
            "metrics": {
                "hit": hit,
                "mrr": rr,
                "precision_at_k": p_at_k,
                **met
            }
        })

        agg["hit"] += hit
        agg["mrr"] += rr
        agg["p_at_k"] += p_at_k
        agg["rouge"] += met["rougeL"]
        agg["bleu"] += met["bleu"]
        agg["cosine"] += met["cosine"]
        agg["relevance"] += met["relevance"]
        agg["faithfulness"] += met["faithfulness"]

    # averages
    final = {
        "strategy": name,
        "avg_hit": agg["hit"] / count,
        "avg_mrr": agg["mrr"] / count,
        "avg_precision_at_k": agg["p_at_k"] / count,
        "avg_rougeL": agg["rouge"] / count,
        "avg_bleu": agg["bleu"] / count,
        "avg_cosine": agg["cosine"] / count,
        "avg_relevance": agg["relevance"] / count,
        "avg_faithfulness": agg["faithfulness"] / count,
        "details": results,
    }

    return final


# ----------------------------- MAIN -----------------------------

def main():
    corpus = load_corpus()
    questions = load_test_questions()

    all_results = []

    for name, (size, overlap) in CHUNK_STRATEGIES.items():
        print(f"\n=== Evaluating Strategy: {name} ===")
        vectordb = build_chroma(corpus, name, size, overlap)
        res = evaluate_strategy(name, vectordb, questions)
        all_results.append(res)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # analysis markdown
    best = max(all_results, key=lambda x: x["avg_hit"] + x["avg_rougeL"])

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Evaluation Results\n\n")
        for r in all_results:
            f.write(f"## Strategy: {r['strategy']}\n")
            f.write(f"- Hit Rate: {r['avg_hit']:.3f}\n")
            f.write(f"- MRR: {r['avg_mrr']:.3f}\n")
            f.write(f"- Precision@K: {r['avg_precision_at_k']:.3f}\n")
            f.write(f"- ROUGE-L: {r['avg_rougeL']:.3f}\n")
            f.write(f"- BLEU: {r['avg_bleu']:.3f}\n")
            f.write(f"- Cosine: {r['avg_cosine']:.3f}\n")
            f.write(f"- Relevance: {r['avg_relevance']:.3f}\n")
            f.write(f"- Faithfulness: {r['avg_faithfulness']:.3f}\n\n")

        f.write("## Best Strategy\n")
        f.write(f"**{best['strategy']}** based on hit rate + ROUGE-L.\n\n")

        f.write("## Common Failure Modes\n")
        f.write("- Wrong context retrieved\n")
        f.write("- Hallucinations in unanswerable questions\n")
        f.write("- Missing details in long factual answers\n")

    print("\nDONE. Results saved → test_results.json & results_analysis.md")


if __name__ == "__main__":
    main()
