"""
Demo: bi-encoder retrieval -> rerank (CrossEncoder)
Пример: предварительный отбор 10 лучших кандидатов с помощью bi-encoder,
затем rerank с помощью cross-encoder и измерение MRR@k на обоих этапах.
"""

import time
import numpy as np
import torch
import csv
from sentence_transformers import SentenceTransformer, CrossEncoder, util

queries = []
with open('queries/queries_random.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        queries.append(row)

corpus = []
try:
    with open("corpus.txt", "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(line.strip())

    if len(corpus) == 0:
        raise ValueError("corpus.txt is empty. Yu may try to fill it with some example documents.")
except (ValueError, FileNotFoundError) as e:
    print(e.strerror)
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Downloading bi-encoder ... it may take more time on the first run")
bi_encoder = SentenceTransformer("DiTy/bi-encoder-russian-msmarco", device=device)

t0 = time.time()
corpus_embeddings = bi_encoder.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
t1 = time.time()
print(f"Corpus encoded: {len(corpus)} documents, time {t1-t0:.2f}s")

print("Downloading cross-encoder (reranker) ... it may take more time on the first run")
reranker = CrossEncoder("DiTy/cross-encoder-russian-msmarco", device=device, max_length=512)


def mrr_at_k(ranked_lists, gold_indices, k=10):
    rr_list = []
    for ranks, gold in zip(ranked_lists, gold_indices):
        for i, doc_idx in enumerate(ranks[:k], start=1):
            if doc_idx == gold:
                rr_list.append(1.0 / i)
                break
        else:
            rr_list.append(0.0)
    return float(np.mean(rr_list))

TOP_K_BI = 20
BATCH_SIZE_RERANK = 8

ranked_lists_bi = []
ranked_lists_rerank = []
gold_list = []
for q_text, gold_idx in queries:
    print("\n" + "="*60)
    print(f"Query: {q_text} (gold idx: {gold_idx})")

    t_bi0 = time.time()
    q_emb = bi_encoder.encode(q_text, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, corpus_embeddings)[0]

    topk = torch.topk(scores, k=min(TOP_K_BI, len(corpus))).indices.cpu().numpy().tolist()
    ranked_lists_bi.append(topk)
    t_bi1 = time.time()
    print(f"bi-encoder top{len(topk)} indices: {topk}")
    print(f"Time: bi-encoder retrieval {t_bi1 - t_bi0:.3f}s")

    pairs = [[q_text, corpus[idx]] for idx in topk]

    t_r0 = time.time()
    rerank_scores = reranker.predict(pairs, batch_size=BATCH_SIZE_RERANK)
    t_r1 = time.time()

    scored = list(zip(topk, rerank_scores))
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    reranked_indices = [idx for idx, sc in scored_sorted]

    PRINT_FIRST = 5
    printed = 0
    print("Reranked (index : score) :")
    for idx, sc in scored_sorted:
        if printed == PRINT_FIRST:
            break
        snippet = corpus[idx]
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        print(f"  {idx} : {sc:.4f} --> {snippet}")
        printed += 1

    print(f"Time: rerank {t_r1 - t_r0:.3f}s (batch_size={BATCH_SIZE_RERANK})")
    total_time = (t_bi1 - t_bi0) + (t_r1 - t_r0)
    print(f"Total time (bi + rerank): {total_time:.3f}s")

    ranked_lists_rerank.append(reranked_indices)
    gold_list.append(int(gold_idx))

mrr_bi = mrr_at_k(ranked_lists_bi, gold_list, k=TOP_K_BI)
mrr_rerank = mrr_at_k(ranked_lists_rerank, gold_list, k=TOP_K_BI)

print(f"\nMRR@{TOP_K_BI} (bi-encoder only): {mrr_bi:.4f}")
print(f"MRR@{TOP_K_BI} (bi-encoder + reranker): {mrr_rerank:.4f}")
print(f"Δ MRR: {mrr_rerank - mrr_bi:+.4f}")
