import math

def calculate_retrieval_metrics(
    evidence,
    retrieved_memories,
    k_values=(3, 5, 10),
    add_redundancy=True,
):
    """
    Evidence-coverage metrics:
    - Recall@K: unique gold evidence covered by top-K
    - NDCG@K: ranking quality w.r.t. first-time coverage of new gold evidence
      (duplicates of the same evidence do NOT increase gain)
    Optionally:
    - effective_evidence@K: how many unique gold evidence covered in top-K
    - redundancy@K: 1 - effective_hits / K  (how much top-K is wasted on repeats/irrelevant)
    """
    # Edge: no gold evidence
    if not evidence:
        out = {}
        for k in k_values:
            out[f"recall@{k}"] = 0.0
            out[f"ndcg@{k}"] = 0.0
            if add_redundancy:
                out[f"effective_evidence@{k}"] = 0
                out[f"redundancy@{k}"] = 1.0  # all wasted
        return out

    gold = set(evidence)
    total_relevant = len(gold)

    # Ensure sorted by score descending (your caller already does this, but keep safe)
    retrieved_memories = [m for m in retrieved_memories if isinstance(m, dict)]
    # if scores exist, keep; otherwise assume already sorted
    if retrieved_memories and "score" in retrieved_memories[0]:
        retrieved_memories = sorted(retrieved_memories, key=lambda x: x.get("score", 0), reverse=True)

    metrics = {}

    for k in k_values:
        topk = retrieved_memories[:k]

        covered = set()   # covered gold evidence ids so far (for recall + ndcg gain)
        dcg = 0.0

        for i, mem in enumerate(topk):
            source_ids = mem.get("source_ids", [])
            # which NEW gold evidence does this memory introduce?
            new_hits = (set(source_ids) & gold) - covered

            if new_hits:
                # gain=1 only once per newly covered evidence-set at this rank (simple + stable)
                dcg += 1.0 / math.log2(i + 2)
                covered |= new_hits

        # Recall@K (unique evidence coverage)
        recall = len(covered) / total_relevant if total_relevant else 0.0
        metrics[f"recall@{k}"] = recall

        # IDCG@K: ideal is covering as many distinct gold evidences as possible ASAP
        num_ideal = min(k, total_relevant)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(num_ideal))
        metrics[f"ndcg@{k}"] = (dcg / idcg) if idcg > 0 else 0.0

        if add_redundancy:
            metrics[f"effective_evidence@{k}"] = len(covered)
            metrics[f"redundancy@{k}"] = 1.0 - (len(covered) / k if k > 0 else 0.0)

    return metrics