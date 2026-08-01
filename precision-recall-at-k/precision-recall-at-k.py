def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    recommended_k = set(recommended[:k])
    relevant_set = set(relevant)
    inter = recommended_k.intersection(relevant_set)
    hits = len(inter)
    return [hits/k, hits/len(relevant)]