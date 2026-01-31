def confusion_matrix(y_true, y_pred,):

    y_true = list(y_true)
    y_pred = list(y_pred)
    labels = list(sorted(set(y_true) | set(y_pred)))
    k = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}

    cm = [[0] * k for _ in range(k)]

    for t, p in zip(y_true, y_pred):
        if t not in idx or p not in idx:
            raise ValueError(f"Label not in labels list: true={t}, pred={p}")
        cm[idx[t]][idx[p]] += 1

    return cm, labels

def tp_fp_fn_tn_from_cm(cm, c):
    k = len(cm)
    total = sum(sum(row) for row in cm)

    tp = cm[c][c]
    fn = sum(cm[c][j] for j in range(k)) - tp
    fp = sum(cm[i][c] for i in range(k)) - tp
    tn = total - tp - fn - fp
    return tp, fp, fn, tn

def safe_div(n, d):
    return n / d if d else 0.0

def compute_monitoring_metrics(system_type, y_true, y_pred):
    y_true, y_pred = list(y_true), list(y_pred)

    metrics = []

    if system_type == "classification":

        cm, labels = confusion_matrix(y_true, y_pred)
        tp, fp, fn, tn = tp_fp_fn_tn_from_cm(cm, c=1)

        n = tp + fp + fn + tn
        acc = safe_div(tp+tn, n)
        precision = safe_div(tp, tp+fp)
        recall = safe_div(tp, tp+fn)
        f1 = safe_div(2*precision*recall, precision+recall)

        metrics = [
            ("accuracy", acc),
            ("precision", precision),
            ("recall", recall),
            ("f1", f1)
        ]
    
    elif system_type == "regression":
        n = len(y_true)
        abs_err = 0.0
        sq_err = 0.0

        for yt, yp in zip(y_true, y_pred):
            err = yt - yp
            abs_err += abs(err)
            sq_err += err * err

        mae = safe_div(abs_err, n)
        rmse = safe_div(sq_err, n) ** 0.5

        metrics = [
            ("mae", mae),
            ("rmse", rmse)
        ]

    elif system_type == "ranking":
        pairs = list(zip(y_true, y_pred))
        pairs.sort(key=lambda x: x[1], reverse=True)

        top3 = pairs[:3]
        relevant_in_top3 = sum(1 for rel, _ in top3 if rel == 1)
        total_relevant = sum(1 for rel in y_true if rel == 1) 
        precision_at_3 = safe_div(relevant_in_top3, 3)
        recall_at_3 = safe_div(relevant_in_top3, total_relevant)
        metrics = [
            ("precision_at_3", precision_at_3),
            ("recall_at_3", recall_at_3),
        ]

    else:
        raise ValueError("system_type must be one of: classification, regression, ranking")

    metrics.sort(key=lambda x: x[0]) # sort alphabetically
    return metrics
