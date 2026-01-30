def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    reference_bins = [count / sum(reference_counts) for count in reference_counts]
    production_bins = [count / sum(production_counts) for count in production_counts]

    tvd = sum([abs(rf - pr) for rf, pr in zip(reference_bins, production_bins)]) * 0.5
    return {'score': tvd, 'drift_detected': (tvd > threshold)}

