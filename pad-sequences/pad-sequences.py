import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    if N == 0:
        L = max_len if max_len else 0
    else:
        L = max_len if max_len else max((len(seq) for seq in seqs), default=0) 

    padded_seqs = np.full((N,L), pad_value)

    for i, seq in enumerate(seqs):
        seq_len = min(L, len(seq))
        padded_seqs[i, :len(seq)] = seq[:seq_len]

    return padded_seqs