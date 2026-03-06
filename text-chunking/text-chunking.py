def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    step_size = chunk_size - overlap
    tokens_list = []
    for i in range(0, len(tokens), step_size):
        chunk = tokens[i: i + chunk_size]
        tokens_list.append(chunk)

        if i + chunk_size >= len(tokens): break

    return tokens_list