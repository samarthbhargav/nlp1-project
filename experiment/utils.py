def jaccard_index(tokens1, tokens2):
    tokens1, tokens2 = set(tokens1), set(tokens2)
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    if union == 0:
        raise ValueError("Union cannot have 0 elements")
    return intersection / union


if __name__ == "__main__":
    print(jaccard_index(["a", "b"], ["a", "b"]))
