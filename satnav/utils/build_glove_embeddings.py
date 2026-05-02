"""GloVe embedding builder for SatNav.

This module provides utilities to build GloVe-based word embeddings
for SatNav vocabulary. The output format is compatible with VLN-CE.

Embedding format (VLN-CE compatible):
    - Index 0: PAD token (all zeros)
    - Index 1: UNK token (mean of all word embeddings)
    - Index 2+: Word embeddings from GloVe

Reference:
    - VLN-CE: vlnce_baselines/models/encoders/instruction_encoder.py
             (_load_embeddings method, lines 51-61)
    - Plan: Section 5.2
"""

import gzip
import json
import os
import numpy as np
from typing import Dict


def load_glove_vectors(glove_path: str, embedding_dim: int = 50) -> Dict[str, np.ndarray]:
    """Load GloVe word vectors from text file.
    
    Args:
        glove_path: Path to GloVe text file (e.g., glove.6B.50d.txt)
        embedding_dim: Expected embedding dimension (default: 50)
    
    Returns:
        Dictionary mapping word to embedding vector
    
    GloVe file format:
        Each line: word value1 value2 ... valueN
        Example: "the 0.418 0.24968 -0.41242 ..."
    """
    glove_vectors = {}
    
    print(f"Loading GloVe vectors from {glove_path}...")
    with open(glove_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            
            if len(vector) != embedding_dim:
                print(f"Warning: Line {line_num}: Expected {embedding_dim}d, "
                      f"got {len(vector)}d for word '{word}'. Skipping.")
                continue
            
            glove_vectors[word] = vector
            
            if line_num % 50000 == 0:
                print(f"  Loaded {line_num} vectors...")
    
    print(f"Loaded {len(glove_vectors)} GloVe vectors")
    return glove_vectors


def build_embeddings(
    vocab_path: str,
    glove_path: str,
    output_path: str,
    embedding_dim: int = 50,
):
    """Build embedding matrix from vocabulary and GloVe vectors.
    
    Output format (VLN-CE compatible):
        - embeddings.json.gz containing a 2D list of shape [vocab_size, embedding_dim]
        - Index 0: PAD token - [0.0, 0.0, ..., 0.0]
        - Index 1: UNK token - mean of all found word embeddings
        - Index 2+: Word embeddings from GloVe (or UNK if not found)
    
    Args:
        vocab_path: Path to vocabulary JSON file (from build_vocab.py)
        glove_path: Path to GloVe text file
        output_path: Path to output embeddings.json.gz file
        embedding_dim: Embedding dimension (default: 50 for GloVe 6B.50d)
    
    Reference:
        VLN-CE embedding format:
            - vlnce_baselines/models/encoders/instruction_encoder.py (lines 51-61)
        Why UNK is averaged: https://bit.ly/3u3hkYg
    """
    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)
    
    word2idx = vocab_data["word2idx"]
    idx2word = vocab_data["idx2word"]
    vocab_size = vocab_data["vocab_size"]
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Load GloVe vectors
    glove_vectors = load_glove_vectors(glove_path, embedding_dim)
    
    # Build embedding matrix
    print("Building embedding matrix...")
    embeddings = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    
    # Collect all embeddings for computing mean (for UNK token)
    found_embeddings = []
    not_found_words = []
    
    for word, idx in word2idx.items():
        if idx == 0:
            # PAD token: all zeros (already initialized)
            continue
        elif idx == 1:
            # UNK token: will be set to mean later
            continue
        else:
            # Regular word: try to find in GloVe
            if word in glove_vectors:
                embeddings[idx] = glove_vectors[word]
                found_embeddings.append(glove_vectors[word])
            else:
                not_found_words.append(word)
    
    # Set UNK token to mean of all found embeddings
    if found_embeddings:
        unk_embedding = np.mean(found_embeddings, axis=0)
        embeddings[1] = unk_embedding
        
        # Set not found words to UNK embedding
        for word in not_found_words:
            idx = word2idx[word]
            embeddings[idx] = unk_embedding
    else:
        print("Warning: No words found in GloVe. UNK will be zero vector.")
    
    # Print statistics
    found_count = len(found_embeddings)
    not_found_count = len(not_found_words)
    coverage = (found_count / (vocab_size - 2)) * 100 if vocab_size > 2 else 0
    
    print(f"\nEmbedding statistics:")
    print(f"  Total vocabulary: {vocab_size}")
    print(f"  Found in GloVe: {found_count} ({coverage:.1f}%)")
    print(f"  Not found (using UNK): {not_found_count}")
    
    if not_found_words and len(not_found_words) <= 20:
        print(f"  Not found words: {not_found_words}")
    elif not_found_words:
        print(f"  Sample not found words: {not_found_words[:20]}")
    
    # Save embeddings in VLN-CE compatible format (gzipped JSON)
    print(f"\nSaving embeddings to {output_path}...")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    embeddings_list = embeddings.tolist()
    
    with gzip.open(output_path, "wt") as f:
        json.dump(embeddings_list, f)
    
    print("Done!")
    print(f"\nOutput format:")
    print(f"  Shape: [{vocab_size}, {embedding_dim}]")
    print(f"  Index 0 (PAD): all zeros")
    print(f"  Index 1 (UNK): mean of all embeddings")
    print(f"  Index 2+: word embeddings")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build GloVe embeddings for SatNav vocabulary"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to vocabulary JSON file (from build_vocab.py)",
    )
    parser.add_argument(
        "--glove",
        type=str,
        required=True,
        help="Path to GloVe text file (e.g., glove.6B.50d.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output embeddings.json.gz file",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=50,
        help="Embedding dimension (default: 50 for GloVe 6B.50d)",
    )
    
    args = parser.parse_args()
    
    build_embeddings(
        vocab_path=args.vocab,
        glove_path=args.glove,
        output_path=args.output,
        embedding_dim=args.embedding_dim,
    )
