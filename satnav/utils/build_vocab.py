"""Vocabulary builder for SatNav datasets.

This module provides utilities to build vocabulary from SatNav dataset
instruction texts. Supports both extracting from dataset or using
provided vocabulary.

Reference:
    - Habitat-Lab: habitat/datasets/utils.py (tokenize and VocabDict classes)
    - Plan: Section 5.1
"""

import json
import re
from collections import Counter
from typing import List, Dict, Optional


# Tokenization regex (same as Habitat-Lab)
SENTENCE_SPLIT_REGEX = re.compile(r"([^\w-]+)")


def tokenize(
    sentence: str,
    regex=SENTENCE_SPLIT_REGEX,
    keep=("'s",),
    remove=(",", "?"),
) -> List[str]:
    """Tokenize a sentence using the same rules as Habitat-Lab.
    
    Args:
        sentence: Input sentence to tokenize
        regex: Regular expression for splitting
        keep: Tokens to keep (with space before them)
        remove: Tokens to remove
    
    Returns:
        List of tokens
    
    Reference:
        Habitat-Lab: habitat/datasets/utils.py (tokenize function, lines 31-44)
    """
    sentence = sentence.lower()
    
    for token in keep:
        sentence = sentence.replace(token, " " + token)
    
    for token in remove:
        sentence = sentence.replace(token, "")
    
    tokens = regex.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def build_vocab_from_episodes(
    episodes: List[Dict],
    min_count: int = 1,
) -> Dict[str, int]:
    """Build vocabulary from episode instructions.
    
    Args:
        episodes: List of episode dictionaries, each containing
                 an "instruction" field with "instruction_text"
        min_count: Minimum frequency for a word to be included
    
    Returns:
        Dictionary mapping word to index
        Format:
            - Index 0: "<pad>" (PAD token)
            - Index 1: "<unk>" (UNK token)
            - Index 2+: Vocabulary words (sorted by frequency)
    
    Reference:
        Habitat-Lab: habitat/datasets/utils.py (VocabFromText class, lines 143-176)
    """
    token_counter = Counter()
    
    # Extract all instruction texts and tokenize
    for episode in episodes:
        instruction_text = episode["instruction"]["instruction_text"]
        tokens = tokenize(instruction_text)
        token_counter.update(tokens)
    
    # Filter by minimum count
    token_list = []
    for token in token_counter:
        if token_counter[token] >= min_count:
            token_list.append(token)
    
    # Sort by frequency (most frequent first)
    token_list = sorted(token_list, key=lambda t: token_counter[t], reverse=True)
    
    # Create word to index mapping
    # Index 0: PAD, Index 1: UNK
    word2idx = {"<pad>": 0, "<unk>": 1}
    for idx, token in enumerate(token_list, start=2):
        word2idx[token] = idx
    
    return word2idx


def load_vocab_from_dataset(dataset_path: str) -> Optional[Dict[str, int]]:
    """Load vocabulary from dataset if provided.
    
    Args:
        dataset_path: Path to the dataset JSON file
    
    Returns:
        Dictionary mapping word to index if vocabulary is provided,
        otherwise None
    """
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    # Check if dataset provides instruction_vocab
    if "instruction_vocab" in data and "word_list" in data["instruction_vocab"]:
        word_list = data["instruction_vocab"]["word_list"]
        # Create word to index mapping
        word2idx = {"<pad>": 0, "<unk>": 1}
        for idx, word in enumerate(word_list, start=2):
            word2idx[word] = idx
        return word2idx
    
    return None


def build_vocab(
    dataset_path: str,
    output_path: str,
    min_count: int = 1,
    use_dataset_vocab: bool = True,
):
    """Build vocabulary from SatNav dataset.
    
    Strategy (flexible approach from plan):
    1. If dataset provides instruction_vocab and use_dataset_vocab=True, use it
    2. Otherwise, extract vocabulary from all episode instructions
    
    Args:
        dataset_path: Path to the dataset JSON file
        output_path: Path to save the vocabulary JSON file
        min_count: Minimum frequency for a word to be included (for extraction)
        use_dataset_vocab: Whether to use dataset-provided vocabulary if available
    
    Output format:
        JSON file with:
        {
            "word2idx": {"<pad>": 0, "<unk>": 1, "word1": 2, ...},
            "idx2word": ["<pad>", "<unk>", "word1", ...],
            "vocab_size": N
        }
    """
    # Load dataset
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    # Try to use dataset vocabulary if requested
    if use_dataset_vocab:
        word2idx = load_vocab_from_dataset(dataset_path)
        if word2idx is not None:
            print(f"Using vocabulary from dataset (vocab_size={len(word2idx)})")
        else:
            print("No vocabulary found in dataset, extracting from episodes")
            word2idx = build_vocab_from_episodes(data["episodes"], min_count)
    else:
        print("Extracting vocabulary from episodes")
        word2idx = build_vocab_from_episodes(data["episodes"], min_count)
    
    # Create reverse mapping
    idx2word = [""] * len(word2idx)
    for word, idx in word2idx.items():
        idx2word[idx] = word
    
    # Save vocabulary
    vocab_data = {
        "word2idx": word2idx,
        "idx2word": idx2word,
        "vocab_size": len(word2idx),
    }
    
    with open(output_path, "w") as f:
        json.dump(vocab_data, f, indent=2)
    
    print(f"Vocabulary saved to {output_path}")
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Sample words: {idx2word[2:min(12, len(idx2word))]}")


class VocabDict:
    """Vocabulary dictionary wrapper for tokenization.
    
    This class provides a simple interface for converting tokens to indices,
    similar to Habitat-Lab's VocabDict class.
    
    Attributes:
        word2idx: Dictionary mapping word to index
        idx2word: List mapping index to word
        vocab_size: Size of vocabulary
    """
    
    def __init__(self, word2idx: Dict[str, int]):
        """Initialize VocabDict from word2idx mapping.
        
        Args:
            word2idx: Dictionary mapping word to index
        """
        self.word2idx = word2idx
        self.idx2word = [""] * len(word2idx)
        for word, idx in word2idx.items():
            self.idx2word[idx] = word
        self.vocab_size = len(word2idx)
    
    def tokens_to_indices(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens to list of indices.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of token indices (UNK for unknown tokens)
        """
        unk_idx = self.word2idx.get("<unk>", 1)
        return [self.word2idx.get(token, unk_idx) for token in tokens]
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    @classmethod
    def load(cls, vocab_path: str) -> "VocabDict":
        """Load VocabDict from JSON file.
        
        Args:
            vocab_path: Path to vocabulary JSON file
            
        Returns:
            VocabDict instance
        """
        with open(vocab_path, "r") as f:
            vocab_data = json.load(f)
        return cls(vocab_data["word2idx"])
    
    def save(self, vocab_path: str) -> None:
        """Save VocabDict to JSON file.
        
        Args:
            vocab_path: Path to save vocabulary JSON file
        """
        vocab_data = {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "vocab_size": self.vocab_size,
        }
        with open(vocab_path, "w") as f:
            json.dump(vocab_data, f, indent=2)


def build_vocab_from_dataset(dataset_path: str, min_count: int = 1) -> VocabDict:
    """Build vocabulary from dataset and return VocabDict.
    
    This is a convenience function that combines load_vocab_from_dataset
    and build_vocab_from_episodes to create a VocabDict.
    
    Args:
        dataset_path: Path to dataset JSON file
        min_count: Minimum frequency for a word to be included
        
    Returns:
        VocabDict instance
    """
    # Try to load from dataset first
    word2idx = load_vocab_from_dataset(dataset_path)
    
    if word2idx is None:
        # Extract from episodes
        with open(dataset_path, "r") as f:
            data = json.load(f)
        word2idx = build_vocab_from_episodes(data["episodes"], min_count)
    
    return VocabDict(word2idx)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build vocabulary from SatNav dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output vocabulary JSON file",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum word frequency to include (default: 1)",
    )
    parser.add_argument(
        "--no-dataset-vocab",
        action="store_true",
        help="Ignore dataset vocabulary and extract from episodes",
    )
    
    args = parser.parse_args()
    
    build_vocab(
        dataset_path=args.dataset,
        output_path=args.output,
        min_count=args.min_count,
        use_dataset_vocab=not args.no_dataset_vocab,
    )

