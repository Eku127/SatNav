from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    """Discretize normalized actions to OpenFly-style tail-vocab tokens."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = float(min_action)
        self.max_action = float(max_action)
        self.bins = np.linspace(self.min_action, self.max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_token_begin_idx = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        token_ids = self.encode_action_token_ids(action)
        if token_ids.ndim == 1:
            return self.tokenizer.decode(token_ids.tolist())
        return self.tokenizer.batch_decode(token_ids.tolist())

    def encode_action_token_ids(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            a_min=self.min_action,
            a_max=self.max_action,
        )
        discretized_action = np.digitize(action, self.bins)
        token_ids = self.tokenizer.vocab_size - discretized_action
        if discretized_action.ndim == 1:
            return np.asarray(token_ids, dtype=np.int64)
        return np.asarray(token_ids, dtype=np.int64)

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1,
            a_min=0,
            a_max=self.bin_centers.shape[0] - 1,
        )
        return self.bin_centers[discretized_actions]
