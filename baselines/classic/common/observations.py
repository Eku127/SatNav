"""Observation preprocessing shared by Seq2Seq and CMA evaluation."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from satnav.utils.build_vocab import VocabDict, tokenize_instruction_in_observation


class InstructionObservationTransform:
    """Tokenize SatNav instructions with the exact training vocabulary."""

    def __init__(self, vocab: VocabDict, max_length: Optional[int] = None) -> None:
        self.vocab = vocab
        self.max_length = max_length

    @classmethod
    def from_file(
        cls, path: str, max_length: Optional[int] = None
    ) -> "InstructionObservationTransform":
        return cls(VocabDict.load(path), max_length=max_length)

    def __call__(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        return tokenize_instruction_in_observation(
            dict(observation),
            self.vocab,
            max_length=self.max_length,
            output_format="numpy",
        )
