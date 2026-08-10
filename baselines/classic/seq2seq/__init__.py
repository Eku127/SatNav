"""Seq2Seq baseline compatibility exports.

Importing this package intentionally opts into the classic PyTorch extra.
Existing ``satnav.models.baselines.seq2seq_policy`` imports remain supported so
old checkpoints and scripts retain their state-dict names.
"""

from satnav.models.baselines.seq2seq_policy import Seq2SeqNet, Seq2SeqPolicy
from baselines.classic.seq2seq.factory import build_seq2seq_adapter

__all__ = ["Seq2SeqNet", "Seq2SeqPolicy", "build_seq2seq_adapter"]
