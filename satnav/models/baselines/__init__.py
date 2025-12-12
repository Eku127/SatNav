"""SatNav baseline models.

This package contains baseline model implementations for VLN,
including Seq2Seq, CMA, and other standard architectures.
"""

from satnav.models.baselines.seq2seq_policy import Seq2SeqPolicy, Seq2SeqNet

__all__ = [
    "Seq2SeqPolicy",
    "Seq2SeqNet",
]

