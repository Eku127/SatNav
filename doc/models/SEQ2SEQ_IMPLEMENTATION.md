# Seq2Seq Model Structure

This document summarizes the current Seq2Seq baseline implemented in
`satnav/models/baselines/seq2seq_policy.py`.

Seq2Seq is the simpler recurrent VLN baseline. It encodes the instruction and
current RGB observation, optionally adds the previous action embedding, and uses
one recurrent state encoder to produce an action distribution.

## Inputs

- `instruction`: tokenized instruction, shape `[B, T]`
- `rgb`: RGB observation, shape `[B, H, W, 3]`
- `prev_actions`: previous discrete action, shape `[B, 1]`
- `masks`: episode-continuation mask, shape `[B, 1]`

The action space is:

```text
STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
```

## Architecture

```text
instruction tokens
  -> InstructionEncoder
  -> instruction embedding [B, 128]

rgb image
  -> TorchVisionResNet50
  -> RGB embedding [B, 256]

previous action
  -> Embedding(num_actions + 1, 32)
  -> previous-action embedding [B, 32]

[instruction, rgb, previous action]
  -> concat
  -> GRU state encoder [B, 512]
  -> categorical action head
  -> action logits [B, 4]
```

## Components

### Instruction Encoder

- Class: `satnav.models.encoders.instruction_encoder.InstructionEncoder`
- Default type: LSTM
- Direction: unidirectional
- Embedding size: `50`
- Hidden size: `128`
- Output mode: final state only
- Output shape: `[B, 128]`

The encoder supports pretrained GloVe-style embeddings or random
initialization. Current training configs use pretrained embeddings generated
for the SatNav vocabulary.

### RGB Encoder

- Class: `satnav.models.encoders.visual_encoder.TorchVisionResNet50`
- Backbone: ImageNet-pretrained ResNet-50
- Spatial output: disabled
- Output size: `256`
- Default training mode: frozen backbone
- Output shape: `[B, 256]`

Seq2Seq uses global visual features, so it does not perform spatial attention.

### Previous Action Embedding

- Enabled by `MODEL.SEQ2SEQ.use_prev_action`
- Embedding size: `32`
- The implementation shifts previous actions by `+1` and uses `masks` so that
  reset timesteps use the zero embedding slot.

### State Encoder

- Class: `satnav.models.encoders.rnn_state_encoder.build_rnn_state_encoder`
- Default type: GRU
- Hidden size: `512`
- Number of recurrent layers: `1`
- Initial state shape: `[1, B, 512]`

The state encoder receives the concatenated instruction, RGB, and optional
previous-action features.

### Action Head

- Class: `satnav.models.base.CategoricalNet`
- Input size: `512`
- Output size: `4`
- Produces a categorical distribution over the four SatNav actions.

## Notes

- Seq2Seq has no depth encoder.
- Seq2Seq has no explicit cross-modal or spatial attention.
- The current main training path uses `offline_trainer` with pre-rendered
  trajectory images; model structure is independent of that trainer choice.
- The current default training config is `configs/baselines/seq2seq_offline_train.yaml`.
- The current eval config is `configs/baselines/seq2seq_eval.yaml`.
