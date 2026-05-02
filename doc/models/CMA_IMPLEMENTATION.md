# CMA Model Structure

This document summarizes the current Cross-Modal Attention model implemented in
`satnav/models/baselines/cma_policy.py`.

CMA extends the recurrent VLN baseline with instruction-conditioned attention
over spatial RGB features. It uses two recurrent state encoders: the first
builds a visual navigation state, and the second integrates the attended text
and attended visual features before action prediction.

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
  -> bidirectional InstructionEncoder
  -> instruction tokens [B, 256, T]

rgb image
  -> TorchVisionResNet50 spatial encoder
  -> RGB spatial features [B, 2112, 4, 4]
  -> flatten spatial grid [B, 2112, 16]

previous action
  -> Embedding(num_actions + 1, 32)
  -> previous-action embedding [B, 32]

RGB spatial features
  -> average pool + linear
  -> RGB pooled feature [B, 256]

[RGB pooled, previous action]
  -> first GRU state encoder
  -> state [B, 512]

state queries instruction tokens
  -> text-state attention
  -> attended text [B, 256]

attended text queries RGB spatial features
  -> text-RGB attention
  -> attended RGB [B, 256]

[state, attended text, attended RGB, previous action]
  -> linear compression [B, 512]
  -> second GRU state encoder [B, 512]
  -> categorical action head
  -> action logits [B, 4]
```

## Components

### Instruction Encoder

- Class: `satnav.models.encoders.instruction_encoder.InstructionEncoder`
- Default type: LSTM
- Direction: bidirectional
- Embedding size: `50`
- Hidden size: `128` per direction
- Output mode: all timesteps
- Output shape: `[B, 256, T]`

CMA requires all instruction timesteps because attention is computed over the
instruction sequence.

### RGB Encoder

- Class: `satnav.models.encoders.visual_encoder.TorchVisionResNet50`
- Backbone: ImageNet-pretrained ResNet-50
- Spatial output: enabled
- Spatial grid: `4 x 4`
- Feature channels: `2048 + 64` positional channels
- Output shape: `[B, 2112, 4, 4]`

The `64` extra channels come from learned spatial embeddings over the `4 x 4`
grid.

### Previous Action Embedding

- Embedding size: `32`
- The current CMA implementation always creates and uses this embedding.
- Previous actions are shifted by `+1` and combined with `masks` so that reset
  timesteps use the zero embedding slot.

### First State Encoder

- Input: pooled RGB feature `[B, 256]` plus previous-action embedding `[B, 32]`
- Default type: GRU
- Hidden size: `512`
- Number of recurrent layers: `1`
- Output shape: `[B, 512]`

This stage creates the state used to query the instruction.

### Cross-Modal Attention

CMA uses two attention steps:

1. Text-state attention:
   - Query: first state encoder output `[B, 512]`
   - Key/value: instruction tokens `[B, 256, T]`
   - Output: attended text feature `[B, 256]`

2. Text-RGB attention:
   - Query: attended text feature `[B, 256]`
   - Key/value: spatial RGB features projected from `[B, 2112, 16]`
   - Output: attended RGB feature `[B, 256]`

Both attention operations use scaled dot-product attention.

### Second State Encoder

- Input: `[state, attended text, attended RGB, previous action]`
- Raw input size: `512 + 256 + 256 + 32 = 1056`
- Compression: linear layer to `512`
- Default type: GRU
- Hidden size: `512`
- Number of recurrent layers: `1`
- Output shape: `[B, 512]`

### Action Head

- Class: `satnav.models.base.CategoricalNet`
- Input size: `512`
- Output size: `4`
- Produces a categorical distribution over the four SatNav actions.

## Recurrent State

CMA has two recurrent encoders, so its initial state has two layers:

```text
[2, B, 512]
```

Layer `0` is used by the first state encoder. Layer `1` is used by the second
state encoder.

## Notes

- CMA has no depth encoder.
- CMA has no progress monitor.
- The current main training path uses `offline_trainer` with pre-rendered
  trajectory images; model structure is independent of that trainer choice.
- The current default training config is `configs/baselines/cma_offline_train.yaml`.
- The current eval config is `configs/baselines/cma_eval.yaml`.
