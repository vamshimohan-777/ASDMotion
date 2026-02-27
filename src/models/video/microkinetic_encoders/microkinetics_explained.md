# MicroKinetic Encoder Deep Technical Explanation

Source file: `src/models/video/microkinetic_encoders/microkinetics.py`

## Notation and Assumptions

- `B`: batch size
- `T`: number of frame-level timesteps in the input sequence
- `D`: input feature width (`d_in`, default 768)
- `C`: conv channel width (`conv_channels`, default 256)
- `K`: selected events per sample (`min(K_max, T)`)
- `K_max`: fixed output event budget (default 32)
- `M`: number of event types (`NUM_EVENT_TYPES`)
- `S`: number of scalar outputs (`num_scalars`, default 8)

Main input tensors:
- `features`: shape `[B, T, D]`
- `mask`: shape `[B, T]`, true for valid timesteps
- optional `conv_features`: shape `[B, T, C]`
- optional `timestamps`: shape `[B, T]`
- optional `delta_t`: shape `[B, T]`

Main outputs:
- `tokens`: `[B, K_max, d_model]`
- `attn_mask`: `[B, K_max]`
- `time_positions`: `[B, K_max]`
- `event_type_id`: `[B, K_max]`
- `token_conf`: `[B, K_max]`
- `event_scalars`: `[B, K_max, S]`
- `delta_t`: `[B, K_max]`

---

## 1. Imports and Global Constant

### `NUM_EVENT_TYPES`

- Imported from `event_types.py`.
- It is the taxonomy cardinality (the number of discrete event classes).
- Used to size `type_head` output.

If this value changes:
- `type_head` output dimension changes.
- Any pretrained checkpoint for `type_head` becomes incompatible unless remapped.

---

## 2. Class: `TemporalConvBlock`

```python
class TemporalConvBlock(nn.Module):
```

A reusable 1D temporal feature extractor block.

### Constructor (`__init__`)

#### Line: `padding = kernel_size // 2`
- Input shape impact: none (hyperparameter setup).
- Purpose: keeps temporal length roughly unchanged for odd kernels.
- Why: if `kernel_size` is odd (3/5/7), output length remains `T`.
- Alternative: causal padding for autoregressive settings.

#### Layer: `nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)`
- Input shape: `[B, in_ch, T]`
- Output shape: `[B, out_ch, T]` (for odd kernel and this padding)
- Math:
  - `y[b, c, t] = sum_i sum_u W[c, i, u] * x[b, i, t + u - p] + b[c]`
- Why: learns local temporal patterns from frame features.
- If removed: no local motion filtering; downstream sees raw features.
- Alternative: depthwise-separable conv, dilated conv, temporal MLP, attention.

#### Layer: dynamic group count + `nn.GroupNorm(n_groups, out_ch)`
- Input shape: `[B, out_ch, T]`
- Output shape: `[B, out_ch, T]`
- Math: normalize each group per sample using group mean/variance.
- Why: stable training for variable batch sizes (better than BatchNorm when B is small).
- If changed to BatchNorm: can be noisy with small B.
- If removed: higher activation scale drift and less stable optimization.
- Alternative: LayerNorm over channel dimension after transpose.

#### Layer: `nn.GELU()`
- Input/output shape: unchanged `[B, out_ch, T]`
- Math: smooth nonlinear gate, approximately `x * Phi(x)`.
- Why: richer than ReLU around zero; common in transformer-like stacks.
- If replaced with ReLU: cheaper, but less smooth gradients near zero.

#### Layer: `nn.Dropout(dropout)`
- Input/output shape: unchanged `[B, out_ch, T]`
- Math: random Bernoulli masking during training.
- Why: regularization to reduce co-adaptation.
- If removed: more overfitting risk.

### Forward

```python
return self.drop(self.act(self.norm(self.conv(x))))
```

- Input shape: `[B, in_ch, T]`
- Output shape: `[B, out_ch, T]`
- Pipeline: conv -> normalize -> nonlinear -> dropout.
- Gradient flow: standard chain rule through all differentiable ops.

Complexity per block:
- Dominated by Conv1d: `O(B * T * in_ch * out_ch * kernel_size)`.

---

## 3. Class: `MicroKineticEncoder`

This class converts dense framewise features into a fixed-size set of event tokens and metadata.

### Constructor Arguments

- `d_in`: frame feature width.
- `d_model`: output token embedding width for transformer compatibility.
- `K_max`: max number of selected event tokens.
- `num_event_types`: number of event labels.
- `num_scalars`: auxiliary scalar regressions per event.
- `conv_channels`: hidden width for temporal conv features.
- `kernel_sizes`: multi-scale temporal receptive fields.
- `dropout`: regularization for projection blocks.

### State Variables

- `self.d_model`, `self.K_max`, `self.num_scalars` store configuration used in forward.

### Block: `self.conv_branches`

- `ModuleList` of `TemporalConvBlock(d_in, C, ks)` for each kernel size.
- With default `[3, 5, 7]`, produces 3 parallel temporal streams.

Input/output for this block:
- Input: shared `x` `[B, D, T]`
- Each branch output: `[B, C, T]`
- After concat: `[B, C * n_branches, T]`

Why:
- Captures short, medium, long local dynamics simultaneously.

Alternative designs:
- Dilated stack with one branch.
- Temporal Inception block with pooling branch.
- Multi-head self-attention over time directly.

### Block: `self.channel_fuse`

```python
Linear(C*n_branches -> C) -> GELU -> LayerNorm(C) -> Dropout
```

Input/output:
- Input: `[B, T, C*n_branches]`
- Output: `[B, T, C]`

Math:
- Learned affine fusion across branch channel dimension per timestep.

Why:
- Compresses concatenated multi-scale descriptors into fixed-width latent `C`.

Alternative:
- 1x1 Conv1d after concat.
- Gated fusion or attention over branches.

### Block: `self.event_gate`

```python
Linear(C -> C/4) -> GELU -> Linear(C/4 -> 1)
```

Input/output:
- Input: `[B, T, C]`
- Output logits: `[B, T, 1]` -> squeeze to `[B, T]`

Why:
- Scores each timestep for event salience before top-k selection.

Alternative:
- Temperature-softmax sampling, Gumbel-topk, sparsemax, differentiable sorting.

### Block: `self.token_proj`

```python
Linear(C -> d_model) -> GELU -> LayerNorm(d_model) -> Dropout
```

Input/output:
- Input: `[B, K, C]`
- Output: `[B, K, d_model]`

Why:
- Converts selected event features into transformer token space.

### Heads

#### `self.scalar_head = Linear(C -> S)`
- Input: `[B, K, C]`
- Output: `[B, K, S]`
- Use: structured numeric descriptors per event.

#### `self.type_head = Linear(C -> M)`
- Input: `[B, K, C]`
- Output logits: `[B, K, M]`
- `argmax` yields integer event IDs.

#### `self.conf_head = Linear(C -> 1) + Sigmoid`
- Input: `[B, K, C]`
- Output confidence: `[B, K, 1]` then `[B, K]`

### Function: `_init_weights`

- For each `Linear`: Xavier uniform + zero bias.
- For each `Conv1d`: Kaiming normal (`nonlinearity="linear"`) + zero bias.

Why:
- Reasonable variance scaling at startup.
- Zero bias avoids early offset drift.

Potential note:
- Conv block uses GELU after conv; some practitioners might use Kaiming tuned for ReLU-like nonlinearity, or simply Xavier for consistency.

---

## 4. Forward Pass: Line-by-Line

Below uses exact line numbers from `microkinetics.py`.

### Line 101: `B, T, D = features.shape`
- Input: `features [B, T, D]`
- Output: scalars `B,T,D`.
- Why: dynamic sizing for later ops.

### Line 102: `device = features.device`
- No tensor change.
- Note: currently unused in this function.
- If removed: no functional change.

### Lines 104-106: conditional precomputed conv path
- If `conv_features` provided, set `fused = conv_features`.
- Required shape assumption: `[B, T, C]`.
- Why: allows caching/reuse of conv features outside this module.

### Line 107: `x = features.permute(0, 2, 1)`
- Input: `[B, T, D]`
- Output: `[B, D, T]`
- Math: axis reorder.
- Why: `Conv1d` expects channels-first.
- If omitted: conv interprets `T` as channels, incorrect semantics.

### Lines 108-110: branch loop
- Start empty list; each branch computes `conv(x)`.
- Per branch input: `[B, D, T]`
- Per branch output: `[B, C, T]`
- Gradient: sums from all later uses into branch parameters.

### Line 111: `multi_scale = torch.cat(branch_outs, dim=1)`
- Inputs: `n_branches` tensors `[B, C, T]`
- Output: `[B, C*n_branches, T]`
- Why: merge multi-receptive-field information by channel stacking.

### Line 112: `multi_scale = multi_scale.permute(0, 2, 1)`
- Input: `[B, C*n_branches, T]`
- Output: `[B, T, C*n_branches]`
- Why: `Linear` expects feature dim last.

### Line 113: `fused = self.channel_fuse(multi_scale)`
- Input: `[B, T, C*n_branches]`
- Output: `[B, T, C]`
- Why: branch fusion and compression.

### Line 115: `gate_logits = self.event_gate(fused).squeeze(-1)`
- Input: `[B, T, C]`
- MLP output before squeeze: `[B, T, 1]`
- After squeeze: `[B, T]`
- Why: one scalar salience logit per timestep.

### Line 116: mask invalid timesteps
```python
gate_logits = gate_logits.masked_fill(~mask.bool(), float("-inf"))
```
- Input `gate_logits`: `[B, T]`, `mask`: `[B, T]`
- Output: `[B, T]`
- Math: invalid positions set to `-inf`.
- Why: ensures invalid frames cannot be selected by top-k.
- If removed: padded/noise frames may be chosen.

### Line 117: `gate_scores = torch.sigmoid(gate_logits)`
- Input/output: `[B, T]`
- Math: `sigma(z)=1/(1+e^-z)` maps logits to (0,1).
- Note: masked positions become exactly 0 due to sigmoid(-inf).

### Line 119: `K = min(self.K_max, T)`
- Scalar computation.
- Why: prevent top-k requesting more than available timesteps.

### Line 120: `topk_scores, topk_indices = torch.topk(gate_scores, K, dim=1)`
- Input: `[B, T]`
- Outputs:
  - `topk_scores`: `[B, K]`
  - `topk_indices`: `[B, K]`
- Why: pick highest-salience timesteps.
- Complexity: approx `O(B * T log K)` (implementation-dependent).

### Line 121: `sorted_order = topk_indices.sort(dim=1).indices`
- Input: `[B, K]`
- Output: `[B, K]` (permutation indices)
- Why: reorder selected events into temporal order.

### Line 122: `topk_indices = topk_indices.gather(1, sorted_order)`
- Input: both `[B, K]`
- Output: `[B, K]` sorted ascending by time index.

### Line 123: `topk_scores = topk_scores.gather(1, sorted_order)`
- Input/output: `[B, K]`
- Why: keep scores aligned with reordered indices.
- Note: `topk_scores` is not used afterwards in this function.

### Line 125: index expansion for gather
```python
idx_expand = topk_indices.unsqueeze(-1).expand(-1, -1, fused.size(-1))
```
- `topk_indices`: `[B, K]`
- After unsqueeze: `[B, K, 1]`
- After expand: `[B, K, C]`
- Why: `torch.gather` on dim=1 expects index tensor matching output rank.

### Line 126: `event_feats = fused.gather(1, idx_expand)`
- `fused`: `[B, T, C]`
- `idx_expand`: `[B, K, C]`
- Output: `[B, K, C]`
- Math: gather rows (timesteps) selected by top-k.
- Why: convert framewise sequence into sparse event sequence.

### Line 128: `tokens = self.token_proj(event_feats)`
- Input: `[B, K, C]`
- Output: `[B, K, d_model]`
- Why: downstream transformer token embeddings.

### Line 129: `scalars = self.scalar_head(event_feats)`
- Input: `[B, K, C]`
- Output: `[B, K, S]`
- Why: auxiliary event-level numerical attributes.

### Line 130: `type_logits = self.type_head(event_feats)`
- Input: `[B, K, C]`
- Output: `[B, K, M]`
- Why: event class distribution per selected event.

### Line 131: `event_type_id = type_logits.argmax(dim=-1)`
- Input: `[B, K, M]`
- Output: `[B, K]` integer class IDs
- Important gradient note: `argmax` is non-differentiable; gradients do not flow through `event_type_id`.
- Usually training should use `type_logits` + cross-entropy elsewhere.

### Line 132: `token_conf = self.conf_head(event_feats).squeeze(-1)`
- Input: `[B, K, C]`
- Output before squeeze: `[B, K, 1]`
- Output after squeeze: `[B, K]`
- Why: confidence estimate for each event token.

### Line 134: `mask_long = mask.long()`
- Input: `[B, T]` bool/int
- Output: `[B, T]` int64
- Why: gather supports this cleanly; then cast back to bool.

### Line 135: `event_mask = mask_long.gather(1, topk_indices).bool()`
- Inputs: `mask_long [B,T]`, indices `[B,K]`
- Output: `[B, K]` bool
- Why: attention mask aligned to selected events.

### Lines 137-140: build `time_positions`
- If `timestamps` is `None`: use frame indices.
  - `topk_indices.float()` -> `[B, K]`
- Else gather provided timestamps.
  - `timestamps [B,T]` + indices `[B,K]` -> `[B,K]`
- Why: provide temporal position metadata for downstream positional encoding logic.

### Lines 142-152: build `delta_events`

Case A: external `delta_t` provided:
- `delta_events = delta_t.gather(1, topk_indices)`
- Input `delta_t [B,T]` -> output `[B,K]`.

Case B: no `delta_t`, no timestamps:
- `delta_events = zeros_like(time_positions)` -> `[B,K]`.

Case C: no `delta_t`, timestamps available:
- Initialize zeros `[B,K]`.
- For `K>1`, compute consecutive differences:
  - `delta_events[:,1:] = time_positions[:,1:] - time_positions[:,:-1]`
  - clamp min 0.0.
- Why clamp: guards against pathological negative deltas.

### Line 154: `pad_len = self.K_max - K`
- Scalar.
- Why: enforce fixed event length across batches for transformer batching.

### Lines 155-162: right-pad to `[B, K_max, ...]`

- `tokens = F.pad(tokens, (0,0,0,pad_len))`
  - `[B,K,d_model] -> [B,K_max,d_model]`
- `event_mask = F.pad(event_mask, (0,pad_len), value=False)`
  - `[B,K] -> [B,K_max]`
- `time_positions = F.pad(time_positions, (0,pad_len))`
  - `[B,K] -> [B,K_max]`
- `event_type_id = F.pad(event_type_id, (0,pad_len))`
  - `[B,K] -> [B,K_max]`
- `token_conf = F.pad(token_conf, (0,pad_len))`
  - `[B,K] -> [B,K_max]`
- `scalars = F.pad(scalars, (0,0,0,pad_len))`
  - `[B,K,S] -> [B,K_max,S]`
- `delta_events = F.pad(delta_events, (0,pad_len))`
  - `[B,K] -> [B,K_max]`

Why:
- Fixed-size tensors simplify downstream transformer stacks and batching.
- `event_mask` marks valid vs padded events.

If removed:
- Output length varies with `T`, making collation and static shape assumptions harder.

### Lines 164-172: return dictionary
- Packages all event token representations and metadata for downstream modules.

---

## 5. Gradient Flow Analysis

### Fully differentiable paths
- `features -> conv branches -> channel_fuse -> event_feats (selected rows) -> token_proj/scalar_head/type_head/conf_head` is differentiable with respect to selected feature values.

### Discrete bottlenecks
- `topk_indices` are discrete from `topk` and sorting/gather on indices.
- Index selection is non-differentiable with respect to gate logits.
- `argmax` for `event_type_id` is non-differentiable.

### Important practical implication
- In this exact function, `gate_scores` and `topk_scores` are not returned/used in later differentiable outputs.
- Therefore, unless another loss is applied directly to gate outputs elsewhere, `event_gate` may receive little or no learning signal.
- This is a key training risk.

Potential fixes:
- Return `gate_logits`/`gate_scores` and supervise them.
- Use differentiable top-k approximations (Gumbel-topk, NeuralSort, Sinkhorn-based relaxations).
- Use soft selection (attention-weighted pooling) during training, hard top-k at inference.

---

## 6. Computational Complexity

Let `L = len(kernel_sizes)`.

### Conv front-end
- Each branch: `O(B * T * D * C * ks)`.
- Total: `O(B * T * D * C * sum(kernel_sizes))`.
- Usually dominant cost.

### Fusion and gating
- `channel_fuse` linear: `O(B * T * (L*C) * C)`.
- `event_gate` MLP: roughly `O(B * T * C^2/4)`.

### Selection and heads
- `topk`: approx `O(B * T log K)`.
- Gather selected features: `O(B * K * C)`.
- Token projection and heads: `O(B * K * C * d_model + B * K * C * (S + M))`.

### End-to-end effect
- If `K_max << T`, downstream transformer cost is strongly reduced because attention later scales with selected event length rather than full frame length.

---

## 7. How It Fits a Transformer Pipeline

Typical integration:
1. Upstream video backbone emits dense frame tokens/features `[B,T,D]`.
2. `MicroKineticEncoder` compresses to sparse, semantically salient event tokens `[B,K_max,d_model]`.
3. Downstream transformer consumes:
   - `tokens` as embeddings,
   - `attn_mask` for valid event positions,
   - optional `time_positions`/`delta_t` for temporal positional bias,
   - optional metadata heads for multi-task losses.

Design philosophy:
- Early temporal locality (convs) + sparse eventization (top-k) + transformer-compatible tokens.
- This trades dense sequence fidelity for salient-event efficiency.

---

## 8. Strengths and Weaknesses

### Strengths
- Multi-scale temporal convolutions capture local kinematic patterns.
- Explicit event selection can reduce downstream compute.
- Fixed `K_max` output is batching-friendly.
- Multi-head outputs (`type`, `conf`, `scalars`) support multi-task supervision.
- Optional external `conv_features` path supports feature caching.

### Weaknesses / Risks
- Hard `topk` is non-differentiable in index path; gating may not train well without auxiliary loss.
- `topk_scores` computed but unused; may indicate missing supervision hook.
- `device` variable unused.
- Sorting after top-k enforces chronological order but discards rank order signal unless scores are used.
- Potential information loss if salient events are missed by gate.

---

## 9. Block-by-Block Checklist (Requested Format)

## Block A: TemporalConvBlock
- Input shape: `[B,D,T]`
- Output shape: `[B,C,T]`
- Math: conv + group normalization + GELU + dropout
- Why needed: local temporal feature extraction with stable training
- Alternatives: depthwise conv, dilated conv, temporal attention

## Block B: Multi-branch concat and fuse
- Input shape: list of `L` tensors `[B,C,T]`
- Output shape: `[B,T,C]`
- Math: concat channels then linear projection for feature fusion
- Why needed: combine multiple receptive fields into compact latent
- Alternatives: additive fusion, learned branch attention, 1x1 conv

## Block C: Event gate and top-k
- Input shape: `[B,T,C]` (to gate), mask `[B,T]`
- Output shape: indices `[B,K]`, selected feats `[B,K,C]`
- Math: MLP scoring, masked sigmoid, top-k selection, time sorting, gather
- Why needed: sparse event tokenization
- Alternatives: differentiable top-k, soft attention pooling, thresholding

## Block D: Token/head projections
- Input shape: `[B,K,C]`
- Output shape: tokens `[B,K,d_model]`, scalars `[B,K,S]`, type logits `[B,K,M]`, conf `[B,K]`
- Math: linear projections with optional normalization/activations
- Why needed: produce transformer inputs and auxiliary predictions
- Alternatives: shared trunk with separate low-rank heads, mixture-of-experts heads

## Block E: Time and delta metadata
- Input shape: indices `[B,K]`, optional timestamps/delta `[B,T]`
- Output shape: `time_positions [B,K]`, `delta_t [B,K]`
- Math: gather or finite differences + clamp
- Why needed: temporal context beyond token content
- Alternatives: relative position encoding learned from indices directly

## Block F: Padding to fixed K_max
- Input shape: variable `K`
- Output shape: fixed `K_max`
- Math: right-padding with zeros/False
- Why needed: static batch shape for downstream transformer
- Alternatives: ragged batching, packed sequence APIs

---

## 10. Architectural Summary (Simple Language)

The module scans a full video feature sequence, finds the most important moments, and turns those moments into a fixed number of compact tokens. It also predicts event type, confidence, scalar descriptors, and timing metadata for each selected moment.

## 11. Architectural Summary (Beginner Version)

Think of this as a highlight picker for video features.
- First, it looks at motion patterns at different time scales.
- Then it scores each frame for importance.
- It keeps only the top moments.
- It converts those moments into embeddings a transformer can read.
- It adds helper info: when the moment happened, what type it may be, and confidence.

## 12. Architectural Summary (Research Reviewer Style)

`MicroKineticEncoder` is a hybrid temporal front-end combining multi-scale 1D convolutions with hard salience-based event sparsification. It maps dense framewise embeddings to fixed-cardinality event tokens suitable for transformer back-ends while emitting auxiliary supervision channels (`type`, `confidence`, `scalars`, timing). The principal concern is optimization of the hard selection mechanism: without explicit gate supervision or differentiable relaxation, event-gating may be weakly trained due to non-differentiable index paths. Nonetheless, the design is computationally attractive when `K_max << T`, enabling substantial sequence-length reduction before global attention.

---

## 13. Suggested Improvements

1. Return `gate_logits`/`gate_scores` and add explicit gate loss.
2. Consider soft-to-hard selection schedule (train soft, infer hard).
3. Remove or use `device` variable.
4. Optionally return `type_logits` for direct CE training in this module boundary.
5. Consider adding score-aware ordering (time + salience) if ranking is important.
