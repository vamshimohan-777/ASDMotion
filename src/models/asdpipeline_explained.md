# ASDPipeline Deep Technical Explanation

Source file: `src/models/pipeline_model.py`

## Scope

This document gives a line-by-line, shape-tracked explanation of `ASDPipeline` and the exact tensor flow through its `forward` method.

Because `ASDPipeline` composes two major submodules, this explanation also references:
- `src/models/video/motion/event_encoder.py` (`ResNetMicroKineticEventEncoder`)
- `src/models/video/motion/behavior_transformer.py` (`BehavioralTransformer`)

---

## Notation

- `B`: batch size (videos per batch)
- `S`: number of sampled windows per video
- `W`: frames per window
- `J`: number of joints (expected `135`)
- `F`: per-joint feature dim (expected `9`)
- `D`: latent embedding dimension (`d_model`, default `256`)
- `K`: selected event tokens per window (`k_max`, default from config)
- `T_tok`: flattened token length to transformer = `S * K`

Input contract to `ASDPipeline.forward`:
- `motion_windows`: `[B, S, W, J, 9]`
- `joint_mask` (optional): `[B, S, W, J]`
- `window_timestamps` (optional): `[B, S, W]`

Primary outputs:
- `logit_final`: `[B]`
- `prob_final`: `[B]`
- event evidence tensors for reporting/explainability.

---

## 1. Class Overview: `ASDPipeline`

`ASDPipeline` is the top-level model that converts multi-window landmark-motion tensors into one ASD logit per video.

High-level flow:
1. Validate shapes and schema.
2. Flatten windows across batch for efficient shared encoding.
3. Encode each window into sparse event tokens (`K` events per window).
4. Flatten all events across all windows into one token sequence per video.
5. Run transformer reasoning over this event sequence.
6. Convert logits to probabilities/confidence/tri-state decision and return evidence tensors.

Architectural philosophy:
- **Window-level local encoding** + **global event-level transformer**.
- Keep token budget bounded (`S*K`) instead of attending over all raw frames.
- Preserve explainability hooks (`event_time_series`, `frame_event_scores`, per-window scores).

---

## 2. Constructor (`__init__`) Line-by-Line

Reference: `pipeline_model.py:18-60`

### Signature and compatibility args
- Accepts legacy arguments (`alpha`, `cnn_backbone`, `nas_search_space`, `num_event_types`) but explicitly discards them.
- Why: backward compatibility with older call sites/checkpoints.

If removed:
- Older training/inference scripts may break when passing outdated kwargs.

### `self.schema = DEFAULT_SCHEMA`
- Stores deterministic joint layout (pose/hands/face slices, total joints).
- Why: enforce strict structural contract for motion tensors.

### `self.theta_high`, `self.theta_low`
- Decision thresholds for output tri-state mapping in forward.

### `self.aggregation_method = "attention"`
- Default token pooling mode in transformer path.

### `self._encoder_frozen = False`
- Tracks freeze state to keep motion encoder in eval mode when desired.

### `self.k_max = int(max(1, K_max))`
- Ensures at least one event token per window.

### `self.architecture = {...}`
- Stores mutable architecture dictionary (encoder + transformer + window aggregation).
- Used by NAS and runtime architecture replacement.

### `self._build_from_architecture(self.architecture)`
- Instantiates submodules based on architecture dictionary.

---

## 3. `_build_from_architecture` Line-by-Line

Reference: `pipeline_model.py:61-90`

### Parse architecture sections
- `encoder_cfg`, `transformer_cfg`, optional `window.aggregation`.

### Head-divisibility guard
- `d_model % n_heads` must be 0 for multi-head attention.
- If not divisible, fallback to first valid in `(2,4,8)` or `1`.

Why:
- Avoid runtime exception inside `TransformerEncoderLayer`.

Alternative:
- Auto-adjust `d_model` instead of heads.

### Build `self.motion_encoder`
- Class: `ResNetMicroKineticEventEncoder`.
- Input shape to that encoder: `[BS, W, J, 9]`.
- Output includes sparse event tokens `[BS, K, D]` and window embeddings `[BS, D]`.

### Build `self.behavior`
- Class: `BehavioralTransformer`.
- Consumes flattened event tokens `[B, S*K, D]` with mask/time metadata.

---

## 4. Architecture Mutation and Freeze Utilities

### `_rebuild_with_arch` (`91-99`)
- Rebuilds model submodules on current device after architecture mutation.
- Required for NAS application.

### `freeze_motion_encoder` (`100-105`)
- Sets `requires_grad=False` for motion encoder parameters.
- Calls `eval()` on encoder.

Gradient implication:
- No encoder updates; gradients stop at frozen params.

### `unfreeze_upper_motion_layers` (`106-140`)
- Partial unfreezing strategy:
  - Freeze all first.
  - Unfreeze last `num_blocks` temporal blocks.
  - Also unfreeze output/projection heads (`temporal_out`, `window_pool`, `event_score_head`, `frame_proj`).
- Fallback branch supports older encoder variant.

Why:
- Stable training first, then gentle domain adaptation.

### `train(self, mode=True)` override (`146-150`)
- Keeps motion encoder in eval mode if `_encoder_frozen` is true.
- Prevents BN/dropout behavior drift in frozen encoder.

---

## 5. `forward` Line-by-Line (Exhaustive)

Reference: `pipeline_model.py:212-323`

## Input validation

### Line 214: `motion = inputs["motion_windows"]`
- Expected shape: `[B, S, W, J, 9]`.

### Lines 215-223: shape checks
- `motion.dim() == 5`
- `J == schema.total_joints` (`135`)
- `F == 9`

Why:
- Hard guarantees for downstream reshape/slicing logic.

If removed:
- Silent shape mismatch can corrupt semantics and gradients.

## Window validity mask

### Line 225: `joint_mask = inputs.get("joint_mask")`

### Lines 226-233
- If present, require `joint_mask.dim()==4` and compute:
  - `window_valid = (joint_mask.float().sum(dim=(2,3)) > 0)`
  - Shape: `[B, S]`
- Else set all windows valid.

Math:
- A window is valid if any joint at any frame is present.

Why:
- Avoid attending to fully empty windows.

Alternative:
- Minimum visible-ratio threshold rather than >0.

## Timestamp validation

### Lines 235-239
- If provided, `window_timestamps` must be rank-3 `[B,S,W]`.

## Flatten windows for shared encoding

### Line 241: `flat_motion = motion.reshape(B*S, W, J, F)`
- Input: `[B,S,W,J,F]`
- Output: `[BS, W, J, F]`

Why:
- Encode each window independently with one batched call.

### Line 242: `flat_mask`
- `[B,S,W,J] -> [BS,W,J]` if mask exists.

### Line 243: `flat_ts`
- `[B,S,W] -> [BS,W]` if timestamps exist.

## Motion encoder call

### Lines 245-250
```python
enc_out = self.motion_encoder(
    flat_motion,
    joint_mask=flat_mask,
    timestamps=flat_ts,
    return_events=True,
)
```

Expected outputs from event encoder:
- `window_embedding`: `[BS,D]`
- `event_vectors`: `[BS,K,D]`
- `event_mask`: `[BS,K]`
- `event_times`: `[BS,K]`
- `event_frame_index`: `[BS,K]`
- `frame_event_scores`: `[BS,W]`

Why `return_events=True`:
- Pipeline needs sparse event tokens for transformer, not only pooled window vectors.

## Unpack encoder outputs

### Lines 251-256
- Extract each tensor by key.
- Pure bookkeeping; no shape change.

## Reshape back to batch/window structure

### Lines 258-265
- `D = flat_window_emb.shape[-1]`
- `K = flat_event_vec.shape[1]`
- `window_embeddings = [BS,D] -> [B,S,D]`
- `event_vectors = [BS,K,D] -> [B,S,K,D]`
- `event_mask = [BS,K] -> [B,S,K]`
- `event_times = [BS,K] -> [B,S,K]`
- `event_frame_idx = [BS,K] -> [B,S,K]`
- `frame_event_scores = [BS,W] -> [B,S,W]`

Why:
- Keep window-local evidence for reporting while still preparing global token sequence.

## Flatten events over windows for global reasoning

### Line 268: `token_embeddings = event_vectors.reshape(B, S*K, D)`
- `[B,S,K,D] -> [B,T_tok,D]`

### Line 269: `token_mask = ...`
- Start: `event_mask.reshape(B,S*K)`
- AND with expanded `window_valid` to suppress events from invalid windows.
- Final shape: `[B, S*K]` bool.

### Line 270: `token_times = event_times.reshape(B, S*K)`
- `[B,S,K] -> [B,T_tok]`

Why:
- Transformer consumes one sequence per sample; all event tokens unified.

## Transformer reasoning

### Lines 272-277
```python
out = self.behavior(
    token_embeddings,
    window_mask=token_mask,
    event_times=token_times,
    aggregation=self.aggregation_method,
)
```

BehavioralTransformer returns at least:
- `logit`: `[B]`
- Optional per-token `window_scores`: `[B,T_tok]`
- Optional attention weights: `[B,T_tok]`

## Final probability/confidence/decision

### Line 278: `logit = out["logit"]` -> `[B]`

### Line 279: `prob = sigmoid(logit)` -> `[B]`

### Line 280: `confidence = max(prob, 1-prob)` -> `[B]`
- Confidence high near 0 or 1, low near 0.5.

### Lines 281-285: tri-state tensor decision
- `1` if `prob >= theta_high`
- `0` if `prob <= theta_low`
- `-1` otherwise

Why:
- Native uncertain zone encoding.
- Later inference maps these semantics to human-readable labels.

If changed to fixed 0.5 threshold:
- No abstain/recheck regime.

## Aggregate token scores to per-window scores

### Line 288: `token_scores = out.get("window_scores")`
- Usually token-level probabilities/saliency from behavior module.

### Lines 289-295
- If present:
  - reshape `[B,S*K] -> [B,S,K]`
  - invalidate non-valid tokens with `-1`
  - per-window score = max over K events -> `[B,S]`

Why max:
- A window is salient if any event token is salient.

Alternative:
- mean/top-m average/attention-weighted reduce.

## Aggregate attention per window

### Lines 296-302
- If attention exists:
  - reshape `[B,S*K] -> [B,S,K]`
  - zero invalid tokens
  - sum over events per window -> `[B,S]`

Interpretation:
- Window-level attention mass.

## Return dictionary

### Lines 304-323
Returns prediction and explainability bundle:
- Main outputs: `logit_final`, `prob_final`, `confidence`, `decision`
- Compatibility aliases: `p_final`, `p_video`, `p_image`, `alpha`
- Evidence: `window_scores`, `attention_weights`, token-level variants
- Rich event series: vectors/times/masks/frame indices/frame scores.

---

## 6. Mathematical Interpretation

Given video `i`, with event tokens `E_i in R^{(S*K) x D}`:

1. Transformer encoding:
- `H_i = Transformer(E_i + PE + TimeMLP(log(1+t)))`

2. Pooling (attention mode in `BehavioralTransformer`):
- `a = softmax(g(H_i))`
- `z = sum_t a_t H_{i,t}`

3. Classification:
- `logit_i = h(z)`
- `p_i = sigma(logit_i)`

4. Decision:
- `1` if `p_i >= theta_high`
- `0` if `p_i <= theta_low`
- `-1` otherwise

---

## 7. Gradient Flow Through ASDPipeline

Differentiable paths:
- `logit_final` gradients flow through:
  - Transformer (`behavior`)
  - Event token embeddings
  - Motion encoder selected event features

Non-smooth/discrete parts:
- Event selection in motion encoder uses `topk` indices (discrete).
- Gradients do not flow through index choices, only through selected values.

Freeze effects:
- When encoder frozen (`requires_grad=False`), only transformer/head paths update.

---

## 8. Computational Complexity

Let `T_tok = S*K`.

1. Motion encoder (dominant early cost):
- ResNet over `B*S*W` pseudo-images + temporal conv stack.

2. Transformer stage:
- Self-attention complexity `O(B * T_tok^2 * D)`.

3. Memory:
- Attention maps scale as `O(B * n_heads * T_tok^2)`.

Design payoff:
- Uses event sparsification (`K`) so transformer cost depends on `S*K`, not raw frame count `S*W`.

---

## 9. Strengths and Weaknesses

### Strengths
- Explicit hierarchy: frame -> window events -> global reasoning.
- Compute-efficient vs dense frame transformers.
- Strong explainability outputs at event/window levels.
- NAS-ready mutable architecture dictionary.
- Training-friendly freeze/unfreeze strategy.

### Weaknesses
- Hard top-k selection is non-differentiable wrt indices.
- Pseudo-image conversion may discard natural skeletal graph structure.
- Multi-stage training/NAS adds operational complexity.
- Decision thresholds are heuristic and require calibration discipline.

---

## 10. Alternative Design Choices

1. Replace hard top-k with soft differentiable tokenization.
2. Replace ResNet pseudo-image path with GCN/Graph Transformer on joints.
3. Use relative time bias in attention instead of additive time MLP only.
4. Learn uncertainty head jointly and calibrate with focal + ECE-aware objectives.
5. Replace max window aggregation with learned window-level attention pooling.

---

## 11. Input/Output Shape Cheat Sheet

- Input `motion_windows`: `[B,S,W,135,9]`
- Flatten windows: `[BS,W,135,9]`
- Event vectors per window: `[BS,K,D]`
- Reshaped event series: `[B,S,K,D]`
- Flattened transformer tokens: `[B,S*K,D]`
- Final logit/prob: `[B]`
- Window evidence scores: `[B,S]`

---

## 12. Quick Plain-Language Summary

`ASDPipeline` turns each video into multiple motion windows, extracts key motion events from each window, merges all events into one sequence, and lets a transformer reason across them to produce a calibrated ASD risk score plus interpretable event evidence.
