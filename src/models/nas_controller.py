
"""
Micro-NAS Controller with defined search space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.video.microkinetic_encoders.microkinetics import (
    MicroKineticEncoder,
    TemporalConvBlock,
)
from src.models.video.transformer_reasoning.event_transformer import TemporalTransformer


DEFAULT_ENCODER_KERNEL_CANDIDATES = [3, 5, 7, 11]
DEFAULT_TRANSFORMER_HEADS_CANDIDATES = [2, 4, 8]
DEFAULT_TRANSFORMER_LAYERS_CANDIDATES = [2, 3, 4]
DEFAULT_TRANSFORMER_FF_CANDIDATES = [512, 1024, 2048]


class NASEncoderCell(nn.Module):
    def __init__(self, d_in: int, dropout: float = 0.2, kernel_candidates: list[int] = None):
        super().__init__()
        # Candidate temporal kernel sizes searched by NAS for event-scale sensitivity.
        self.kernel_candidates = [int(v) for v in (kernel_candidates or DEFAULT_ENCODER_KERNEL_CANDIDATES)]
        if not self.kernel_candidates:
            raise ValueError("kernel_candidates must contain at least one kernel size.")

        # One temporal-conv branch per kernel size.
        self.kernel_branches = nn.ModuleList()
        for ks in self.kernel_candidates:
            # Each branch maps raw token channels to shared conv embedding space.
            branch = TemporalConvBlock(d_in, 256, ks, dropout=dropout)
            self.kernel_branches.append(branch)

        # Learnable architecture logits over kernel branches.
        self.alpha_kernel = nn.Parameter(torch.zeros(len(self.kernel_candidates)))

    def forward(self, x, tau=1.0, use_gumbel=True):
        # During search, Gumbel-Softmax approximates discrete architecture sampling.
        if use_gumbel and self.training:
            weights_k = F.gumbel_softmax(self.alpha_kernel, tau=tau, hard=False)
        else:
            # During eval/discretized use plain softmax probabilities.
            weights_k = F.softmax(self.alpha_kernel, dim=0)
        # Weighted branch mixture lets gradients optimize architecture and weights jointly.
        out = sum(w * branch(x) for w, branch in zip(weights_k, self.kernel_branches))
        return out


class MicroNASController(nn.Module):
    def __init__(
        self,
        d_in: int = 768,
        d_model: int = 256,
        K_max: int = 32,
        num_event_types: int = 12,
        num_scalars: int = 8,
        dropout: float = 0.3,
        encoder_kernel_candidates: list[int] = None,
        transformer_heads_candidates: list[int] = None,
        transformer_layers_candidates: list[int] = None,
        transformer_ff_candidates: list[int] = None,
    ):
        super().__init__()
        # Cache key dimensions because downstream encoder/transformer rely on consistency.
        self.d_in = d_in
        self.d_model = d_model
        self.K_max = K_max
        # When True, architecture is frozen to a single best configuration.
        self.is_discretized = False

        def _normalize_candidates(values, defaults, field_name):
            # Accept scalar or list input while guaranteeing non-empty int candidates.
            if values is None:
                out = list(defaults)
            elif isinstance(values, (list, tuple)):
                out = [int(v) for v in values]
            else:
                out = [int(values)]
            if not out:
                raise ValueError(f"{field_name} must contain at least one value.")
            return out

        self.encoder_kernel_candidates = _normalize_candidates(
            encoder_kernel_candidates, DEFAULT_ENCODER_KERNEL_CANDIDATES, "encoder_kernel_candidates"
        )
        self.transformer_heads_candidates = _normalize_candidates(
            transformer_heads_candidates, DEFAULT_TRANSFORMER_HEADS_CANDIDATES, "transformer_heads_candidates"
        )
        self.transformer_layers_candidates = _normalize_candidates(
            transformer_layers_candidates, DEFAULT_TRANSFORMER_LAYERS_CANDIDATES, "transformer_layers_candidates"
        )
        self.transformer_ff_candidates = _normalize_candidates(
            transformer_ff_candidates, DEFAULT_TRANSFORMER_FF_CANDIDATES, "transformer_ff_candidates"
        )

        # Temperature schedule controls exploration (high tau) to exploitation (low tau).
        self.tau = 1.0
        self._tau_start = 1.0
        self._tau_end = 0.1
        self._total_steps = 1
        self._current_step = 0

        # EMA of architecture weights tracks search stability over time.
        self._ema_alpha = None

        # NAS-searched temporal-conv pre-encoder.
        self.nas_cell = NASEncoderCell(
            d_in,
            dropout=dropout,
            kernel_candidates=self.encoder_kernel_candidates,
        )

        # Micro-kinetic event encoder converts frame-level stream into event tokens.
        self.encoder = MicroKineticEncoder(
            d_in=d_in,
            d_model=d_model,
            K_max=K_max,
            num_event_types=num_event_types,
            num_scalars=num_scalars,
            conv_channels=256,
            kernel_sizes=[3, 5, 7],
            dropout=dropout,
        )

        # Full Cartesian product of transformer architecture candidates.
        self.transformer_candidates = nn.ModuleList()
        self._transformer_configs = []
        for n_heads in self.transformer_heads_candidates:
            for n_layers in self.transformer_layers_candidates:
                for ff_dim in self.transformer_ff_candidates:
                    # Persist config for reporting/discretization later.
                    cfg = {
                        "n_heads": n_heads,
                        "num_encoder_layers": n_layers,
                        "dim_ff": ff_dim,
                    }
                    self._transformer_configs.append(cfg)
                    self.transformer_candidates.append(
                        TemporalTransformer(
                            d_model=d_model,
                            n_heads=n_heads,
                            scalars_dim=num_scalars,
                            num_encoder_layers=n_layers,
                            dim_ff=ff_dim,
                            dropout=dropout,
                            num_event_types=num_event_types,
                        )
                    )

        # Learnable architecture logits over transformer variants.
        self.alpha_transformer = nn.Parameter(torch.zeros(len(self.transformer_candidates)))
        # Populated once architecture is fixed.
        self._best_transformer_idx = None

    def configure_temperature(self, tau_start=1.0, tau_end=0.1, total_steps=1000):
        # Linear annealing schedule for smoother NAS convergence.
        self._tau_start = tau_start
        self._tau_end = tau_end
        self._total_steps = max(total_steps, 1)
        self._current_step = 0
        self.tau = tau_start

    def step_temperature(self):
        # Move one step in the annealing schedule.
        self._current_step += 1
        progress = min(self._current_step / self._total_steps, 1.0)
        self.tau = self._tau_start + (self._tau_end - self._tau_start) * progress

    def clip_arch_grads(self, max_norm=1.0):
        # Architecture gradients can spike; clipping prevents unstable jumps.
        arch_params = [p for p in self.arch_parameters() if p.grad is not None]
        if arch_params:
            torch.nn.utils.clip_grad_norm_(arch_params, max_norm=max_norm)

    def _update_ema(self):
        with torch.no_grad():
            # Compare current architecture preference to smoothed historical trend.
            current = F.softmax(self.alpha_transformer, dim=0).detach()
            if self._ema_alpha is None:
                self._ema_alpha = current.clone()
            else:
                self._ema_alpha = 0.99 * self._ema_alpha + 0.01 * current

    @property
    def arch_weight_variance(self):
        # Low variance implies architecture selection has stabilized.
        if self._ema_alpha is None:
            return 0.0
        current = F.softmax(self.alpha_transformer, dim=0).detach()
        return ((current - self._ema_alpha) ** 2).mean().item()

    def arch_parameters(self):
        # After discretization there is no architecture optimization target left.
        if self.is_discretized:
            return []
        return [self.alpha_transformer, self.nas_cell.alpha_kernel]

    @staticmethod
    def get_random_config():
        import random
        # Utility for random architecture baselines/ablation studies.
        return {
            "transformer": {
                "n_heads": random.choice(DEFAULT_TRANSFORMER_HEADS_CANDIDATES),
                "num_encoder_layers": random.choice(DEFAULT_TRANSFORMER_LAYERS_CANDIDATES),
                "dim_ff": random.choice(DEFAULT_TRANSFORMER_FF_CANDIDATES),
            },
            "encoder_kernel": random.choice(DEFAULT_ENCODER_KERNEL_CANDIDATES),
        }

    def model_parameters(self):
        # Split standard weights from architecture logits for separate optimizers.
        arch_set = set(id(p) for p in self.arch_parameters())
        return [p for p in self.parameters() if id(p) not in arch_set]

    def forward(self, features: torch.Tensor, mask: torch.Tensor,
                timestamps: torch.Tensor = None, delta_t: torch.Tensor = None) -> dict:
        # Conv branch expects [B, C, T], while main tokens are [B, T, C].
        conv_in = features.permute(0, 2, 1)
        # Search/fix the best temporal receptive field for event extraction.
        nas_features = self.nas_cell(conv_in, tau=self.tau, use_gumbel=(not self.is_discretized))
        nas_features = nas_features.permute(0, 2, 1)

        # Encode sequence into sparse informative event tokens for transformer reasoning.
        enc_out = self.encoder(features, mask, conv_features=nas_features,
                               timestamps=timestamps, delta_t=delta_t)
        # Forward event diagnostics to final output for interpretability.
        event_payload = {
            "event_type_id": enc_out.get("event_type_id"),
            "event_mask": enc_out.get("attn_mask"),
            "event_confidence": enc_out.get("token_conf"),
        }

        if self.is_discretized:
            # Fast inference path: run only selected transformer architecture.
            out = self.transformer_candidates[self._best_transformer_idx](enc_out)
            out.update(event_payload)
            return out
        else:
            if self.training:
                # NAS search path: soft sample transformer architecture weights.
                weights = F.gumbel_softmax(self.alpha_transformer, tau=self.tau, hard=False)
                self._update_ema()
            else:
                # Deterministic weighted ensemble when evaluating unfrozen search model.
                weights = F.softmax(self.alpha_transformer, dim=0)

            combined = None
            for w, transformer in zip(weights, self.transformer_candidates):
                # Aggregate outputs as weighted architecture mixture.
                out = transformer(enc_out)
                if combined is None:
                    combined = {k: w * v for k, v in out.items()}
                else:
                    for k in combined:
                        combined[k] = combined[k] + w * out[k]
            # Ensure probability is consistent with mixed logit.
            combined["prob"] = torch.sigmoid(combined["logit"])
            combined.update(event_payload)
            return combined

    def get_current_config(self):
        # Prefer explicitly selected transformer if already discretized/applied.
        if hasattr(self, "_best_transformer_idx") and self._best_transformer_idx is not None:
            best_cfg = self._transformer_configs[self._best_transformer_idx]
        else:
            # Otherwise report argmax over architecture logits.
            best_idx = self.alpha_transformer.argmax().item()
            best_cfg = self._transformer_configs[best_idx]

        best_kernel_idx = self.nas_cell.alpha_kernel.argmax().item()
        best_kernel = self.encoder_kernel_candidates[best_kernel_idx]

        return {
            "transformer": best_cfg,
            "encoder_kernel": best_kernel,
        }

    def discretize(self):
        # Freeze search into one architecture for stable and cheaper inference.
        config = self.get_current_config()
        self.is_discretized = True

        if not hasattr(self, "_best_transformer_idx") or self._best_transformer_idx is None:
            self._best_transformer_idx = self.alpha_transformer.argmax().item()

        # Stop gradients on architecture logits after selection.
        for p in [self.alpha_transformer, self.nas_cell.alpha_kernel]:
            p.requires_grad = False

        print("[NAS] Architecture:")
        print(f"  Transformer: heads={config['transformer']['n_heads']}, "
              f"layers={config['transformer']['num_encoder_layers']}, "
              f"ff_dim={config['transformer']['dim_ff']}")
        print(f"  Encoder kernel: {config['encoder_kernel']}")
        return config

    def apply_config(self, config):
        # Load externally supplied architecture (e.g., from prior NAS run).
        self.is_discretized = True
        target_cfg = config["transformer"]
        for idx, cfg in enumerate(self._transformer_configs):
            if (cfg["n_heads"] == target_cfg["n_heads"] and
                cfg["num_encoder_layers"] == target_cfg["num_encoder_layers"] and
                cfg["dim_ff"] == target_cfg["dim_ff"]):
                self._best_transformer_idx = idx
                break

        target_kernel = config["encoder_kernel"]
        # Force kernel alpha to one-hot-like state for deterministic behavior.
        best_kernel_idx = self.encoder_kernel_candidates.index(target_kernel)
        self.nas_cell.alpha_kernel.data.fill_(-float("inf"))
        self.nas_cell.alpha_kernel.data[best_kernel_idx] = 10.0

        for p in [self.alpha_transformer, self.nas_cell.alpha_kernel]:
            p.requires_grad = False

    def arch_entropy_loss(self):
        # Entropy regularizer: tune sign in training objective depending on
        # whether you want exploration (maximize) or confidence (minimize).
        entropy = 0.0
        for alpha in self.arch_parameters():
            p = F.softmax(alpha, dim=0)
            entropy += -(p * (p + 1e-8).log()).sum()
        return entropy

