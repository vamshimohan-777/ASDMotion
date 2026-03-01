"""Model module `src/models/nas_controller.py` that transforms inputs into features used for prediction."""

# Import `copy` to support computations in this stage of output generation.
import copy
# Import `random` to support computations in this stage of output generation.
import random
# Import symbols from `dataclasses` used in this stage's output computation path.
from dataclasses import dataclass

# Import `numpy as np` to support computations in this stage of output generation.
import numpy as np
# Import `torch` to support computations in this stage of output generation.
import torch


# Define a reusable pipeline function whose outputs feed later steps.
def default_search_space():
    """Executes this routine and returns values used by later pipeline output steps."""
    # Return `{` as this function's contribution to downstream output flow.
    return {
        "encoder": {
            "branch_blocks": [2, 3, 4],
            "branch_channels": [64, 128, 256],
            "kernel_size": [5, 7, 9, 11, 13],
            "use_dilation": [False, True],
            "residual": [False, True],
            "embedding_dim": [128, 256, 512],
            "fusion_dim": [128, 256, 512],
            "modality_dropout": [0.0, 0.1, 0.2, 0.3],
        },
        "transformer": {
            "layers": [2, 3, 4],
            "heads": [2, 4],
            "ff_dim": [256, 384, 512],
            "dropout": [0.1, 0.2, 0.3],
        },
        "window": {
            "size": [32, 48, 64],
            "stride": [8, 16, 24, 32],
            "aggregation": ["attention", "topk"],
        },
    }


# Define a reusable pipeline function whose outputs feed later steps.
def _choice(values, rng):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Set `values` for subsequent steps so downstream prediction heads receive the right feature signal.
    values = list(values)
    # Branch on `not values` to choose the correct output computation path.
    if not values:
        # Raise explicit error to stop invalid state from producing misleading outputs.
        raise ValueError("Search-space candidate list is empty.")
    # Return `copy.deepcopy(rng.choice(values))` as this function's contribution to downstream output flow.
    return copy.deepcopy(rng.choice(values))


# Define a reusable pipeline function whose outputs feed later steps.
def sample_architecture(search_space, rng):
    """Executes this routine and returns values used by later pipeline output steps."""
    # Return `{` as this function's contribution to downstream output flow.
    return {
        "encoder": {k: _choice(v, rng) for k, v in search_space["encoder"].items()},
        "transformer": {k: _choice(v, rng) for k, v in search_space["transformer"].items()},
        "window": {k: _choice(v, rng) for k, v in search_space["window"].items()},
    }


# Define a reusable pipeline function whose outputs feed later steps.
def compute_fitness(metrics):
    """
    Multi-objective fitness:
      0.40 * AUC
    + 0.25 * Sensitivity@90Specificity
    + 0.15 * Calibration quality
    + 0.10 * Cross-validation stability
    + 0.10 * Efficiency score
    """
    # Record `auc` as a metric describing current output quality.
    auc = float(metrics.get("auc", 0.0))
    # Record `sens90` as a metric describing current output quality.
    sens90 = float(metrics.get("sens_at_90_spec", 0.0))
    # Set `calibration` for subsequent steps so downstream prediction heads receive the right feature signal.
    calibration = float(metrics.get("calibration_quality", 0.0))
    # Set `stability` for subsequent steps so downstream prediction heads receive the right feature signal.
    stability = float(metrics.get("cv_stability", 0.0))
    # efficiency_penalty in [0,1], lower is better.
    # Set `eff_pen` for subsequent steps so downstream prediction heads receive the right feature signal.
    eff_pen = float(metrics.get("efficiency_penalty", 1.0))
    # Set `eff_score` for subsequent steps so downstream prediction heads receive the right feature signal.
    eff_score = float(np.clip(1.0 - eff_pen, 0.0, 1.0))

    # Set `score` for subsequent steps so downstream prediction heads receive the right feature signal.
    score = (
        0.40 * auc
        + 0.25 * sens90
        + 0.15 * calibration
        + 0.10 * stability
        + 0.10 * eff_score
    )
    # Return `float(score)` as this function's contribution to downstream output flow.
    return float(score)


# Execute this statement so downstream prediction heads receive the right feature signal.
@dataclass
class NASCandidate:
    """`NASCandidate` groups related operations that shape intermediate and final outputs."""
    # Execute this statement so downstream prediction heads receive the right feature signal.
    architecture: dict
    # Execute this statement so downstream prediction heads receive the right feature signal.
    metrics: dict | None = None
    # Execute this statement so downstream prediction heads receive the right feature signal.
    fitness: float = -1e9


# Define class `MicroGeneticNAS` to package related logic in the prediction pipeline.
class MicroGeneticNAS:
    """`MicroGeneticNAS` groups related operations that shape intermediate and final outputs."""
    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(
        self,
        population_size=20,
        generations=20,
        tournament_size=3,
        mutation_rate=0.15,
        crossover=True,
        elite_count=2,
        seed=42,
    ):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Compute `self.population_size` as an intermediate representation used by later output layers.
        self.population_size = int(max(4, population_size))
        # Set `self.generations` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.generations = int(max(1, generations))
        # Compute `self.tournament_size` as an intermediate representation used by later output layers.
        self.tournament_size = int(max(2, tournament_size))
        # Set `self.mutation_rate` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.mutation_rate = float(np.clip(mutation_rate, 0.0, 1.0))
        # Set `self.crossover` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.crossover = bool(crossover)
        # Set `self.elite_count` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.elite_count = int(max(1, elite_count))
        # Set `self.rng` for subsequent steps so downstream prediction heads receive the right feature signal.
        self.rng = random.Random(int(seed))

    # Define a reusable pipeline function whose outputs feed later steps.
    def _random_population(self, search_space):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Return `[` as this function's contribution to downstream output flow.
        return [
            NASCandidate(architecture=sample_architecture(search_space, self.rng))
            # Iterate through items to accumulate output-relevant computations.
            for _ in range(self.population_size)
        ]

    # Define evaluation logic used to measure prediction quality.
    def _evaluate(self, population, evaluate_fn):
        """Computes validation metrics used to judge model quality and influence training decisions."""
        # Iterate over `population` so each item contributes to final outputs/metrics.
        for cand in population:
            # Branch on `cand.metrics is None` to choose the correct output computation path.
            if cand.metrics is None:
                # Record `metrics` as a metric describing current output quality.
                metrics = evaluate_fn(copy.deepcopy(cand.architecture))
                # Record `cand.metrics` as a metric describing current output quality.
                cand.metrics = metrics
                # Set `cand.fitness` for subsequent steps so downstream prediction heads receive the right feature signal.
                cand.fitness = compute_fitness(metrics)
        # Call `population.sort` and use its result in later steps so downstream prediction heads receive the right feature signal.
        population.sort(key=lambda c: c.fitness, reverse=True)
        # Return `population` as this function's contribution to downstream output flow.
        return population

    # Define a reusable pipeline function whose outputs feed later steps.
    def _tournament_pick(self, population):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `k` for subsequent steps so downstream prediction heads receive the right feature signal.
        k = min(self.tournament_size, len(population))
        # Set `contenders` for subsequent steps so downstream prediction heads receive the right feature signal.
        contenders = self.rng.sample(population, k=k)
        # Call `contenders.sort` and use its result in later steps so downstream prediction heads receive the right feature signal.
        contenders.sort(key=lambda c: c.fitness, reverse=True)
        # Return `contenders[0]` as this function's contribution to downstream output flow.
        return contenders[0]

    # Define a reusable pipeline function whose outputs feed later steps.
    def _crossover(self, arch_a, arch_b):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Branch on `not self.crossover` to choose the correct output computation path.
        if not self.crossover:
            # Return `copy.deepcopy(arch_a)` as this function's contribution to downstream output flow.
            return copy.deepcopy(arch_a)
        # Compute `child` as an intermediate representation used by later output layers.
        child = {"encoder": {}, "transformer": {}, "window": {}}
        # Iterate over `child.keys()` so each item contributes to final outputs/metrics.
        for section in child.keys():
            # Set `keys` for subsequent steps so downstream prediction heads receive the right feature signal.
            keys = set(arch_a.get(section, {}).keys()) | set(arch_b.get(section, {}).keys())
            # Iterate over `keys` so each item contributes to final outputs/metrics.
            for k in keys:
                # Branch on `self.rng.random() < 0.5` to choose the correct output computation path.
                if self.rng.random() < 0.5:
                    # Compute `child[section][k]` as an intermediate representation used by later output layers.
                    child[section][k] = copy.deepcopy(arch_a.get(section, {}).get(k))
                else:
                    # Compute `child[section][k]` as an intermediate representation used by later output layers.
                    child[section][k] = copy.deepcopy(arch_b.get(section, {}).get(k))
        # Return `child` as this function's contribution to downstream output flow.
        return child

    # Define a reusable pipeline function whose outputs feed later steps.
    def _mutate(self, arch, search_space):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Set `out` for subsequent steps so downstream prediction heads receive the right feature signal.
        out = copy.deepcopy(arch)
        # Iterate over `out.items()` so each item contributes to final outputs/metrics.
        for section, params in out.items():
            # Iterate over `list(params.keys())` so each item contributes to final outputs/metrics.
            for key in list(params.keys()):
                # Branch on `self.rng.random() < self.mutation_rate` to choose the correct output computation path.
                if self.rng.random() < self.mutation_rate:
                    # Set `out[section][key]` for subsequent steps so downstream prediction heads receive the right feature signal.
                    out[section][key] = _choice(search_space[section][key], self.rng)
        # Keep transformer head compatibility with embedding/fusion dim.
        # Set `fusion_dim` for subsequent steps so downstream prediction heads receive the right feature signal.
        fusion_dim = int(out["encoder"].get("fusion_dim", 256))
        # Call `int` and use its result in later steps so downstream prediction heads receive the right feature signal.
        head_choices = [h for h in search_space["transformer"]["heads"] if fusion_dim % int(h) == 0]
        # Branch on `not head_choices` to choose the correct output computation path.
        if not head_choices:
            # Compute `head_choices` as an intermediate representation used by later output layers.
            head_choices = [2]
        # Branch on `fusion_dim % int(out["transformer"]["heads"]) != 0` to choose the correct output computation path.
        if fusion_dim % int(out["transformer"]["heads"]) != 0:
            # Call `int` and use its result in later steps so downstream prediction heads receive the right feature signal.
            out["transformer"]["heads"] = int(_choice(head_choices, self.rng))
        # Return `out` as this function's contribution to downstream output flow.
        return out

    # Define a reusable pipeline function whose outputs feed later steps.
    def evolve(self, evaluate_fn, search_space=None, on_generation_end=None):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Compute `search_space` as an intermediate representation used by later output layers.
        search_space = copy.deepcopy(search_space) if search_space is not None else default_search_space()
        # Set `population` for subsequent steps so downstream prediction heads receive the right feature signal.
        population = self._random_population(search_space)
        # Compute `history` as an intermediate representation used by later output layers.
        history = []

        # Iterate over `range(1, self.generations + 1)` so each item contributes to final outputs/metrics.
        for gen in range(1, self.generations + 1):
            # Set `population` for subsequent steps so downstream prediction heads receive the right feature signal.
            population = self._evaluate(population, evaluate_fn)
            # Set `best` for subsequent steps so downstream prediction heads receive the right feature signal.
            best = population[0]
            # Call `history.append` and use its result in later steps so downstream prediction heads receive the right feature signal.
            history.append(
                {
                    "generation": gen,
                    "best_fitness": float(best.fitness),
                    "best_metrics": copy.deepcopy(best.metrics),
                    "best_architecture": copy.deepcopy(best.architecture),
                }
            )
            # Branch on `on_generation_end is not None` to choose the correct output computation path.
            if on_generation_end is not None:
                # Call `on_generation_end` and use its result in later steps so downstream prediction heads receive the right feature signal.
                on_generation_end(history[-1])

            # Set `elites` for subsequent steps so downstream prediction heads receive the right feature signal.
            elites = [copy.deepcopy(c) for c in population[: self.elite_count]]
            # Compute `next_pop` as an intermediate representation used by later output layers.
            next_pop = elites
            # Repeat computation while condition holds, affecting convergence and final outputs.
            # Repeat while `len(next_pop) < self.population_size` so iterative updates converge to stable outputs.
            while len(next_pop) < self.population_size:
                # Set `p1` for subsequent steps so downstream prediction heads receive the right feature signal.
                p1 = self._tournament_pick(population)
                # Set `p2` for subsequent steps so downstream prediction heads receive the right feature signal.
                p2 = self._tournament_pick(population)
                # Compute `child_arch` as an intermediate representation used by later output layers.
                child_arch = self._crossover(p1.architecture, p2.architecture)
                # Compute `child_arch` as an intermediate representation used by later output layers.
                child_arch = self._mutate(child_arch, search_space)
                # Call `next_pop.append` and use its result in later steps so downstream prediction heads receive the right feature signal.
                next_pop.append(NASCandidate(architecture=child_arch))
            # Set `population` for subsequent steps so downstream prediction heads receive the right feature signal.
            population = next_pop

        # Set `population` for subsequent steps so downstream prediction heads receive the right feature signal.
        population = self._evaluate(population, evaluate_fn)
        # Set `best` for subsequent steps so downstream prediction heads receive the right feature signal.
        best = population[0]
        # Return `{` as this function's contribution to downstream output flow.
        return {
            "best_architecture": copy.deepcopy(best.architecture),
            "best_metrics": copy.deepcopy(best.metrics),
            "best_fitness": float(best.fitness),
            "history": history,
        }


# Define class `MicroNASController` to package related logic in the prediction pipeline.
class MicroNASController(torch.nn.Module):
    """
    Backward-compatibility stub for older imports.
    """

    # Define a reusable pipeline function whose outputs feed later steps.
    def __init__(self, *args, **kwargs):
        """Executes this routine and returns values used by later pipeline output steps."""
        # Execute this statement so downstream prediction heads receive the right feature signal.
        del args, kwargs
        # Call `super` and use its result in later steps so downstream prediction heads receive the right feature signal.
        super().__init__()
        # Compute `self._zero` as an intermediate representation used by later output layers.
        self._zero = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    # Define a loss computation that guides optimization toward better outputs.
    def arch_entropy_loss(self):
        """Computes an objective term that steers optimization and therefore changes final predictions."""
        # Return `self._zero.sum() * 0.0` as this function's contribution to downstream output flow.
        return self._zero.sum() * 0.0

