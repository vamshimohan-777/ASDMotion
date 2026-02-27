import copy
import random
from dataclasses import dataclass

import numpy as np
import torch


def default_search_space():
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


def _choice(values, rng):
    values = list(values)
    if not values:
        raise ValueError("Search-space candidate list is empty.")
    return copy.deepcopy(rng.choice(values))


def sample_architecture(search_space, rng):
    return {
        "encoder": {k: _choice(v, rng) for k, v in search_space["encoder"].items()},
        "transformer": {k: _choice(v, rng) for k, v in search_space["transformer"].items()},
        "window": {k: _choice(v, rng) for k, v in search_space["window"].items()},
    }


def compute_fitness(metrics):
    """
    Multi-objective fitness:
      0.40 * AUC
    + 0.25 * Sensitivity@90Specificity
    + 0.15 * Calibration quality
    + 0.10 * Cross-validation stability
    + 0.10 * Efficiency score
    """
    auc = float(metrics.get("auc", 0.0))
    sens90 = float(metrics.get("sens_at_90_spec", 0.0))
    calibration = float(metrics.get("calibration_quality", 0.0))
    stability = float(metrics.get("cv_stability", 0.0))
    # efficiency_penalty in [0,1], lower is better.
    eff_pen = float(metrics.get("efficiency_penalty", 1.0))
    eff_score = float(np.clip(1.0 - eff_pen, 0.0, 1.0))

    score = (
        0.40 * auc
        + 0.25 * sens90
        + 0.15 * calibration
        + 0.10 * stability
        + 0.10 * eff_score
    )
    return float(score)


@dataclass
class NASCandidate:
    architecture: dict
    metrics: dict | None = None
    fitness: float = -1e9


class MicroGeneticNAS:
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
        self.population_size = int(max(4, population_size))
        self.generations = int(max(1, generations))
        self.tournament_size = int(max(2, tournament_size))
        self.mutation_rate = float(np.clip(mutation_rate, 0.0, 1.0))
        self.crossover = bool(crossover)
        self.elite_count = int(max(1, elite_count))
        self.rng = random.Random(int(seed))

    def _random_population(self, search_space):
        return [
            NASCandidate(architecture=sample_architecture(search_space, self.rng))
            for _ in range(self.population_size)
        ]

    def _evaluate(self, population, evaluate_fn):
        for cand in population:
            if cand.metrics is None:
                metrics = evaluate_fn(copy.deepcopy(cand.architecture))
                cand.metrics = metrics
                cand.fitness = compute_fitness(metrics)
        population.sort(key=lambda c: c.fitness, reverse=True)
        return population

    def _tournament_pick(self, population):
        k = min(self.tournament_size, len(population))
        contenders = self.rng.sample(population, k=k)
        contenders.sort(key=lambda c: c.fitness, reverse=True)
        return contenders[0]

    def _crossover(self, arch_a, arch_b):
        if not self.crossover:
            return copy.deepcopy(arch_a)
        child = {"encoder": {}, "transformer": {}, "window": {}}
        for section in child.keys():
            keys = set(arch_a.get(section, {}).keys()) | set(arch_b.get(section, {}).keys())
            for k in keys:
                if self.rng.random() < 0.5:
                    child[section][k] = copy.deepcopy(arch_a.get(section, {}).get(k))
                else:
                    child[section][k] = copy.deepcopy(arch_b.get(section, {}).get(k))
        return child

    def _mutate(self, arch, search_space):
        out = copy.deepcopy(arch)
        for section, params in out.items():
            for key in list(params.keys()):
                if self.rng.random() < self.mutation_rate:
                    out[section][key] = _choice(search_space[section][key], self.rng)
        # Keep transformer head compatibility with embedding/fusion dim.
        fusion_dim = int(out["encoder"].get("fusion_dim", 256))
        head_choices = [h for h in search_space["transformer"]["heads"] if fusion_dim % int(h) == 0]
        if not head_choices:
            head_choices = [2]
        if fusion_dim % int(out["transformer"]["heads"]) != 0:
            out["transformer"]["heads"] = int(_choice(head_choices, self.rng))
        return out

    def evolve(self, evaluate_fn, search_space=None, on_generation_end=None):
        search_space = copy.deepcopy(search_space) if search_space is not None else default_search_space()
        population = self._random_population(search_space)
        history = []

        for gen in range(1, self.generations + 1):
            population = self._evaluate(population, evaluate_fn)
            best = population[0]
            history.append(
                {
                    "generation": gen,
                    "best_fitness": float(best.fitness),
                    "best_metrics": copy.deepcopy(best.metrics),
                    "best_architecture": copy.deepcopy(best.architecture),
                }
            )
            if on_generation_end is not None:
                on_generation_end(history[-1])

            elites = [copy.deepcopy(c) for c in population[: self.elite_count]]
            next_pop = elites
            while len(next_pop) < self.population_size:
                p1 = self._tournament_pick(population)
                p2 = self._tournament_pick(population)
                child_arch = self._crossover(p1.architecture, p2.architecture)
                child_arch = self._mutate(child_arch, search_space)
                next_pop.append(NASCandidate(architecture=child_arch))
            population = next_pop

        population = self._evaluate(population, evaluate_fn)
        best = population[0]
        return {
            "best_architecture": copy.deepcopy(best.architecture),
            "best_metrics": copy.deepcopy(best.metrics),
            "best_fitness": float(best.fitness),
            "history": history,
        }


class MicroNASController(torch.nn.Module):
    """
    Backward-compatibility stub for older imports.
    """

    def __init__(self, *args, **kwargs):
        del args, kwargs
        super().__init__()
        self._zero = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def arch_entropy_loss(self):
        return self._zero.sum() * 0.0

