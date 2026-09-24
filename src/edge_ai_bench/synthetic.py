"""Deterministic synthetic ECG-like data for plumbing tests only."""

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class SyntheticBatch:
    signals: list[list[list[float]]]
    labels: list[list[int]]


def make_batch(samples: int, channels: int, timesteps: int, labels: int, seed: int) -> SyntheticBatch:
    generator = random.Random(seed)
    signals: list[list[list[float]]] = []
    targets: list[list[int]] = []
    for sample_index in range(samples):
        sample = []
        for channel_index in range(channels):
            frequency = 1.0 + ((sample_index + channel_index) % 4) * 0.25
            sample.append([
                math.sin(2.0 * math.pi * frequency * step / timesteps)
                + generator.uniform(-0.01, 0.01)
                for step in range(timesteps)
            ])
        signals.append(sample)
        targets.append([(sample_index + label_index) % 2 for label_index in range(labels)])
    return SyntheticBatch(signals=signals, labels=targets)
