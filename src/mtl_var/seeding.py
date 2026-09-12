"""The single seeding utility for the whole package.

Two variants existed across the research scripts - `walk_forward.set_seeds()` (hardcoded to
SEED=42) and `verify_seed_sensitivity.set_seeds_custom(seed)` (parameterised). They seed the
same three RNGs in the same order, so this unifies them: `set_seeds()` with no argument is
byte-for-byte the old `set_seeds()`, and `set_seeds(43)` is the old `set_seeds_custom(43)`.

That distinction mattered historically: reusing a helper that re-seeded internally to a
hardcoded 42 silently nullified every alternate seed in an early version of the
seed-sensitivity study (RESULTS.md section 11). Callers that need a specific seed must call
this immediately before building the model, with nothing RNG-consuming in between.
"""
import random

import numpy as np

SEED = 42


def set_seeds(seed: int = SEED) -> int:
    """Seed python `random`, numpy, and TensorFlow. Returns the seed used."""
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return seed
