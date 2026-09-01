"""Scoring: pairwise semantic distance, run policies, and interpretive baselines.

The score produced here is the raw mean pairwise cosine distance between the
embeddings of a run's first `n_use` valid words. It is NOT the published DAT
score: see `dat_score` for why the x100 convention is deliberately omitted.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from datbench.validate import WordCheck

# The embedder is injected rather than constructed here, so scoring is testable
# with a fixture and never touches the network.
EmbedFn = Callable[[Sequence[str]], "dict[str, list[float]]"]

__all__ = [
    "COMMON_NOUNS",
    "DEFAULT_CATEGORIES",
    "BaselineStats",
    "ScoreResult",
    "category_floor",
    "cosine_distance",
    "dat_score",
    "random_baseline",
    "score_run",
]


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """1 - cosine similarity, range 0..2."""
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    if va.ndim != 1 or vb.ndim != 1:
        raise ValueError(f"expected 1-D vectors, got shapes {va.shape} and {vb.shape}")
    if va.shape != vb.shape:
        raise ValueError(f"dimension mismatch: {va.shape[0]} vs {vb.shape[0]}")

    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    # Raise rather than return nan: a nan would propagate silently into the mean
    # and quietly delete a whole run's score.
    if na == 0.0 or nb == 0.0:
        raise ValueError("zero-norm vector has no direction, cosine distance undefined")

    sim = float(np.dot(va, vb) / (na * nb))
    # Float error can push an identical pair to 1.0000000000000002, which would
    # come back as a tiny negative distance.
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def dat_score(
    words: Sequence[str],
    vecs: Mapping[str, Sequence[float]],
    *,
    n_use: int = 7,
) -> float | None:
    """Mean pairwise cosine distance over the first `n_use` embedded words.

    With the default n_use=7 that is C(7,2)=21 pairs. Words absent from `vecs`
    are skipped and do not consume the n_use budget. Returns None if fewer than
    two words have vectors.

    Returns the RAW mean distance, NOT x100. The x100 convention belongs to
    GloVe-scale DAT scores; we are on local embeddings and must not imply
    comparability with the published human norms.
    """
    if n_use < 2:
        raise ValueError(f"n_use must be >= 2, got {n_use}")

    used: list[Sequence[float]] = []
    for w in words:
        v = vecs.get(w)
        if v is None:
            continue
        used.append(v)
        if len(used) == n_use:
            break

    if len(used) < 2:
        return None

    dists = [cosine_distance(a, b) for a, b in itertools.combinations(used, 2)]
    return sum(dists) / len(dists)


@dataclass(frozen=True)
class ScoreResult:
    score: float | None
    n_candidates: int
    n_valid: int
    n_words_used: int
    valid_rate: float
    scored: bool
    reason: str | None


def score_run(
    checks: Sequence[WordCheck],
    vecs: Mapping[str, Sequence[float]],
    *,
    policy: str = "strict",
    n_use: int = 7,
    min_words: int = 4,
) -> ScoreResult:
    """Apply a scoring policy to one run's validated words.

    policy="strict":  needs >= n_use valid, embedded words.
    policy="lenient": needs >= min_words, and scores on however many are
                      available (n_words_used records it, because fewer words
                      means fewer pairs and a different variance).

    A refused run gets score=None and a reason. It is never score 0.0 -- that
    would reward invalidity with a floor instead of an exclusion.
    """
    if policy == "strict":
        need = n_use
    elif policy == "lenient":
        need = min_words
    else:
        raise ValueError(f"unknown policy {policy!r}, expected 'strict' or 'lenient'")
    if need < 2:
        raise ValueError(f"policy {policy!r} needs a threshold of >= 2 words, got {need}")

    n_candidates = len(checks)
    valid = [c.clean for c in checks if c.valid]
    n_valid = len(valid)
    valid_rate = n_valid / n_candidates if n_candidates else 0.0

    def refuse(reason: str, n_words_used: int = 0) -> ScoreResult:
        return ScoreResult(
            score=None,
            n_candidates=n_candidates,
            n_valid=n_valid,
            n_words_used=n_words_used,
            valid_rate=valid_rate,
            scored=False,
            reason=reason,
        )

    if n_valid < need:
        return refuse(f"policy {policy!r} needs >= {need} valid words, got {n_valid}")

    embedded = [w for w in valid if w in vecs][:n_use]
    n_words_used = len(embedded)
    # Coverage is a separate failure from validity: a strict score computed on 5
    # words is not a strict score, so say which of the two conditions failed.
    if n_words_used < need:
        return refuse(
            f"policy {policy!r} needs >= {need} embedded words, but only "
            f"{n_words_used} of {n_valid} valid words have vectors",
            n_words_used=n_words_used,
        )

    try:
        score = dat_score(embedded, vecs, n_use=n_use)
    except ValueError as exc:
        # A degenerate vector is a data problem with this run, not a reason to
        # abort the whole scoring stage.
        return refuse(f"embedding error: {exc}", n_words_used=n_words_used)

    if score is None:
        return refuse(
            f"fewer than 2 of {n_words_used} words could be embedded",
            n_words_used=n_words_used,
        )

    return ScoreResult(
        score=score,
        n_candidates=n_candidates,
        n_valid=n_valid,
        n_words_used=n_words_used,
        valid_rate=valid_rate,
        scored=True,
        reason=None,
    )


@dataclass(frozen=True)
class BaselineStats:
    mean: float
    sd: float
    n: int
    k: int
    p05: float
    p50: float
    p95: float
    seed: int
    # Kept last with a default so the contract's field order still constructs
    # positionally; percentile_of needs the empirical distribution, and
    # out/baselines.json does not carry it.
    draws: tuple[float, ...] = field(default=())

    def percentile_of(self, score: float) -> float:
        """Fraction of the baseline distribution below `score`, in 0..1."""
        if self.draws:
            below = sum(1 for d in self.draws if d < score)
            return below / len(self.draws)
        # Reconstructed from baselines.json, where the draws are not stored: fall
        # back to a normal approximation instead of refusing to answer.
        if self.sd <= 0.0:
            return math.nan
        z = (score - self.mean) / self.sd
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def z_of(self, score: float) -> float:
        if self.sd <= 0.0:
            # A zero-spread baseline makes z undefined; nan says so instead of
            # implying the score sits exactly at chance.
            return math.nan
        return (score - self.mean) / self.sd


def random_baseline(
    embed_fn: EmbedFn,
    vocab: Sequence[str] | None = None,
    *,
    n_draws: int = 1000,
    k: int = 7,
    seed: int = 0,
) -> BaselineStats:
    """Chance distribution for one embedder: draw k random common nouns, score, repeat.

    This is what makes an arbitrary-scale number interpretable. `vocab` defaults
    to COMMON_NOUNS. Words the embedder cannot represent are dropped from the
    pool, so a partial embedder degrades the pool rather than the statistics.
    """
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}")
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    words = list(dict.fromkeys(w for w in (vocab if vocab is not None else COMMON_NOUNS) if w))
    if not words:
        raise ValueError("vocab is empty")

    # One embed call for the whole vocab: the draws are sampled from it, so
    # n_draws=1000 costs one round trip, not a thousand.
    vecs = embed_fn(words)
    pool = [w for w in words if w in vecs and any(vecs[w])]
    if len(pool) < k:
        raise ValueError(f"need >= {k} embeddable vocab words for the baseline, got {len(pool)}")

    rng = random.Random(seed)  # never the global module: the baseline must be reproducible
    draws: list[float] = []
    for _ in range(n_draws):
        s = dat_score(rng.sample(pool, k), vecs, n_use=k)
        if s is not None:
            draws.append(s)

    arr = np.asarray(draws, dtype=np.float64)
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return BaselineStats(
        mean=float(arr.mean()),
        sd=sd,
        n=int(arr.size),
        k=k,
        p05=float(np.percentile(arr, 5)),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        seed=seed,
        draws=tuple(draws),
    )


def category_floor(
    embed_fn: EmbedFn,
    categories: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, float]:
    """Score semantically-tight sets (7 animals, 7 tools) -> a low anchor.

    Defaults to DEFAULT_CATEGORIES. A category whose words the embedder cannot
    represent is omitted rather than reported as a number that is not a score.
    """
    cats = DEFAULT_CATEGORIES if categories is None else categories
    words = list(dict.fromkeys(w for ws in cats.values() for w in ws if w))
    if not words:
        return {}

    vecs = embed_fn(words)
    out: dict[str, float] = {}
    for name, ws in cats.items():
        # n_use = len(ws) so a category of any size is scored on all of it.
        s = dat_score(ws, vecs, n_use=max(2, len(ws)))
        if s is not None:
            out[name] = s
    return out


# Seven common concrete nouns per category. The floor these produce is the
# "maximally unimaginative but still valid" anchor a real run should beat.
DEFAULT_CATEGORIES: dict[str, list[str]] = {
    "animals": ["dog", "cat", "horse", "cow", "sheep", "goat", "pig"],
    "tools": ["hammer", "saw", "drill", "wrench", "pliers", "screwdriver", "chisel"],
    "colours": ["red", "blue", "green", "yellow", "orange", "purple", "brown"],
    "furniture": ["table", "chair", "bed", "sofa", "desk", "shelf", "cabinet"],
}

# Default vocab for random_baseline. Curated on purpose: /usr/share/dict/words is
# web2, whose archaic and technical entries would put "chance" somewhere no model
# actually operates, making the percentile meaningless.
COMMON_NOUNS: list[str] = [
    # animals
    "dog", "cat", "horse", "cow", "sheep", "goat", "pig", "chicken", "duck",
    "goose", "rabbit", "mouse", "rat", "squirrel", "deer", "bear", "wolf", "fox",
    "lion", "tiger", "elephant", "monkey", "whale", "dolphin", "shark", "salmon",
    "frog", "snake", "lizard", "turtle", "spider", "ant", "bee", "moth",
    "butterfly", "beetle", "worm", "snail", "crab", "lobster", "shrimp", "eagle",
    "hawk", "owl", "crow", "sparrow", "pigeon", "parrot", "penguin", "camel",
    "donkey", "bat", "seal", "otter",
    # body
    "arm", "leg", "hand", "foot", "finger", "thumb", "toe", "head", "hair",
    "eye", "ear", "nose", "mouth", "tooth", "tongue", "lip", "chin", "neck",
    "shoulder", "elbow", "wrist", "knee", "ankle", "back", "chest", "heart",
    "lung", "bone", "skin", "blood", "brain",
    # food and drink
    "bread", "butter", "cheese", "milk", "egg", "meat", "bacon", "fish", "rice",
    "bean", "corn", "wheat", "flour", "sugar", "salt", "pepper", "honey", "jam",
    "soup", "salad", "sandwich", "pizza", "cake", "pie", "cookie", "candy",
    "chocolate", "apple", "orange", "banana", "grape", "lemon", "peach", "pear",
    "cherry", "melon", "potato", "tomato", "onion", "garlic", "carrot",
    "cabbage", "lettuce", "pumpkin", "mushroom", "almond", "coffee", "tea",
    "juice", "water", "wine", "beer", "oil", "vinegar", "noodle",
    # household
    "table", "chair", "bed", "sofa", "desk", "shelf", "cabinet", "drawer",
    "mirror", "lamp", "candle", "clock", "carpet", "curtain", "pillow",
    "blanket", "mattress", "towel", "soap", "brush", "comb", "basket", "bucket",
    "box", "bag", "jar", "bottle", "cup", "mug", "plate", "bowl", "spoon",
    "fork", "knife", "pot", "pan", "kettle", "oven", "stove", "sink", "broom",
    "ladder",
    # building
    "house", "roof", "wall", "floor", "ceiling", "door", "window", "gate",
    "fence", "stair", "hall", "kitchen", "attic", "basement", "garage", "porch",
    "chimney", "brick", "cement", "plank", "beam", "nail", "screw", "hinge",
    "lock", "key", "pipe", "wire", "cable", "tile",
    # tools and machines
    "hammer", "saw", "drill", "wrench", "pliers", "screwdriver", "chisel", "axe",
    "shovel", "spade", "rake", "anvil", "clamp", "ruler", "tape", "needle",
    "thread", "scissors", "pin", "button", "zipper", "rope", "chain", "hook",
    "wheel", "gear", "spring", "lever", "pulley", "magnet", "battery", "engine",
    "motor", "pump", "valve", "piston",
    # clothing
    "shirt", "coat", "jacket", "sweater", "dress", "skirt", "sock", "shoe",
    "boot", "sandal", "hat", "cap", "glove", "scarf", "belt", "necklace",
    "bracelet", "pocket", "collar", "sleeve", "hood", "apron",
    # plants and landscape
    "tree", "oak", "pine", "maple", "birch", "leaf", "branch", "root", "trunk",
    "bark", "flower", "rose", "tulip", "daisy", "lily", "grass", "weed", "moss",
    "fern", "bush", "vine", "seed", "forest", "field", "meadow", "hill",
    "mountain", "valley", "cliff", "cave", "rock", "stone", "pebble", "sand",
    "soil", "mud", "clay", "dust", "river", "stream", "lake", "pond", "sea",
    "ocean", "wave", "beach", "shore", "island", "desert", "swamp", "glacier",
    "volcano", "canyon",
    # sky and weather
    "sun", "moon", "star", "planet", "comet", "cloud", "rain", "snow", "hail",
    "ice", "frost", "fog", "mist", "wind", "storm", "thunder", "lightning",
    "rainbow", "sky", "shadow", "fire", "smoke", "ash", "flame", "steam",
    # vehicles
    "car", "truck", "bus", "train", "tram", "bicycle", "motorcycle", "wagon",
    "cart", "sled", "boat", "ship", "canoe", "kayak", "raft", "ferry",
    "submarine", "plane", "jet", "helicopter", "rocket", "tractor", "taxi",
    "van", "tire", "sail", "anchor", "oar", "paddle", "mast", "rudder",
    "propeller",
    # places
    "city", "town", "village", "street", "road", "path", "bridge", "tunnel",
    "park", "garden", "farm", "barn", "mill", "factory", "shop", "market",
    "bank", "school", "church", "temple", "castle", "tower", "museum",
    "library", "hospital", "hotel", "restaurant", "theater", "stadium",
    "prison", "harbor", "airport", "station",
    # made objects
    "book", "paper", "pencil", "pen", "ink", "map", "letter", "envelope",
    "stamp", "newspaper", "magazine", "poster", "picture", "photograph",
    "painting", "statue", "coin", "money", "wallet", "purse", "ticket",
    "calendar", "notebook", "folder", "ribbon", "toy", "doll", "ball", "kite",
    "drum", "guitar", "piano", "flute", "violin", "trumpet", "horn", "bell",
    "whistle", "radio", "camera", "telephone", "computer", "keyboard", "screen",
    "printer", "television", "umbrella", "fan", "heater", "telescope",
    "microscope", "compass", "lantern", "torch", "flag", "tent", "net", "trap",
    "cage", "saddle", "harness", "leash", "nest", "hive",
    # materials
    "feather", "fur", "wool", "leather", "silk", "cotton", "linen", "rubber",
    "plastic", "metal", "iron", "steel", "copper", "silver", "gold", "coal",
    "sponge", "glass", "wax", "paint", "glue", "chalk", "soot", "straw",
]
