"""Independent check of the local-embedding scoring premise.

Question: do LM Studio's embedders actually rank semantic spread the way the DAT needs?
If a tight category (7 animals) does not score clearly BELOW random nouns, then local
embeddings cannot support this benchmark and the user needs to know before running it.

Touches no project file. Sequential, cache-free, deliberately small.
"""
import itertools, json, random, urllib.request

BASE = "http://localhost:1234/v1"

TIGHT = {
    "animals":   ["dog", "cat", "horse", "cow", "sheep", "goat", "pig"],
    "tools":     ["hammer", "wrench", "screwdriver", "pliers", "chisel", "drill", "saw"],
    "colours":   ["red", "blue", "green", "yellow", "purple", "orange", "brown"],
}
# The DAT paper's own illustrative contrast: cat/dog are close, cat/thimble far.
PAPER_PAIR = [("cat", "dog"), ("cat", "thimble")]

COMMON_NOUNS = """town river bottle music window candle garden pencil ladder engine
forest mirror pillow basket blanket church market bridge finger pocket shadow silver
button carpet coffee dinner flower guitar hammer island jacket kitchen letter machine
needle office pepper pillow rabbit saddle temple tunnel valley wallet winter yellow
anchor barrel cactus doctor eagle fabric ginger honey insect jungle kettle lantern
meadow napkin orchid parrot quilt ribbon sponge thread umbrella velvet whistle
alcohol balloon canyon desert effort fabric galaxy harbour injury jockey kingdom
liquid mammal nostril outlet parade quarry rescue saliva thunder utopia vacuum
wisdom anxiety brother ceiling destiny economy fantasy gravity harmony identity
journey justice kindness liberty memory mystery opinion passion quality reality
""".split()


def embed(model, words):
    req = urllib.request.Request(
        f"{BASE}/embeddings",
        data=json.dumps({"model": model, "input": list(words)}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    out = {}
    for item in d["data"]:
        out[words[item["index"]]] = item["embedding"]
    return out


def cos_dist(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return 1.0 - dot / (na * nb)


def spread(words, vecs):
    """Mean pairwise cosine distance -- the DAT statistic."""
    pairs = [(a, b) for a, b in itertools.combinations(words, 2) if a in vecs and b in vecs]
    if not pairs:
        return None
    return sum(cos_dist(vecs[a], vecs[b]) for a, b in pairs) / len(pairs)


def main():
    models = [m["id"] for m in json.load(urllib.request.urlopen(f"{BASE}/models", timeout=30))["data"]
              if "embed" in m["id"]]
    print(f"embedders found: {len(models)}\n")

    rng = random.Random(0)
    draws = [rng.sample(COMMON_NOUNS, 7) for _ in range(30)]
    vocab = sorted({w for d in draws for w in d}
                   | {w for ws in TIGHT.values() for w in ws}
                   | {"cat", "dog", "thimble"})

    verdicts = {}
    for model in models:
        vecs = embed(model, vocab)
        dim = len(next(iter(vecs.values())))
        rand_scores = [s for d in draws if (s := spread(d, vecs)) is not None]
        rand_mean = sum(rand_scores) / len(rand_scores)
        rand_min = min(rand_scores)

        print(f"--- {model}  (dim={dim})")
        print(f"    random-noun baseline : mean {rand_mean:.4f}   min {rand_min:.4f}   max {max(rand_scores):.4f}")
        floors = {}
        for name, ws in TIGHT.items():
            floors[name] = spread(ws, vecs)
            print(f"    floor / {name:8s}     : {floors[name]:.4f}")
        for a, b in PAPER_PAIR:
            print(f"    d({a},{b})".ljust(26) + f": {cos_dist(vecs[a], vecs[b]):.4f}")

        # The premise: every tight category must sit below the random baseline, and
        # ideally below its weakest draw. Separation is the headroom the score needs.
        worst_floor = max(floors.values())
        ok = worst_floor < rand_mean
        strict_ok = worst_floor < rand_min
        sep = (rand_mean - worst_floor) / rand_mean * 100
        verdicts[model] = (ok, strict_ok, sep)
        print(f"    => floors below baseline mean: {ok}   below weakest draw: {strict_ok}"
              f"   separation: {sep:.1f}%\n")

    print("=" * 72)
    for m, (ok, strict_ok, sep) in verdicts.items():
        tag = "USABLE" if strict_ok else ("WEAK" if ok else "BROKEN")
        print(f"{tag:7} {m}   (separation {sep:.1f}%)")
    if all(v[1] for v in verdicts.values()):
        print("\nPREMISE HOLDS: every embedder separates tight categories from chance.")
    elif all(v[0] for v in verdicts.values()):
        print("\nPREMISE HOLDS WEAKLY: floors below mean but overlapping the draw range.")
    else:
        print("\nPREMISE FAILS for at least one embedder -- do not use it for scoring.")


if __name__ == "__main__":
    main()
