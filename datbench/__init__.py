"""dat-bench: benchmark LLMs on the Divergent Association Task.

Scores come from local LM Studio embeddings, not the GloVe 840B-300d vectors used
by the published DAT. They are therefore on an arbitrary scale and are NOT
comparable to the published human norms (mean ~78, range ~50-95). Interpret every
score against the per-embedder random-noun baseline in out/baselines.json.

See CONTRACT.md for the module interface authority.
"""

__version__ = "0.1.0"
