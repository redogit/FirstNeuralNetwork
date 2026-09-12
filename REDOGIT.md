# REDOGIT

`REDOGIT` means rebuild without falsifying history.

For this repository:

1. **Input** — the original neural-network experiment and its compile/runtime assumptions.
2. **Difference** — preserve the useful idea; remove accidental complexity and invalid syntax.
3. **Successor** — a minimal, deterministic, buildable implementation under `successors/redogit-2026/`.
4. **Check** — executable smoke checks must distinguish success from failure.
5. **Lineage** — predecessor files remain in Git history and are not rewritten to pretend the successor was always there.
6. **Next** — only add layers, activations, optimizers, datasets, or parallelism when a measured need appears.

A redo is not a reset. It is a verified successor with provenance.
