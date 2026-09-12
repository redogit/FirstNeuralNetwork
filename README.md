# FirstNeuralNetwork — REDO

This repository preserves the original C# neural-network experiment and now carries a clean successor beside it.

## Current path

- `Class1.cs` and the original project remain historical inputs.
- `successors/redogit-2026/` is the current minimal rebuild.
- The successor removes invalid parallel-training syntax and keeps mutation deterministic and sequential.
- The successor is intentionally small: one logistic neuron, one OR dataset, explicit loss/error behavior, and a process exit code that acts as a smoke test.

## Run the successor

```bash
cd successors/redogit-2026
dotnet run
```

The program exits with code `0` only when all four OR cases classify correctly.

## REDOGIT rule

Do not erase the predecessor. Observe it, keep its lineage, rebuild the smallest coherent successor, test the successor, and only then extend it.

See [`REDOGIT.md`](REDOGIT.md).
