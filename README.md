# FirstNeuralNetwork — REDO

This repository preserves the original C# neural-network experiment and carries explicit successors beside it.

## Current build

From the repository root:

```bash
dotnet build REDOGIT.slnx --configuration Release
dotnet run --project successors/redogit-2026/FirstNeuralNetwork.Redo.csproj --configuration Release --no-build
```

`REDOGIT.slnx` is the current solution entry point. The older solution, root source files, and project remain historical inputs rather than being silently rewritten.

The current .NET 10 successor removes invalid parallel-training syntax and keeps training deterministic and sequential. It is intentionally small: one logistic neuron, one OR dataset, and an executable completion condition. The program exits with code `0` only when all four OR cases classify correctly.

## Generated ACDN C++ successor

A separate generated successor, the **Adaptive Connectivity Diagnostic Network**, remains preserved under [`successors/`](successors/README.md) as an exact reversible archive transport:

`successors/adaptive-connectivity-diagnostic-network-cpp.zip.b64`

Decoded ZIP SHA-256:

`f1146e15b4ca8e0bc6060d19fe219876f883f223898f64279c46c538145905e7`

ACDN is a separate C++20 experiment in sparse modular recurrent groups and controlled connection-on / connection-off structural adaptation. It does not replace the original C# project or the smaller `redogit-2026` rebuild.

## REDOGIT rule

Do not erase the predecessor. Observe it, keep its lineage, rebuild the smallest coherent successor, test the successor, and only then extend it.

See [`REDOGIT.md`](REDOGIT.md).
