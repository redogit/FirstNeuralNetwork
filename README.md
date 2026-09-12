# FirstNeuralNetwork — REDO

This repository preserves the original C# neural-network experiment and carries explicit successors beside it.

## Current build — v2

From the repository root:

```bash
dotnet build REDOGIT.slnx --configuration Release
dotnet run --project successors/redogit-2026-v2/FirstNeuralNetwork.SelfCheck/FirstNeuralNetwork.SelfCheck.csproj --configuration Release --no-build
```

`REDOGIT.slnx` is the current solution entry point. v2 separates reusable behavior from verification:

- `FirstNeuralNetwork.Core` — reusable deterministic logistic neuron and training API;
- `FirstNeuralNetwork.SelfCheck` — executable OR truth-table contract.

The program exits with code `0` only when all four OR cases classify correctly. GitHub Actions builds this same root solution and runs the same verifier.

## Preserved predecessors

- The original solution, root source files, and project remain historical inputs.
- `successors/redogit-2026/` is the first verified REDOGIT successor and remains intact as the predecessor to v2.
- The separate generated **Adaptive Connectivity Diagnostic Network** C++20 experiment remains under [`successors/`](successors/README.md). Its reversible archive transport is `successors/adaptive-connectivity-diagnostic-network-cpp.zip.b64`, with decoded ZIP SHA-256 `f1146e15b4ca8e0bc6060d19fe219876f883f223898f64279c46c538145905e7`.

## REDOGIT rule

Do not erase the predecessor. Observe it, keep its lineage, rebuild the smallest coherent successor, test the successor, and only then extend it.

See [`REDOGIT.md`](REDOGIT.md).
