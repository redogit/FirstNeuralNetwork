# FirstNeuralNetwork — REDO

This repository preserves the original C# neural-network experiment and now carries clean successors beside it.

## Current path

- `Class1.cs` and the original project remain historical inputs.
- `successors/redogit-2026/` is the current minimal rebuild.
- The REDO successor removes invalid parallel-training syntax and keeps mutation deterministic and sequential.
- The REDO successor is intentionally small: one logistic neuron, one OR dataset, explicit loss/error behavior, and a process exit code that acts as a smoke test.

## Run the REDO successor

```bash
cd successors/redogit-2026
dotnet run
```

The program exits with code `0` only when all four OR cases classify correctly.

## Generated ACDN C++ successor

A second generated successor, the **Adaptive Connectivity Diagnostic Network**, is preserved under [`successors/`](successors/README.md) as an exact reversible archive transport:

`successors/adaptive-connectivity-diagnostic-network-cpp.zip.b64`

Decoded ZIP SHA-256:

`f1146e15b4ca8e0bc6060d19fe219876f883f223898f64279c46c538145905e7`

ACDN is a separate C++20 experiment in sparse modular recurrent groups and controlled connection-on / connection-off structural adaptation. It does not replace the original C# project or the smaller `redogit-2026` rebuild.

## REDOGIT rule

Do not erase the predecessor. Observe it, keep its lineage, rebuild the smallest coherent successor, test the successor, and only then extend it.

See [`REDOGIT.md`](REDOGIT.md).
