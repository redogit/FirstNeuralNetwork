# FirstNeuralNetwork REDO — 2026

This is the current successor to the repository's original single-neuron experiment.

## Why it exists

The predecessor captures the idea but mixes generic floating-point experimentation with an invalid `Parallel.For` construct and shared mutable training state. This successor keeps the useful minimum and removes those confounds.

## Contract

Input: two binary values.

Output: a logistic probability and binary OR classification.

Training: deterministic sequential gradient descent.

Completion check: all four OR truth-table rows must classify correctly; otherwise the process exits nonzero.

## Run

```bash
dotnet run --project FirstNeuralNetwork.Redo.csproj
```

## Next bounded step

Only introduce a second layer when the target requires a non-linearly-separable function such as XOR. That change should arrive as another successor with its own executable check rather than silently changing this baseline.
