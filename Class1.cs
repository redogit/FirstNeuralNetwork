using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
public class Neuron<T> where T : IFloatingPoint<T>
{
    public T[] Weights { get; }
    public T Bias { get; set; }

    public Neuron(T[] weights, T bias)
    {
        Weights = weights;
        Bias = bias;
    }

    public T ComputeOutput(IReadOnlyList<T> inputs)
    {
        if (inputs.Count != Weights.Length)
        {
            throw new ArgumentException("The number of inputs must match the number of weights.");
        }

        T weightedSum = inputs.Zip(Weights, (x, w) => x * w).Aggregate((x, y) => x + y) + Bias;
        return Sigmoid(weightedSum);
    }

    public void Train(IReadOnlyList<(T[] inputs, T output)> trainingData, T learningRate, int epochs)
    {
        var returnObj = Parallel.For(int i = 0; i < epochs; i++)
        {
            foreach (var (inputs, desiredOutput) in trainingData)
            {
                T actualOutput = ComputeOutput(inputs);
                T error = desiredOutput - actualOutput;

                for (int j = 0; j < Weights.Length; j++)
                {
                    Weights[j] += learningRate * error * inputs[j];
                }

                Bias += learningRate * error;
            }
        }
    }

    private static T Sigmoid(T x)
    {
        return T.One / (T.One + T.Exp(-x));
    }
}

public static class Program
{
    public static void Main()
    {
        var neuron = new Neuron<float>(new[] { 0.0f, 0.0f }, 0.0f);

        var trainingData = new[]
        {
            (inputs: new[] { 0.0f, 0.0f }, output: 0.0f),
            (inputs: new[] { 1.0f, 0.0f }, output: 1.0f),
            (inputs: new[] { 0.0f, 1.0f }, output: 1.0f),
            (inputs: new[] { 1.0f, 1.0f }, output: 1.0f),
        };

        neuron.Train(trainingData, learningRate: 0.1f, epochs: 10000);

        Console.WriteLine(neuron.ComputeOutput(new[] { 0.0f, 0.0f })); // Outputs ~0
        Console.WriteLine(neuron.ComputeOutput(new[] { 1.0f, 0.0f })); // Outputs ~1
        Console.WriteLine(neuron.ComputeOutput(new[] { 0.0f, 1.0f })); // Outputs ~1
        Console.WriteLine(neuron.ComputeOutput(new[] { 1.0f, 1.0f })); // Outputs ~1
    }
}
