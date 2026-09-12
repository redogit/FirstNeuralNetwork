namespace FirstNeuralNetwork.Core;

public sealed class LogisticNeuron
{
    private readonly double[] _weights;

    public LogisticNeuron(int inputCount)
    {
        if (inputCount <= 0) throw new ArgumentOutOfRangeException(nameof(inputCount));
        _weights = new double[inputCount];
    }

    public double Bias { get; private set; }
    public IReadOnlyList<double> Weights => _weights;

    public double Predict(ReadOnlySpan<double> inputs)
    {
        if (inputs.Length != _weights.Length)
            throw new ArgumentException("Input count must match weight count.", nameof(inputs));

        var z = Bias;
        for (var i = 0; i < _weights.Length; i++)
            z += _weights[i] * inputs[i];

        return 1.0 / (1.0 + Math.Exp(-z));
    }

    public void Train(IReadOnlyList<Sample> samples, double learningRate, int epochs)
    {
        if (samples.Count == 0) throw new ArgumentException("At least one sample is required.", nameof(samples));
        if (learningRate <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate));
        if (epochs <= 0) throw new ArgumentOutOfRangeException(nameof(epochs));

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            foreach (var sample in samples)
            {
                if (sample.Inputs.Length != _weights.Length)
                    throw new ArgumentException("Every sample must match the neuron input count.", nameof(samples));

                var prediction = Predict(sample.Inputs);
                var gradient = prediction - sample.Expected;

                for (var i = 0; i < _weights.Length; i++)
                    _weights[i] -= learningRate * gradient * sample.Inputs[i];

                Bias -= learningRate * gradient;
            }
        }
    }
}

public sealed record Sample(double[] Inputs, double Expected);
