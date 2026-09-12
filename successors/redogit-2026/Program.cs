namespace FirstNeuralNetwork.Redo;

public sealed class LogisticNeuron
{
    private readonly double[] _weights;

    public LogisticNeuron(int inputCount)
    {
        if (inputCount <= 0) throw new ArgumentOutOfRangeException(nameof(inputCount));
        _weights = new double[inputCount];
    }

    public double Bias { get; private set; }

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
        if (learningRate <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate));
        if (epochs <= 0) throw new ArgumentOutOfRangeException(nameof(epochs));

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            foreach (var sample in samples)
            {
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

public static class Program
{
    public static int Main()
    {
        var samples = new[]
        {
            new Sample([0, 0], 0),
            new Sample([1, 0], 1),
            new Sample([0, 1], 1),
            new Sample([1, 1], 1)
        };

        var neuron = new LogisticNeuron(inputCount: 2);
        neuron.Train(samples, learningRate: 0.2, epochs: 4_000);

        var allCorrect = true;
        foreach (var sample in samples)
        {
            var probability = neuron.Predict(sample.Inputs);
            var predicted = probability >= 0.5 ? 1.0 : 0.0;
            var correct = predicted == sample.Expected;
            allCorrect &= correct;

            Console.WriteLine($"[{string.Join(',', sample.Inputs)}] => {probability:F6} => {predicted:0} {(correct ? "PASS" : "FAIL")}");
        }

        return allCorrect ? 0 : 1;
    }
}
