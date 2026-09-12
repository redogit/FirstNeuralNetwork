using FirstNeuralNetwork.Core;

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
