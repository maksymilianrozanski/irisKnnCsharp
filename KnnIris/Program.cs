using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Globalization;
using System.Linq;
using LaYumba.Functional;
using LaYumba.Functional.Option;

namespace KnnIris
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            const string trainingDataUrl =
                "https://gist.githubusercontent.com/maksymilianrozanski/fa386a964426d8b0b8fac75a026a9dbd/raw/b5aa5f57df07a4e8ecec6ed9a7eeba5ac8fe9aa3/irisTrainingData";
            const string validationDataUrl =
                "https://gist.githubusercontent.com/maksymilianrozanski/3b631b25cccc2baa2731956aaecc7716/raw/1d62a26bdeaf91fe4dcaf56d1fa96de13ab75e5a/irisValidationData";

            var downloader = new CsvDownloader();

            var trainingDataCsv = downloader.DownloadFromUrl(new Uri(trainingDataUrl));
            var validationDataCsv = downloader.DownloadFromUrl(new Uri(validationDataUrl));

            var trainingData = trainingDataCsv
                .Map(Knn.CsvToFeaturesWithLabel).Match(error =>
                {
                    Console.WriteLine(error.Message);
                    Environment.Exit(1);
                    return new List<FeaturesWithLabel>();
                }, labels => labels).ToList();

            var validationData = validationDataCsv
                .Map(Knn.CsvToFeaturesWithLabel).Match(error =>
                {
                    Console.WriteLine(error.Message);
                    Environment.Exit(1);
                    return new List<FeaturesWithLabel>();
                }, labels => labels).ToList();

            var knn3 = Knn.Predict.Apply(trainingData).Apply(3);

            var predictionResults = Knn.PredictAll(knn3, validationData);


            var possibleLabels = Knn.CollectLabels(trainingData).ToList();
            var stats = Knn.PredictionStatistics(possibleLabels, predictionResults);
            Knn.PrintStatistics(stats);

            Console.WriteLine("End");
        }
    }

    public class Knn
    {
        public static double EuclideanDist(IEnumerable<double> first, IEnumerable<double> second) =>
            first.Zip(second)
                .Select(tuple => Math.Pow(tuple.First - tuple.Second, 2))
                .Sum().Pipe(Math.Sqrt);

        public static readonly Func<IEnumerable<FeaturesWithLabel>, int, IEnumerable<double>, string> Predict =
            (trainingData, k, predictorValues) =>
                trainingData.Select(it => (it, EuclideanDist(it.Features, predictorValues)))
                    .OrderBy(it => it.Item2)
                    .Take(k)
                    .GroupBy(it => it.it.Label)
                    .OrderByDescending(it => it.Count())
                    .Select(it => it.Key).First();

        public static readonly
            Func<Func<IEnumerable<double>, string>, IEnumerable<FeaturesWithLabel>, IEnumerable<(string, string)>>
            PredictAll = (predictingFunc, validationData) => validationData.Map(it =>
            {
                var expected = it.Label;
                var predicted = predictingFunc(it.Features);
                return (expected, predicted);
            });

        /// <summary>
        /// Aggregates results of prediction
        /// </summary>
        /// <param name="possibleLabels">all possible labels from training dataset</param>
        /// <param name="expectedAndPredicted">tuple of labels - expected and predicted value</param>
        /// <returns>outer dictionary key (predicted value) maps to count of actual (true) values</returns>
        public static ImmutableDictionary<string, ImmutableDictionary<string, int>> PredictionStatistics(
            List<string> possibleLabels,
            IEnumerable<(string, string)> expectedAndPredicted)
        {
            var seed = possibleLabels
                .ToImmutableDictionary(k => k, i => ImmutableDictionary.Create<string, int>().AddRange(
                    possibleLabels.Map(it => KeyValuePair.Create(it, 0))));

            return expectedAndPredicted
                .Aggregate(seed, (acc, tuple)
                    =>
                {
                    var (expected, predicted) = tuple;
                    var current = acc[expected][predicted];
                    return acc.SetItem(expected, acc[expected].SetItem(predicted, current + 1));
                });
        }

        public static void PrintStatistics(
            ImmutableDictionary<string, ImmutableDictionary<string, int>> predictionStatistics)
        {
            predictionStatistics.ForEach(it =>
            {
                var currentKey = it.Key;

                var totalPredictedAsCurrent = it.Value.Map(i => i.Value).Sum();
                var correctPredictionsOfCurrent = it.Value[currentKey];
                var totalActualValuesOfCurrent = predictionStatistics.ToList()
                    .Map(it => it.Value)
                    .Map(it => it[currentKey])
                    .Sum();
                var precision = (double) correctPredictionsOfCurrent / (double) totalActualValuesOfCurrent;
                var recall = (double) correctPredictionsOfCurrent / (double) totalPredictedAsCurrent;

                Console.WriteLine("Statistics for " + currentKey);
                Console.WriteLine($"items predicted as {currentKey}: {currentKey}");
                Console.WriteLine("actual values:");
                it.Value.ForEach(it => Console.WriteLine($"{it.Key} : {it.Value}"));
                Console.WriteLine($"total {currentKey}: {totalActualValuesOfCurrent} in validation dataset");
                Console.WriteLine($"correct predictions of {currentKey}: {correctPredictionsOfCurrent}");
                Console.WriteLine($"Precision of {currentKey} prediction: {precision}");
                Console.WriteLine($"Recall of {currentKey} prediction: {recall}");
                Console.WriteLine("\n");
            });
        }

        public static IEnumerable<string> CollectLabels(IEnumerable<FeaturesWithLabel> data) =>
            data.GroupBy(it => it.Label).Select(it => it.Key);

        public static IEnumerable<FeaturesWithLabel> CsvToFeaturesWithLabel(string csv) =>
            csv.Trim().TrimEnd().Split("\n").Map(ToFeaturesWithLabel);

        private static FeaturesWithLabel ToFeaturesWithLabel(string csvRow) =>
            csvRow.Split(',')
                .Pipe(it => (it.SkipLast(1), it.TakeLast(1).First()))
                .Pipe(it => (it.Item1.Map(it2 => double.Parse(it2, CultureInfo.InvariantCulture)), it.Item2))
                .Pipe(it => new FeaturesWithLabel(it.Item1.ToList(), it.Item2));
    }

    public readonly struct FeaturesWithLabel
    {
        public List<double> Features { get; }
        public string Label { get; }

        public FeaturesWithLabel(List<double> features, string label)
        {
            Features = features;
            Label = label;
        }
    }
}