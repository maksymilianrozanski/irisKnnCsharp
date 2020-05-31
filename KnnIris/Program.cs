using System;
using System.Collections.Generic;
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
                }, labels => labels);

            var validationData = validationDataCsv
                .Map(Knn.CsvToFeaturesWithLabel).Match(error =>
                {
                    Console.WriteLine(error.Message);
                    Environment.Exit(1);
                    return new List<FeaturesWithLabel>();
                }, labels => labels).ToList();

            var knn3 = Knn.Predict.Apply(trainingData).Apply(3);

            Knn.PredictAll(knn3, validationData)
                .ForEach(it =>
                {
                    var (item1, item2) = it;
                    if(item1 != item2) Console.Write("DIFFERENT !!! ");
                    Console.WriteLine($"(expected - predicted) : ({item1} - {item2})");
                });

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