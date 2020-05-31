using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using LaYumba.Functional;

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
                .Map(Knn.CsvToFeaturesWithLabel);

            var validationData = validationDataCsv
                .Map(Knn.CsvToFeaturesWithLabel);

            Console.WriteLine("End");
        }
    }

    public class Knn
    {
        public static double EuclideanDist(List<double> first, List<double> second) =>
            Math.Sqrt(first.Zip(second)
                .Select(tuple => Math.Pow(tuple.First - tuple.Second, 2))
                .Aggregate((d, d1) => d + d1));

        public static string Predict(IEnumerable<FeaturesWithLabel> data, int kValue, List<double> predictorValues) =>
            data.Select(it => (it, EuclideanDist(it.Features, predictorValues)))
                .OrderBy(it => it.Item2)
                .Take(kValue)
                .GroupBy(it => it.it.Label)
                .OrderByDescending(it => it.Count())
                .Select(it => it.Key).First();

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