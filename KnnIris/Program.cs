using System;
using System.Collections.Generic;
using System.Linq;
using LaYumba.Functional;

namespace KnnIris
{
    internal static class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Enter url to training data csv, or press Enter twice to use Iris dataset");
            Console.WriteLine("csv format using ',' as value separator and '.' as decimal point separator is expected");
            Console.WriteLine("csv must not contain table header");
            Console.WriteLine("data label in the last column is expected");

            const string irisTrainingDataUrl =
                "https://gist.githubusercontent.com/maksymilianrozanski/fa386a964426d8b0b8fac75a026a9dbd/raw/b5aa5f57df07a4e8ecec6ed9a7eeba5ac8fe9aa3/irisTrainingData";
            const string irisValidationDataUrl =
                "https://gist.githubusercontent.com/maksymilianrozanski/3b631b25cccc2baa2731956aaecc7716/raw/1d62a26bdeaf91fe4dcaf56d1fa96de13ab75e5a/irisValidationData";

            var trainingDataUrl = Console.ReadLine();
            
            Console.WriteLine("Enter url to validation data (or leave blank to use default dataset)");
            var validationDataUrl = Console.ReadLine();

            Console.WriteLine(
                "Enter 'm' to use Manhattan distance, or press [Enter] to use Euclidean Distance (default)");
            var distanceFunc = Console.ReadLine() == "m" ? Knn.ManhattanDist : Knn.EuclideanDist;

            (Either<NetworkRequestError, string>, Either<NetworkRequestError, string>) DownloadedData(string url1,
                string url2)
            {
                var downloader = new CsvDownloader();
                return url1 == "" || url2 == ""
                    ? (downloader.DownloadFromUrl(new Uri(irisTrainingDataUrl)),
                        downloader.DownloadFromUrl(new Uri(irisValidationDataUrl)))
                    : (downloader.DownloadFromUrl(new Uri(url1)), downloader.DownloadFromUrl(new Uri(url2)));
            }

            var (trainingDataCsv, validationDataCsv) =
                DownloadedData(trainingDataUrl, validationDataUrl);

            var trainingData = trainingDataCsv
                .Map(CsvParser.CsvToFeaturesWithLabel).Match(error =>
                {
                    Console.WriteLine(error.Message);
                    Environment.Exit(1);
                    return new List<FeaturesWithLabel>();
                }, labels => labels).ToList();

            var validationData = validationDataCsv
                .Map(CsvParser.CsvToFeaturesWithLabel).Match(error =>
                {
                    Console.WriteLine(error.Message);
                    Environment.Exit(1);
                    return new List<FeaturesWithLabel>();
                }, labels => labels).ToList();

            var knn = Knn.Predict.Apply(distanceFunc).Apply(trainingData);

            Console.WriteLine("Please wait...");
            
            for (var k = 1; k <= 5; k++)
            {
                Console.WriteLine($"Values for {k} nearest neighbours");
                var predictionResults = Knn.PredictAll(knn.Apply(k), validationData);
                var possibleLabels = Knn.CollectLabels(trainingData).ToList();
                Statistics.PredictionStatistics(possibleLabels, predictionResults)
                    .Pipe(Statistics.ExplainStatistics)
                    .Pipe(Console.WriteLine);
            }

            Console.WriteLine("End");
        }
    }
}