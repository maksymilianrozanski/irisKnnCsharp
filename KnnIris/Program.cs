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
            Console.WriteLine("Hello World!");

            const string trainingDataUrl =
                "https://gist.githubusercontent.com/maksymilianrozanski/fa386a964426d8b0b8fac75a026a9dbd/raw/b5aa5f57df07a4e8ecec6ed9a7eeba5ac8fe9aa3/irisTrainingData";
            const string validationDataUrl =
                "https://gist.githubusercontent.com/maksymilianrozanski/3b631b25cccc2baa2731956aaecc7716/raw/1d62a26bdeaf91fe4dcaf56d1fa96de13ab75e5a/irisValidationData";

            var downloader = new CsvDownloader();

            var trainingDataCsv = downloader.DownloadFromUrl(new Uri(trainingDataUrl));
            var validationDataCsv = downloader.DownloadFromUrl(new Uri(validationDataUrl));

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

            var knn3 = Knn.Predict.Apply(trainingData).Apply(3);

            var predictionResults = Knn.PredictAll(knn3, validationData);

            var possibleLabels = Knn.CollectLabels(trainingData).ToList();
            Statistics.PredictionStatistics(possibleLabels, predictionResults)
                .Pipe(Statistics.ExplainStatistics)
                .Pipe(Console.WriteLine);

            Console.WriteLine("End");
        }
    }
}