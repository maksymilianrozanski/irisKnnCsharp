using System;
using System.Collections.Generic;
using System.Linq;
using KnnIris;
using LaYumba.Functional;
using NUnit.Framework;

namespace TestProject1
{
    public class KnnTests
    {
        private const string R = "red";
        private const string G = "green";
        private const string Y = "yellow";

        readonly List<FeaturesWithLabel> _trainingData = new List<FeaturesWithLabel>
        {
            new FeaturesWithLabel(new List<double> {3, 7}, R),
            new FeaturesWithLabel(new List<double> {4, 7}, R),
            new FeaturesWithLabel(new List<double> {5, 7}, R),
            new FeaturesWithLabel(new List<double> {2, 6}, R),
            new FeaturesWithLabel(new List<double> {3, 6}, G),
            new FeaturesWithLabel(new List<double> {4, 6}, G),
            new FeaturesWithLabel(new List<double> {5, 6}, G),
            new FeaturesWithLabel(new List<double> {6, 6}, R),
            new FeaturesWithLabel(new List<double> {2, 5}, R),
            new FeaturesWithLabel(new List<double> {3, 5}, G),
            new FeaturesWithLabel(new List<double> {4, 5}, G),
            new FeaturesWithLabel(new List<double> {5, 5}, G),
            new FeaturesWithLabel(new List<double> {6, 5}, R),
            new FeaturesWithLabel(new List<double> {4, 2}, R),
            new FeaturesWithLabel(new List<double> {3, 4}, G),
            new FeaturesWithLabel(new List<double> {4, 4}, G),
            new FeaturesWithLabel(new List<double> {5, 4}, G),
            new FeaturesWithLabel(new List<double> {6, 4}, R),
            new FeaturesWithLabel(new List<double> {3, 3}, R),
            new FeaturesWithLabel(new List<double> {4, 3}, R),
            new FeaturesWithLabel(new List<double> {5, 3}, R),
            new FeaturesWithLabel(new List<double> {8, 3}, G),
            new FeaturesWithLabel(new List<double> {2, 2}, Y),
            new FeaturesWithLabel(new List<double> {3, 2}, Y),
            new FeaturesWithLabel(new List<double> {4, 2}, Y),
            new FeaturesWithLabel(new List<double> {7, 2}, G),
            new FeaturesWithLabel(new List<double> {8, 2}, G),
            new FeaturesWithLabel(new List<double> {2, 1}, Y),
            new FeaturesWithLabel(new List<double> {3, 1}, Y),
            new FeaturesWithLabel(new List<double> {4, 1}, Y),
            new FeaturesWithLabel(new List<double> {6, 1}, G),
            new FeaturesWithLabel(new List<double> {7, 1}, G),
            new FeaturesWithLabel(new List<double> {8, 1}, G),
        };

        [Test]
        public void ShouldPredictRedValues()
        {
            var expectedRed = new List<List<double>>
            {
                new List<double> {2, 7},
                new List<double> {4.1, 6.9},
                new List<double> {6, 7},
                new List<double> {5.9, 3.9}
            };

            for (var k = 1; k <= 3; k++)
            {
                foreach (var result in expectedRed.Select(predicted =>
                    Knn.Predict(Knn.EuclideanDist, _trainingData, k, predicted)))
                {
                    Assert.AreEqual(R, result);
                }
            }
        }

        [Test]
        public void ShouldPredictGreenValues()
        {
            var expectedGreen = new List<List<double>>
            {
                new List<double> {4, 6},
                new List<double> {3.5, 6},
                new List<double> {5.2, 5}
            };

            for (var k = 1; k <= 3; k++)
            {
                foreach (var result in expectedGreen.Select(predicted =>
                    Knn.Predict(Knn.ManhattanDist, _trainingData, k, predicted)))
                {
                    Assert.AreEqual(G, result);
                }
            }
        }

        /// <summary>
        /// used dataset:
        /// https://gist.githubusercontent.com/bshmueli/65ffbef546a2c799f55e1515c03e4d69/raw/2654db5a73e84673649962870dce16b38c322297/mc_metrics_part1.py
        /// </summary>
        [Test]
        public void ShouldCreateStatisticsDictionary()
        {
            var predictionResults =
                AnimalsData(out var c, out var f, out var h, out var possibleLabels);

            var result = Statistics.PredictionStatistics(possibleLabels, predictionResults);
            Assert.AreEqual(4, result[c][c], "4 of items classified as cat are cats");
            Assert.AreEqual(1, result[c][f], "1 item classified as cat is fish");
            Assert.AreEqual(1, result[c][h], "1 item classified as cat is a hen");
            Assert.AreEqual(6, result[f][c], "6 of items classified as fish are cats");
            Assert.AreEqual(2, result[f][f], "2 of items classified as fish are fish");
            Assert.AreEqual(2, result[f][h], "2 of items classified as fish are hens");
            Assert.AreEqual(3, result[h][c], "3 of items classified as hen are cats");
            Assert.AreEqual(0, result[h][f], "0 of items classified as hen are fish");
            Assert.AreEqual(6, result[h][h], "6 of items classified as hen are hens");

            Console.Write(Statistics.ExplainStatistics(result));
        }

        private static IEnumerable<(string First, string Second)> AnimalsData(out string c, out string f, out string h,
            out List<string> possibleLabels)
        {
            c = "cat";
            f = "fish";
            h = "hen";

            possibleLabels = new List<string> {c, f, h};

            var actualValues = new List<string>
                {c, c, c, c, c, c, f, f, f, f, f, f, f, f, f, f, h, h, h, h, h, h, h, h, h};
            var predictedValues = new List<string>
                {c, c, c, c, h, f, c, c, c, c, c, c, h, h, f, f, c, c, c, h, h, h, h, h, h};

            var predictionResults = actualValues.Zip(predictedValues);
            return predictionResults;
        }

        [Test]
        public void ShouldReturnPredictionExplanation()
        {
            var predictionResults =
                AnimalsData(out var c, out var f, out var h, out var possibleLabels);

            var expected =
                @"Statistics for cat
items predicted as cat: 6
actual values:
cat : 4, fish : 1, hen : 1,
total cat: 13 in validation dataset
correct predictions of cat: 4
Precision of cat prediction: 0.30769
Recall of cat prediction: 0.66667

Statistics for fish
items predicted as fish: 10
actual values:
cat : 6, fish : 2, hen : 2,
total fish: 3 in validation dataset
correct predictions of fish: 2
Precision of fish prediction: 0.66667
Recall of fish prediction: 0.20000

Statistics for hen
items predicted as hen: 9
actual values:
cat : 3, fish : 0, hen : 6,
total hen: 9 in validation dataset
correct predictions of hen: 6
Precision of hen prediction: 0.66667
Recall of hen prediction: 0.66667
".Replace("\r", "");

            var result = Statistics.PredictionStatistics(possibleLabels, predictionResults)
                .Pipe(Statistics.ExplainStatistics)
                .Replace("\r", "");
            Assert.AreEqual(expected, result);
        }
    }
}