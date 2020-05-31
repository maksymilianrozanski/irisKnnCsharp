using System;
using System.Collections.Generic;
using System.Linq;
using KnnIris;
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
                foreach (var result in expectedRed.Select(predicted => Knn.Predict(_trainingData, k, predicted)))
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
                foreach (var result in expectedGreen.Select(predicted => Knn.Predict(_trainingData, k, predicted)))
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
            var C = "cat";
            var F = "fish";
            var H = "hen";

            var possibleLabels = new List<string> {C, F, H};

            var actualValues = new List<string>
                {C, C, C, C, C, C, F, F, F, F, F, F, F, F, F, F, H, H, H, H, H, H, H, H, H};
            var predictedValues = new List<string>
                {C, C, C, C, H, F, C, C, C, C, C, C, H, H, F, F, C, C, C, H, H, H, H, H, H};

            var predictionResults = actualValues.Zip(predictedValues);

            var result = Knn.PredictionStatistics(possibleLabels, predictionResults);
            Assert.AreEqual(4, result[C][C], "4 of items classified as cat are cats");
            Assert.AreEqual(1, result[C][F], "1 item classified as cat is fish");
            Assert.AreEqual(1, result[C][H], "1 item classified as cat is a hen");
            Assert.AreEqual(6, result[F][C], "6 of items classified as fish are cats");
            Assert.AreEqual(2, result[F][F], "2 of items classified as fish are fish");
            Assert.AreEqual(2, result[F][H], "2 of items classified as fish are hens");
            Assert.AreEqual(3, result[H][C], "3 of items classified as hen are cats");
            Assert.AreEqual(0, result[H][F], "0 of items classified as hen are fish");
            Assert.AreEqual(6, result[H][H], "6 of items classified as hen are hens");
        }
    }
}