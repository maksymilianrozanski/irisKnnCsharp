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
                foreach (var result in expectedRed.Select(predicted => Knn.Predict(_trainingData, predicted, k)))
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
                foreach (var result in expectedGreen.Select(predicted => Knn.Predict(_trainingData, predicted, k)))
                {
                    Assert.AreEqual(G, result);
                }
            }
        }
    }
}