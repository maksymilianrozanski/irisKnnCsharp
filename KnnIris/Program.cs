using System;
using System.Collections.Generic;
using System.Linq;

namespace KnnIris
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var r = "red";
            var g = "green";
            var y = "yellow";
            var trainingData = new List<FeaturesWithLabel>
            {
                new FeaturesWithLabel(new List<double> {3, 7}, r),
                new FeaturesWithLabel(new List<double> {4, 7}, r),
                new FeaturesWithLabel(new List<double> {5, 7}, r),
                new FeaturesWithLabel(new List<double> {2, 6}, r),
                new FeaturesWithLabel(new List<double> {3, 6}, g),
                new FeaturesWithLabel(new List<double> {4, 6}, g),
                new FeaturesWithLabel(new List<double> {5, 6}, g),
                new FeaturesWithLabel(new List<double> {6, 6}, r),
                new FeaturesWithLabel(new List<double> {2, 5}, r),
                new FeaturesWithLabel(new List<double> {3, 5}, g),
                new FeaturesWithLabel(new List<double> {4, 5}, g),
                new FeaturesWithLabel(new List<double> {5, 5}, g),
                new FeaturesWithLabel(new List<double> {6, 5}, r),
                new FeaturesWithLabel(new List<double> {4, 2}, r),
                new FeaturesWithLabel(new List<double> {3, 4}, g),
                new FeaturesWithLabel(new List<double> {4, 4}, g),
                new FeaturesWithLabel(new List<double> {5, 4}, g),
                new FeaturesWithLabel(new List<double> {6, 4}, r),
                new FeaturesWithLabel(new List<double> {3, 3}, r),
                new FeaturesWithLabel(new List<double> {4, 3}, r),
                new FeaturesWithLabel(new List<double> {5, 3}, r),
                new FeaturesWithLabel(new List<double> {8, 3}, g),
                new FeaturesWithLabel(new List<double> {2, 2}, y),
                new FeaturesWithLabel(new List<double> {3, 2}, y),
                new FeaturesWithLabel(new List<double> {4, 2}, y),
                new FeaturesWithLabel(new List<double> {7, 2}, g),
                new FeaturesWithLabel(new List<double> {8, 2}, g),
                new FeaturesWithLabel(new List<double> {2, 1}, y),
                new FeaturesWithLabel(new List<double> {3, 1}, y),
                new FeaturesWithLabel(new List<double> {4, 1}, y),
                new FeaturesWithLabel(new List<double> {6, 1}, g),
                new FeaturesWithLabel(new List<double> {7, 1}, g),
                new FeaturesWithLabel(new List<double> {8, 1}, g),
            };

            var result = Knn.Predict(trainingData, new List<double> {1.5,5}, 4);
            Console.WriteLine(result);
        }
    }

    public class Knn
    {
        public static double EuclideanDist(List<double> first, List<double> second) =>
            Math.Sqrt(first.Zip(second)
                .Select(tuple => Math.Pow(tuple.First - tuple.Second, 2))
                .Aggregate((d, d1) => d + d1));

        public static string Predict(IEnumerable<FeaturesWithLabel> data, List<double> predictorValues, int kValue) =>
            data.Select(it => (it, EuclideanDist(it.Features, predictorValues)))
                .OrderBy(it => it.Item2)
                .Take(kValue)
                .GroupBy(it => it.it.Label)
                .OrderByDescending(it => it.Count())
                .Select(it => it.Key).First();
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