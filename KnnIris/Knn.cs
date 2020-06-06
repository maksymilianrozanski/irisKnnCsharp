using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Globalization;
using System.Linq;
using System.Text;
using LaYumba.Functional;

namespace KnnIris
{
    public class Knn
    {
        public static Func<IEnumerable<double>, IEnumerable<double>, double> EuclideanDist => (first, second) =>
            first.Zip(second)
                .Select(tuple => Math.Pow(tuple.First - tuple.Second, 2))
                .Sum().Pipe(Math.Sqrt);

        public static Func<IEnumerable<double>, IEnumerable<double>, double> ManhattanDist => (first, second) =>
            first.Zip(second)
                .Select(tuple => Math.Abs(tuple.First - tuple.Second))
                .Sum();

        public static readonly Func<
            Func<IEnumerable<double>, IEnumerable<double>, double>,
            IEnumerable<FeaturesWithLabel>, int, IEnumerable<double>,
            string> Predict =
            (distanceFunc, trainingData, k, predictorValues) =>
                trainingData.Select(it => (it, distanceFunc(it.Features, predictorValues)))
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

        public static IEnumerable<string> CollectLabels(IEnumerable<FeaturesWithLabel> data) =>
            data.GroupBy(it => it.Label).Select(it => it.Key);
    }
}