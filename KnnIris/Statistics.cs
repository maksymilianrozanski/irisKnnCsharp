using System.Collections.Generic;
using System.Collections.Immutable;
using System.Globalization;
using System.Linq;
using System.Text;
using LaYumba.Functional;

namespace KnnIris
{
    public static class Statistics
    {
        /// <summary>
        /// Aggregates results of prediction
        /// </summary>
        /// <param name="possibleLabels">all possible labels from training dataset</param>
        /// <param name="expectedAndPredicted">tuple of labels - expected and predicted value</param>
        /// <returns>outer dictionary key (predicted value) maps to count of actual (true) values</returns>
        public static ImmutableSortedDictionary<string, ImmutableSortedDictionary<string, int>> PredictionStatistics(
            List<string> possibleLabels,
            IEnumerable<(string, string)> expectedAndPredicted)
        {
            var seed = possibleLabels
                .ToImmutableSortedDictionary(k => k, i => ImmutableSortedDictionary.Create<string, int>().AddRange(
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

        public static string ExplainStatistics(
            ImmutableSortedDictionary<string, ImmutableSortedDictionary<string, int>> predictionStatistics)
        {
            return predictionStatistics.Map(it =>
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

                return new StringBuilder().AppendLine($"Statistics for {currentKey}")
                    .AppendLine($"items predicted as {currentKey}: {totalPredictedAsCurrent}")
                    .AppendLine("actual values:")
                    .AppendLine(it.Value.Map(it => $"{it.Key} : {it.Value},").Aggregate((s, s1) => s + " " + s1))
                    .AppendLine($"total {currentKey}: {totalActualValuesOfCurrent} in validation dataset")
                    .AppendLine($"correct predictions of {currentKey}: {correctPredictionsOfCurrent}")
                    .AppendLine($"Precision of {currentKey} prediction: {precision.ToString("N5", CultureInfo.InvariantCulture)}")
                    .AppendLine($"Recall of {currentKey} prediction: {recall.ToString("N5", CultureInfo.InvariantCulture)}")
                    .ToString();
            }).Aggregate((s, s1) => $"{s}\n{s1}");
        }
    }
}