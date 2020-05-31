using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using LaYumba.Functional;
using Double = System.Double;

namespace KnnIris
{
    public static class CsvParser
    {
        public static IEnumerable<FeaturesWithLabel> CsvToFeaturesWithLabel(string csv) =>
            csv.Trim().TrimEnd().Split("\n").Map<string, FeaturesWithLabel>(ToFeaturesWithLabel);

        private static FeaturesWithLabel ToFeaturesWithLabel(string csvRow) =>
            csvRow.Split(',')
                .Pipe(it => (it.SkipLast(1), it.TakeLast(1).First()))
                .Pipe(it => (it.Item1.Map(it2 => double.Parse(it2, CultureInfo.InvariantCulture)), it.Item2))
                .Pipe(it => new FeaturesWithLabel(it.Item1.ToList(), it.Item2));
    }
}