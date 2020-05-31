using System.Collections.Generic;
using System.Linq;
using KnnIris;
using LaYumba.Functional;
using NUnit.Framework;

namespace TestProject1
{
    public class CsvParsingTests
    {
        [Test]
        public void ShouldReturnFeaturesWithLabelFromCsv()
        {
            const string input = "6.1,2.6,5.6,1.4,virginica\n5.0,3.6,1.4,0.2,setosa\n5.5,2.3,4.0,1.3,versicolor";

            var expected = new List<FeaturesWithLabel>
            {
                new FeaturesWithLabel(new List<double> {6.1, 2.6, 5.6, 1.4}, "virginica"),
                new FeaturesWithLabel(new List<double> {5.0, 3.6, 1.4, 0.2}, "setosa"),
                new FeaturesWithLabel(new List<double> {5.5, 2.3, 4.0, 1.3}, "versicolor")
            };

            var result = Knn.CsvToFeaturesWithLabel(input).ToList();

            Assert.AreEqual(expected.Count, result.Count);

            expected.Zip(result).ForEach(AssertFeaturesWithLabelEqual);
        }

        private static void AssertFeaturesWithLabelEqual((FeaturesWithLabel, FeaturesWithLabel) twoItems)
        {
            var (first, second) = twoItems;
            Assert.AreEqual(first.Features.Count, second.Features.Count, "Both should have equal number of features");
            first.Features.Zip(second.Features)
                .ForEach(it => Assert.AreEqual(it.First, it.Second));
            first.Label.Zip(second.Label)
                .ForEach(it => Assert.AreEqual(it.First, it.Second));
        }
    }
}