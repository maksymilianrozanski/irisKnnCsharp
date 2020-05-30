using System.Collections.Generic;
using KnnIris;
using NUnit.Framework;

namespace TestProject1
{
    public class Tests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Test1()
        {
            Assert.Pass();
        }

        [Test]
        public void ShouldReturnExpectedDistance1()
        {
            var first = new List<double> {0, 1};
            var second = new List<double> {0, 10};

            var expected = 9.0;
            var result = Knn.EuclideanDist(first, second);
            Assert.AreEqual(expected, result, 0.0001);
        }

        [Test]
        public void ShouldReturnExpectedDistance2()
        {
            var first = new List<double> {0, 0};
            var second = new List<double> {10, 10};
            var third = new List<double> {10, -10};
            var fourth = new List<double> {-10, 10};
            var fifth = new List<double> {-10, -10};

            var expected = 14.142;
            Assert.AreEqual(expected, Knn.EuclideanDist(first, second), 0.1);
            Assert.AreEqual(expected, Knn.EuclideanDist(first, third), 0.1);
            Assert.AreEqual(expected, Knn.EuclideanDist(first, fourth), 0.1);
            Assert.AreEqual(expected, Knn.EuclideanDist(first, fifth), 0.1);
        }

        [Test]
        public void ShouldReturnZeroForTheSamePointTwice()
        {
            var point = new List<double> {3, 4, 5};
            Assert.AreEqual(0, Knn.EuclideanDist(point, point), 0.0001);
        }
    }
}