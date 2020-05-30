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
        }
    }

    public class Knn
    {
        public static double EuclideanDist(List<double> first, List<double> second) =>
            Math.Sqrt(first.Zip(second)
                .Select(tuple => Math.Pow(tuple.First - tuple.Second, 2))
                .Aggregate((d, d1) => d + d1));
    }
}