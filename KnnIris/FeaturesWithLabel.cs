using System.Collections.Generic;

namespace KnnIris
{
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