//
// Created by Andreea Zaharia on 31/01/2022.
//

#ifndef FLOWER_CPP_SYNTHETIC_DATASET_H
#define FLOWER_CPP_SYNTHETIC_DATASET_H

#include <cstddef>
#include <vector>
class SyntheticDataset {
public:
  // Generates the synthetic dataset of size size around given vector m of size
  // ms_size and given bias b.
  SyntheticDataset(std::vector<double> ms, double b, size_t size);

  // Returns the size of the dataset.
  size_t size();

  // Returns the dataset.
  std::vector<std::vector<double>> get_data_points();

  int get_features_count();

private:
  std::vector<double> ms;
  double b;

  // The label is the last position in the vector.
  // TODO: consider changing this to a pair with the label.

  std::vector<std::vector<double>> data_points;
};

#endif // FLOWER_CPP_SYNTHETIC_DATASET_H
