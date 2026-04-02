//
// Created by Andreea Zaharia on 01/02/2022.
//

#ifndef FLOWER_CPP_LINE_FIT_MODEL_H
#define FLOWER_CPP_LINE_FIT_MODEL_H

#include <vector>

#include "linear_algebra_util.h"
#include "synthetic_dataset.h"
#include <cstddef>
class LineFitModel {
public:
  LineFitModel(int num_iterations, double learning_rate, int num_params);

  std::vector<double> predict(std::vector<std::vector<double>> X);

  std::tuple<size_t, double, double> train_SGD(SyntheticDataset &dataset);

  std::tuple<size_t, double, double> evaluate(SyntheticDataset &test_dataset);

  std::vector<double> get_pred_weights();

  void set_pred_weights(std::vector<double> new_pred_weights);

  double get_bias();

  void set_bias(double new_bias);

  size_t get_model_size();

private:
  int num_iterations;
  int batch_size;
  double learning_rate;

  std::vector<double> pred_weights;
  double pred_b;

  double compute_mse(std::vector<double> true_y, std::vector<double> pred);
};

#endif // FLOWER_CPP_LINE_FIT_MODEL_H
