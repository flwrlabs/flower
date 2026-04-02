//
// Created by Andreea Zaharia on 01/02/2022.
//

#include "line_fit_model.h"
#include "synthetic_dataset.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

LineFitModel::LineFitModel(int num_iterations, double learning_rate,
                           int num_params)
    : num_iterations(num_iterations), learning_rate(learning_rate) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<> distr(-10.0, 10.0);
  for (int i = 0; i < num_params; i++) {
    this->pred_weights.push_back(distr(mt));
  }

  this->pred_b = 0.0;
  this->batch_size = 64;
}
std::vector<double> LineFitModel::get_pred_weights() {
  std::vector<double> copy_of_weights(this->pred_weights);
  return copy_of_weights;
}

void LineFitModel::set_pred_weights(std::vector<double> new_weights) {
  this->pred_weights.assign(new_weights.begin(), new_weights.end());
}

double LineFitModel::get_bias() { return this->pred_b; }
void LineFitModel::set_bias(double new_bias) { this->pred_b = new_bias; }

size_t LineFitModel::get_model_size() { return this->pred_weights.size(); };

std::vector<double> LineFitModel::predict(std::vector<std::vector<double>> X) {
  std::vector<double> prediction(X.size(), 0.0);
  for (int i = 0; i < X.size(); i++) {
    for (int j = 0; j < X[i].size(); j++) {
      prediction[i] += this->pred_weights[j] * X[i][j];
    }
    prediction[i] += this->pred_b;
  }

  return prediction;
}

std::tuple<size_t, double, double>
LineFitModel::train_SGD(SyntheticDataset &dataset) {
  int features = dataset.get_features_count();
  std::vector<std::vector<double>> data_points = dataset.get_data_points();

  std::vector<double> data_indices(dataset.size());
  for (int i = 0; i < dataset.size(); i++) {
    data_indices.push_back(i);
  }

  std::vector<double> dW(features);
  std::vector<double> err(batch_size, 10000);
  std::vector<double> pW(features);
  double training_error = 0.0;
  for (int iteration = 0; iteration < num_iterations; iteration++) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data_indices.begin(), data_indices.end(), g);

    std::vector<std::vector<double>> X(this->batch_size,
                                       std::vector<double>(features));
    std::vector<double> y(this->batch_size);

    for (int i = 0; i < this->batch_size; i++) {
      std::vector<double> point = data_points[data_indices[i]];
      y[i] = point.back();
      point.pop_back();
      X[i] = point;
    }

    pW = this->pred_weights;
    double pB = this->pred_b;
    double dB;

    std::vector<double> pred = predict(X);

    err = LinearAlgebraUtil::subtract_vector(y, pred);

    dW = LinearAlgebraUtil::multiply_matrix_vector(
        LinearAlgebraUtil::transpose_vector(X), err);
    dW = LinearAlgebraUtil::multiply_vector_scalar(dW,
                                                   (-2.0 / this->batch_size));

    dB = (-2.0 / this->batch_size) *
         std::accumulate(err.begin(), err.end(), 0.0);

    this->pred_weights = LinearAlgebraUtil::subtract_vector(
        pW, LinearAlgebraUtil::multiply_vector_scalar(dW, learning_rate));
    this->pred_b = pB - learning_rate * dB;

    if (iteration % 250 == 0) {
      training_error = this->compute_mse(y, predict(X));
      std::cout << "Iteration: " << iteration
                << "  Training error: " << training_error << '\n';
    }
  }
  std::cout << "Local model:" << std::endl;
  for (size_t i = 0; i < pred_weights.size(); i++) {
    std::cout << "  m" << i << "_local = " << std::fixed << pred_weights[i]
              << std::endl;
  }
  std::cout << "  b_local = " << std::fixed << pred_b << std::endl << std::endl;

  double accuracy = training_error;
  return std::make_tuple(dataset.size(), training_error, accuracy);
}

double LineFitModel::compute_mse(std::vector<double> true_y,
                                 std::vector<double> pred) {
  double error = 0.0;

  for (int i = 0; i < true_y.size(); i++) {
    error += (pred[i] - true_y[i]) * (pred[i] - true_y[i]);
  }

  return error / (1.0 * true_y.size());
}

std::tuple<size_t, double, double>
LineFitModel::evaluate(SyntheticDataset &test_dataset) {
  std::vector<std::vector<double>> data_points = test_dataset.get_data_points();
  int num_features = data_points[0].size();
  std::vector<std::vector<double>> X(test_dataset.size(),
                                     std::vector<double>(num_features));
  std::vector<double> y(test_dataset.size());

  for (int i = 0; i < test_dataset.size(); i++) {
    std::vector<double> point = data_points[i];
    y[i] = point.back();
    point.pop_back();
    X[i] = point;
  }

  double test_loss = compute_mse(y, predict(X));
  return std::make_tuple(test_dataset.size(), test_loss, test_loss);
}
