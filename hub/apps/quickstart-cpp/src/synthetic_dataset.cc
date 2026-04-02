//
// Created by Andreea Zaharia on 31/01/2022.
//

#include "synthetic_dataset.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

SyntheticDataset::SyntheticDataset(std::vector<double> ms, double b,
                                   size_t size) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<> distr(-10.0, 10.0);
  std::cout << "True parameters: " << std::endl;
  for (int i = 0; i < ms.size(); i++) {
    std::cout << std::fixed << "  m" << i << " = " << ms[i] << std::endl;
  }

  std::cout << "  b = " << std::fixed << b << std::endl;

  std::vector<std::vector<double>> xs(size, std::vector<double>(ms.size()));
  std::vector<double> ys(size, 0);
  for (int m_ind = 0; m_ind < ms.size(); m_ind++) {
    std::uniform_real_distribution<double> distx(-10.0, 10.0);

    for (int i = 0; i < size; i++) {
      xs[i][m_ind] = distx(mt);
    }
  }

  for (int i = 0; i < size; i++) {
    ys[i] = b;
    for (int m_ind = 0; m_ind < ms.size(); m_ind++) {
      ys[i] += ms[m_ind] * xs[i][m_ind];
    }
  }

  std::vector<std::vector<double>> data_points;
  for (int i = 0; i < size; i++) {
    std::vector<double> data_point;
    data_point.insert(data_point.end(), xs[i].begin(), xs[i].end());
    data_point.push_back(ys[i]);

    data_points.push_back(data_point);
  }

  this->data_points = data_points;
}

size_t SyntheticDataset::size() { return this->data_points.size(); }

int SyntheticDataset::get_features_count() {
  return this->data_points[0].size() - 1;
}

std::vector<std::vector<double>> SyntheticDataset::get_data_points() {
  return this->data_points;
}
