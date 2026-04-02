//
// Created by Andreea Zaharia on 07/02/2022.
//

#include "linear_algebra_util.h"
#include <iostream>
#include <vector>

std::vector<double> LinearAlgebraUtil::subtract_vector(std::vector<double> v1,
                                                       std::vector<double> v2) {
  std::vector<double> result(v1.size());
  for (int i = 0; i < v1.size(); i++) {
    result[i] = v1[i] - v2[i];
  }
  return result;
}

std::vector<double>
LinearAlgebraUtil::multiply_matrix_vector(std::vector<std::vector<double>> mat,
                                          std::vector<double> v) {
  std::vector<double> result(mat.size(), 0.0);
  for (int i = 0; i < mat.size(); i++) {
    result[i] = 0;
    for (int j = 0; j < mat[0].size(); j++) {
      result[i] += mat[i][j] * v[j];
    }
  }
  return result;
}

std::vector<double> LinearAlgebraUtil::add_vector_scalar(std::vector<double> v,
                                                         double a) {
  for (int i = 0; i < v.size(); i++) {
    v[i] += a;
  }
  return v;
}

std::vector<double>
LinearAlgebraUtil::multiply_vector_scalar(std::vector<double> v, double a) {
  for (int i = 0; i < v.size(); i++) {
    v[i] *= a;
  }

  return v;
}

std::vector<std::vector<double>>
LinearAlgebraUtil::transpose_vector(std::vector<std::vector<double>> v) {
  std::vector<std::vector<double>> vT(v[0].size(),
                                      std::vector<double>(v.size()));
  for (int i = 0; i < v.size(); i++) {
    for (int j = 0; j < v[0].size(); j++) {
      vT[j][i] = v[i][j];
    }
  }

  return vT;
}
