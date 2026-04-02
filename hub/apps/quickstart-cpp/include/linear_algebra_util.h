//
// Created by Andreea Zaharia on 07/02/2022.
//

#ifndef FLOWER_CPPV2_LINEAR_ALGEBRA_UTIL_H
#define FLOWER_CPPV2_LINEAR_ALGEBRA_UTIL_H

#include <vector>

class LinearAlgebraUtil {
public:
  static std::vector<double> subtract_vector(std::vector<double> v1,
                                             std::vector<double> v2);

  static std::vector<double>
  multiply_matrix_vector(std::vector<std::vector<double>> mat,
                         std::vector<double> v);

  static std::vector<double> add_vector_scalar(std::vector<double> v, double a);

  static std::vector<double> multiply_vector_scalar(std::vector<double> v,
                                                    double a);

  static std::vector<std::vector<double>>
  transpose_vector(std::vector<std::vector<double>> v);
};

#endif // FLOWER_CPPV2_LINEAR_ALGEBRA_UTIL_H
