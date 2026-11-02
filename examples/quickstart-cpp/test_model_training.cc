// Copyright 2026 Flower Labs GmbH. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "line_fit_model.h"
#include "synthetic_dataset.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  // A dataset smaller than the batch must use every row exactly once.
  // The old index vector prepended 32 extra zeros before appending 0..31.
  // Its 64-element batch therefore included row zero 33 times instead of once.
  std::vector<std::vector<double>> rows(32, {0.0, 1.0});
  rows[0].back() = 0.0;
  SyntheticDataset small_dataset(rows);
  LineFitModel one_step(1, 0.01, 1);
  one_step.set_pred_weights({0.0});
  one_step.set_bias(0.0);
  one_step.train_SGD(small_dataset);
  const double expected_bias = 2.0 * 0.01 * 31.0 / 32.0;
  if (std::abs(one_step.get_bias() - expected_bias) > 1e-12) {
    std::cerr << "Incorrect sample weighting: expected bias " << expected_bias
              << ", got " << one_step.get_bias() << '\n';
    return 1;
  }

  // A high-leverage first row must not dominate every minibatch.
  rows.clear();
  for (int i = 0; i < 64; ++i) {
    const double x0 = i == 0 ? 20.0 : (i % 2 == 0 ? 1.0 : -1.0);
    const double x1 = i == 0 ? 20.0 : (i % 4 < 2 ? 1.0 : -1.0);
    rows.push_back({x0, x1, 3.5 * x0 + 9.3 * x1 + 1.7});
  }
  SyntheticDataset training_dataset(rows);
  LineFitModel model(1500, 0.01, 2);
  model.set_pred_weights({0.0, 0.0});
  model.set_bias(0.0);
  model.train_SGD(training_dataset);
  const double loss = std::get<1>(model.evaluate(training_dataset));
  if (!std::isfinite(loss) || loss > 1e-10) {
    std::cerr << "Training failed to converge: loss=" << loss << '\n';
    return 1;
  }
  SyntheticDataset empty_dataset(std::vector<std::vector<double>>{});
  try {
    model.train_SGD(empty_dataset);
    std::cerr << "Empty training data should be rejected\n";
    return 1;
  } catch (const std::invalid_argument &) {
  }
  std::cout << "PASS: unbiased sampling, small batches, finite convergence, "
               "empty-data rejection\n";
  return 0;
}
