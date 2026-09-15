#include "simple_client.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {

bool is_little_endian() {
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t *>(&value) == 1;
}

std::string double_to_little_endian_bytes(double value) {
  std::string bytes(sizeof(double), '\0');
  std::memcpy(bytes.data(), &value, sizeof(double));
  if (!is_little_endian()) {
    std::reverse(bytes.begin(), bytes.end());
  }
  return bytes;
}

double double_from_little_endian_bytes(const char *data) {
  std::string bytes(data, sizeof(double));
  if (!is_little_endian()) {
    std::reverse(bytes.begin(), bytes.end());
  }
  double value = 0.0;
  std::memcpy(&value, bytes.data(), sizeof(double));
  return value;
}

std::string vector_to_little_endian_bytes(const std::vector<double> &values) {
  std::string bytes;
  bytes.reserve(values.size() * sizeof(double));
  for (double value : values) {
    bytes += double_to_little_endian_bytes(value);
  }
  return bytes;
}

std::vector<double> vector_from_little_endian_bytes(const std::string &bytes) {
  if (bytes.size() % sizeof(double) != 0) {
    throw std::runtime_error("C++ tensor byte length must be divisible by 8");
  }
  std::vector<double> values;
  values.reserve(bytes.size() / sizeof(double));
  for (size_t offset = 0; offset < bytes.size(); offset += sizeof(double)) {
    values.push_back(double_from_little_endian_bytes(bytes.data() + offset));
  }
  return values;
}

} // namespace
/**
 * Initializer
 */
SimpleFlwrClient::SimpleFlwrClient(std::string client_id, LineFitModel &model,
                                   SyntheticDataset &training_dataset,
                                   SyntheticDataset &validation_dataset,
                                   SyntheticDataset &test_dataset)
    : model(model), training_dataset(training_dataset),
      validation_dataset(validation_dataset), test_dataset(test_dataset){

                                              };

/**
 * Return the current local model parameters
 * Simple string are used for now to test communication, needs updates in the
 * future
 */
flwr_local::ParametersRes SimpleFlwrClient::get_parameters() {
  // Serialize
  std::vector<double> pred_weights = this->model.get_pred_weights();
  double pred_b = this->model.get_bias();
  std::list<std::string> tensors;

  tensors.push_back(vector_to_little_endian_bytes(pred_weights));
  tensors.push_back(double_to_little_endian_bytes(pred_b));

  std::string tensor_str = "cpp_double";
  return flwr_local::ParametersRes(flwr_local::Parameters(tensors, tensor_str));
};

void SimpleFlwrClient::set_parameters(flwr_local::Parameters params) {

  std::list<std::string> s = params.getTensors();
  std::cout << "Received " << s.size() << " Layers from server:" << std::endl;

  if (s.empty() == 0) {
    // Layer 1
    auto layer = s.begin();
    std::vector<double> weights = vector_from_little_endian_bytes(*layer);
    this->model.set_pred_weights(weights);
    for (auto x : this->model.get_pred_weights())
      for (size_t j = 0; j < this->model.get_pred_weights().size(); j++)
        std::cout << "  m" << j << "_server = " << std::fixed
                  << this->model.get_pred_weights()[j] << std::endl;

    // Layer 2 = Bias
    auto layer_2 = std::next(layer, 1);
    if (layer_2->size() != sizeof(double)) {
      throw std::runtime_error("Bias tensor must contain exactly one double");
    }
    this->model.set_bias(double_from_little_endian_bytes(layer_2->data()));
    std::cout << "  b_server = " << std::fixed << this->model.get_bias()
              << std::endl;
  }
};

flwr_local::PropertiesRes
SimpleFlwrClient::get_properties(flwr_local::PropertiesIns ins) {
  flwr_local::PropertiesRes p;
  p.setPropertiesRes(
      static_cast<flwr_local::Properties>(ins.getPropertiesIns()));
  return p;
}

/**
 * Refine the provided weights using the locally held dataset
 * Simple settings are used for testing, needs updates in the future
 */
flwr_local::FitRes SimpleFlwrClient::fit(flwr_local::FitIns ins) {
  std::cout << "Fitting..." << std::endl;
  flwr_local::FitRes resp;

  flwr_local::Parameters p = ins.getParameters();
  this->set_parameters(p);

  std::tuple<size_t, float, double> result =
      this->model.train_SGD(this->training_dataset);

  resp.setParameters(this->get_parameters().getParameters());
  resp.setNum_example(std::get<0>(result));

  return resp;
};

/**
 * Evaluate the provided weights using the locally held dataset
 * Needs updates in the future
 */
flwr_local::EvaluateRes
SimpleFlwrClient::evaluate(flwr_local::EvaluateIns ins) {
  std::cout << "Evaluating..." << std::endl;
  flwr_local::EvaluateRes resp;
  flwr_local::Parameters p = ins.getParameters();
  this->set_parameters(p);

  // Evaluation returns a number_of_examples, a loss and an "accuracy"
  std::tuple<size_t, double, double> result =
      this->model.evaluate(this->test_dataset);

  resp.setNum_example(std::get<0>(result));
  resp.setLoss(std::get<1>(result));

  flwr_local::Scalar loss_metric = flwr_local::Scalar();
  loss_metric.setDouble(std::get<2>(result));
  std::map<std::string, flwr_local::Scalar> metric = {{"loss", loss_metric}};
  resp.setMetrics(metric);

  return resp;
};
