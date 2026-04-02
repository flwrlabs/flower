#include "simple_client.h"
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

  std::ostringstream oss1, oss2; // Possibly unnecessary
  oss1.write(reinterpret_cast<const char *>(pred_weights.data()),
             pred_weights.size() * sizeof(double));
  tensors.push_back(oss1.str());

  oss2.write(reinterpret_cast<const char *>(&pred_b), sizeof(double));
  tensors.push_back(oss2.str());

  std::string tensor_str = "cpp_double";
  return flwr_local::ParametersRes(flwr_local::Parameters(tensors, tensor_str));
};

void SimpleFlwrClient::set_parameters(flwr_local::Parameters params) {

  std::list<std::string> s = params.getTensors();
  std::cout << "Received " << s.size() << " Layers from server:" << std::endl;

  if (s.empty() == 0) {
    // Layer 1
    auto layer = s.begin();
    size_t num_bytes = (*layer).size();
    const char *weights_char = (*layer).c_str();
    const double *weights_double =
        reinterpret_cast<const double *>(weights_char);
    std::vector<double> weights(weights_double,
                                weights_double + num_bytes / sizeof(double));
    this->model.set_pred_weights(weights);
    for (auto x : this->model.get_pred_weights())
      for (size_t j = 0; j < this->model.get_pred_weights().size(); j++)
        std::cout << "  m" << j << "_server = " << std::fixed
                  << this->model.get_pred_weights()[j] << std::endl;

    // Layer 2 = Bias
    auto layer_2 = std::next(layer, 1);
    num_bytes = (*layer_2).size();
    const char *bias_char = (*layer_2).c_str();
    const double *bias_double = reinterpret_cast<const double *>(bias_char);
    this->model.set_bias(bias_double[0]);
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
