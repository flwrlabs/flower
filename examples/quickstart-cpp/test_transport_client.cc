// Test-only client: echo the incoming tensors through the real Fleet transport.
#include "supernode_client.h"

#include <exception>
#include <iostream>
#include <stdexcept>

namespace {
class EchoClient : public flwr_local::Client {
public:
  flwr_local::ParametersRes get_parameters() override {
    throw std::runtime_error("Unexpected get_parameters in transport test");
  }

  flwr_local::PropertiesRes get_properties(flwr_local::PropertiesIns) override {
    throw std::runtime_error("Unexpected get_properties in transport test");
  }

  flwr_local::FitRes fit(flwr_local::FitIns ins) override {
    return flwr_local::FitRes(ins.getParameters(), 1, 1, 0.0f, {});
  }

  flwr_local::EvaluateRes evaluate(flwr_local::EvaluateIns) override {
    throw std::runtime_error("Unexpected evaluate in transport test");
  }
};
} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    return 1;
  }
  EchoClient client;
  try {
    flwr_quickstart::start_client(argv[1], &client);
  } catch (const std::exception &e) {
    std::cerr << "[flwr-cpp] fatal: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
