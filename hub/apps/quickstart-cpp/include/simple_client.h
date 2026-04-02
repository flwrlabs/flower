/***********************************************************************************************************
 * 
 * @file simple_client.h
 *
 * @brief Define an example flower client, train and test method
 *
 * @version 1.0
 *
 * @date 06/09/2021
 *
 * ********************************************************************************************************/
#pragma once
#include "client.h"
#include "line_fit_model.h"
#include "synthetic_dataset.h"
#include <ctime>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
/**
 * Validate the network on the entire test set
 *
 */

class SimpleFlwrClient : public flwr_local::Client {
public:
  SimpleFlwrClient(std::string client_id, LineFitModel &model,
                   SyntheticDataset &training_dataset,
                   SyntheticDataset &validation_dataset,
                   SyntheticDataset &test_dataset);
  void set_parameters(flwr_local::Parameters params);

  virtual flwr_local::ParametersRes get_parameters() override;
  virtual flwr_local::PropertiesRes
  get_properties(flwr_local::PropertiesIns ins) override;
  virtual flwr_local::EvaluateRes
  evaluate(flwr_local::EvaluateIns ins) override;
  virtual flwr_local::FitRes fit(flwr_local::FitIns ins) override;

private:
  int64_t client_id;
  LineFitModel &model;
  SyntheticDataset &training_dataset;
  SyntheticDataset &validation_dataset;
  SyntheticDataset &test_dataset;
};
