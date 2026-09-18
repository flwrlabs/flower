// Regression test for the RecordSet tensor-ordering bug.
//
// `parameters_to_parameters_record` stores tensors under decimal-string keys
// ("0", "1", ... "10"). `parameters_record_to_parameters` used to iterate that
// `std::map` directly, so it read the keys back in lexicographic order, where
// "10" sorts before "2". Any model with 10 or more tensors was silently
// reassembled in the wrong order.
//
// The test drives the public serde API: a FitRes is converted to a RecordSet
// and back to a FitIns, which exercises the same
// parameters_to_parameters_record -> parameters_record_to_parameters path that
// the client uses for every train and evaluate reply.
//
// Wired into CMake as the `test_tensor_order` target.

#include "recorddict_serde.h"

#include <iostream>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const std::string &what) {
  if (condition) {
    std::cout << "  ok   - " << what << "\n";
  } else {
    std::cout << "  FAIL - " << what << "\n";
    ++failures;
  }
}

std::string tensor_of(double value) {
  return std::string(reinterpret_cast<const char *>(&value), sizeof(value));
}

// Round-trip `count` tensors through FitRes -> RecordSet -> FitIns and confirm
// the order survives.
void check_round_trip(size_t count) {
  std::list<std::string> tensors;
  for (size_t i = 0; i < count; ++i) {
    tensors.push_back(tensor_of(static_cast<double>(i)));
  }
  const flwr_local::Parameters params(tensors, "cpp_double");
  const flwr_local::FitRes fit_res(params, 1, 1, 0.0f, {});

  // FitRes -> RecordSet (stores tensors under "0", "1", ...).
  const flwr_local::RecordSet recordset =
      flwr_quickstart::recorddict_from_fit_res(fit_res);

  // Re-key that same ParametersRecord as a FitIns input and read it back.
  const auto &stored = recordset.getParametersRecords().at("fitres.parameters");
  flwr_local::RecordSet ins_recordset;
  ins_recordset.setParametersRecords({{"fitins.parameters", stored}});
  ins_recordset.setConfigsRecords({{"fitins.config", {{"k", 1}}}});
  // FitIns accessors are non-const in the Flower C++ SDK.
  flwr_local::FitIns ins = flwr_quickstart::recorddict_to_fit_ins(ins_recordset);

  const auto &before = params.getTensors();
  const auto after = ins.getParameters().getTensors();

  bool same = before.size() == after.size();
  if (same) {
    auto it_before = before.begin();
    auto it_after = after.begin();
    for (; it_before != before.end(); ++it_before, ++it_after) {
      if (*it_before != *it_after) {
        same = false;
        break;
      }
    }
  }
  expect(same, std::to_string(count) + " tensors keep their order");
  expect(ins.getParameters().getTensor_type() == "cpp_double",
         std::to_string(count) + " tensors keep tensor_type");
}

} // namespace

int main() {
  std::cout << "test_tensor_order\n";

  // 10 is the boundary where lexicographic order diverges from numeric order.
  check_round_trip(1);
  check_round_trip(2);
  check_round_trip(9);
  check_round_trip(10);
  check_round_trip(11);
  check_round_trip(12);
  check_round_trip(21);
  check_round_trip(100);

  if (failures == 0) {
    std::cout << "PASS\n";
    return 0;
  }
  std::cout << "FAILURES: " << failures << "\n";
  return 1;
}
