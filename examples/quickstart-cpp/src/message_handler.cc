#include "message_handler.h"

#include "recorddict_serde.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace flwr_quickstart {
namespace {

double unix_seconds_now() {
  using clock = std::chrono::system_clock;
  const auto now = clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

std::string make_message_id(uint64_t node_id) {
  static std::atomic<uint64_t> counter{0};
  std::ostringstream oss;
  oss << "cpp-" << node_id << "-" << std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count()
      << "-" << counter.fetch_add(1, std::memory_order_relaxed);
  return oss.str();
}

flwr::proto::Message make_reply(const flwr::proto::Message &in,
                                const flwr_local::RecordSet &content) {
  flwr::proto::Message out;
  *out.mutable_content() = recorddict_to_proto(content);

  const auto &in_meta = in.metadata();
  auto *meta = out.mutable_metadata();
  meta->set_run_id(in_meta.run_id());
  meta->set_message_id(make_message_id(in_meta.dst_node_id()));
  meta->set_src_node_id(in_meta.dst_node_id());
  meta->set_dst_node_id(in_meta.src_node_id());
  meta->set_reply_to_message_id(in_meta.message_id());
  meta->set_group_id(in_meta.group_id());
  meta->set_created_at(unix_seconds_now());
  const double max_ttl = in_meta.created_at() + in_meta.ttl() - meta->created_at();
  meta->set_ttl(max_ttl > 1.0 ? max_ttl : 1.0);
  meta->set_message_type(in_meta.message_type());
  if (in_meta.has_dst_task_id()) {
    meta->set_src_task_id(in_meta.dst_task_id());
  }
  if (in_meta.has_src_task_id()) {
    meta->set_dst_task_id(in_meta.src_task_id());
  }
  return out;
}

} // namespace

flwr::proto::Message handle_message(flwr_local::Client *client,
                                    const flwr::proto::Message &message) {
  const std::string type = message.metadata().message_type();
  std::cout << "[flwr-cpp] received message_type=" << type
            << " message_id=" << message.metadata().message_id()
            << " run_id=" << message.metadata().run_id() << std::endl;

  if (type == "get_parameters") {
    return make_reply(message,
                      recorddict_from_get_parameters_res(client->get_parameters()));
  }
  if (type == "get_properties") {
    flwr_local::PropertiesIns ins;
    return make_reply(message,
                      recorddict_from_get_properties_res(client->get_properties(ins)));
  }
  if (type == "train") {
    flwr_local::RecordSet recordset = recorddict_from_proto(message.content());
    flwr_local::FitRes fit_res = client->fit(recorddict_to_fit_ins(recordset));
    return make_reply(message, recorddict_from_fit_res(fit_res));
  }
  if (type == "evaluate") {
    flwr_local::RecordSet recordset = recorddict_from_proto(message.content());
    flwr_local::EvaluateRes eval_res =
        client->evaluate(recorddict_to_evaluate_ins(recordset));
    return make_reply(message, recorddict_from_evaluate_res(eval_res));
  }
  if (type == "reconnect") {
    flwr_local::RecordSet recordset;
    recordset.setConfigsRecords({{"config", {{"reason", 4}}}});
    return make_reply(message, recordset);
  }

  throw std::runtime_error("Unknown Flower message type: " + type);
}

} // namespace flwr_quickstart
