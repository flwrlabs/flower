#include "message_handler.h"
#include <chrono>
#include <stdexcept>

flwr_local::RecordDict _get_parameters(flwr_local::Client *client) {
  return recorddict_from_get_parameters_res(client->get_parameters());
}

flwr_local::RecordDict _fit(flwr_local::Client *client,
                            const flwr_local::RecordDict &content) {
  flwr_local::FitIns fit_ins = recorddict_to_fit_ins(content, true);
  flwr_local::FitRes fit_res = client->fit(fit_ins);
  return recorddict_from_fit_res(fit_res);
}

flwr_local::RecordDict _evaluate(flwr_local::Client *client,
                                 const flwr_local::RecordDict &content) {
  flwr_local::EvaluateIns evaluate_ins =
      recorddict_to_evaluate_ins(content, true);
  flwr_local::EvaluateRes evaluate_res = client->evaluate(evaluate_ins);
  return recorddict_from_evaluate_res(evaluate_res);
}

std::tuple<flwr_local::Message, int, bool>
handle_message(flwr_local::Client *client,
               const flwr_local::Message &message) {
  const std::string &msg_type = message.metadata.message_type;

  flwr_local::RecordDict result_content;
  int sleep_duration = 0;
  bool keep_going = true;

  if (msg_type == "reconnect") {
    keep_going = false;
  } else if (msg_type == "get_parameters") {
    result_content = _get_parameters(client);
  } else if (msg_type == "train") {
    if (message.content) {
      result_content = _fit(client, *message.content);
    }
  } else if (msg_type == "evaluate") {
    if (message.content) {
      result_content = _evaluate(client, *message.content);
    }
  } else {
    throw std::runtime_error("Unknown message type: " + msg_type);
  }

  // Build reply Message
  flwr_local::Message reply;
  reply.metadata.run_id = message.metadata.run_id;
  reply.metadata.src_node_id = message.metadata.dst_node_id;
  reply.metadata.dst_node_id = message.metadata.src_node_id;
  reply.metadata.reply_to_message_id = message.metadata.message_id;
  reply.metadata.group_id = message.metadata.group_id;
  reply.metadata.message_type = msg_type;
  reply.metadata.ttl = 3600.0;
  reply.metadata.created_at =
      static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count());
  reply.content = result_content;

  return {reply, sleep_duration, keep_going};
}
