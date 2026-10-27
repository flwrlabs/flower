#include "communicator.h"
#include "serde.h"

#include <iostream>
#include <mutex>

static std::mutex node_mutex;
static std::optional<uint64_t> stored_node_id;
static double stored_heartbeat_interval = 30.0;
static std::optional<flwr_local::Message> current_message;

void register_node(Communicator *communicator) {
  flwr::proto::RegisterNodeFleetRequest request;
  flwr::proto::RegisterNodeFleetResponse response;

  // The server identifies nodes by the public key in the request body.
  // It must match the key sent in the auth metadata header.
  request.set_public_key(communicator->public_key_pem());

  if (!communicator->send_register_node(request, &response)) {
    // "Public key already in use" means the node is already registered.
    // Proceed to ActivateNode which will look it up by the same key.
    std::cerr << "RegisterNode failed; proceeding to ActivateNode anyway."
              << std::endl;
  }
}

uint64_t activate_node(Communicator *communicator, double heartbeat_interval) {
  flwr::proto::ActivateNodeRequest request;
  flwr::proto::ActivateNodeResponse response;

  request.set_public_key(communicator->public_key_pem());
  request.set_heartbeat_interval(heartbeat_interval);

  if (!communicator->send_activate_node(request, &response)) {
    std::cerr << "Failed to activate node." << std::endl;
    return 0;
  }

  uint64_t node_id = response.node_id();
  {
    std::lock_guard<std::mutex> lock(node_mutex);
    stored_node_id = node_id;
    stored_heartbeat_interval = heartbeat_interval;
  }

  return node_id;
}

void deactivate_node(Communicator *communicator) {
  std::lock_guard<std::mutex> lock(node_mutex);
  if (!stored_node_id) {
    return;
  }

  flwr::proto::DeactivateNodeRequest request;
  flwr::proto::DeactivateNodeResponse response;

  request.set_node_id(*stored_node_id);

  communicator->send_deactivate_node(request, &response);
}

void unregister_node(Communicator *communicator) {
  std::lock_guard<std::mutex> lock(node_mutex);
  if (!stored_node_id) {
    return;
  }

  flwr::proto::UnregisterNodeFleetRequest request;
  flwr::proto::UnregisterNodeFleetResponse response;

  request.set_node_id(*stored_node_id);

  communicator->send_unregister_node(request, &response);

  stored_node_id.reset();
}

bool send_heartbeat(Communicator *communicator, double heartbeat_interval) {
  std::lock_guard<std::mutex> lock(node_mutex);
  if (!stored_node_id) {
    return false;
  }

  flwr::proto::SendNodeHeartbeatRequest request;
  flwr::proto::SendNodeHeartbeatResponse response;

  auto *node = new flwr::proto::Node();
  node->set_node_id(*stored_node_id);
  request.set_allocated_node(node);
  request.set_heartbeat_interval(heartbeat_interval);

  bool success = communicator->send_heartbeat(request, &response);
  return success && response.success();
}

std::optional<flwr_local::Message> receive(Communicator *communicator) {
  uint64_t node_id;
  {
    std::lock_guard<std::mutex> lock(node_mutex);
    if (!stored_node_id) {
      return std::nullopt;
    }
    node_id = *stored_node_id;
  }

  flwr::proto::PullMessagesRequest request;
  flwr::proto::PullMessagesResponse response;

  auto *node = new flwr::proto::Node();
  node->set_node_id(node_id);
  request.set_allocated_node(node);

  if (!communicator->send_pull_messages(request, &response)) {
    return std::nullopt;
  }

  if (response.messages_list_size() == 0) {
    return std::nullopt;
  }

  const auto &proto_msg = response.messages_list(0);
  flwr_local::Message msg = message_from_proto(proto_msg);

  // Inflate content from object tree (Flower 1.27+ inflatable objects protocol).
  // The proto message always carries an empty RecordDict as placeholder content;
  // actual payload is stored as inflatable objects referenced by the object tree.
  if (response.message_object_trees_size() > 0) {
    const auto &msg_tree = response.message_object_trees(0);

    // Collect all object IDs from the tree
    std::vector<std::string> obj_ids;
    collect_object_ids(msg_tree, obj_ids);

    // Pull all objects from server
    std::cerr << "[DEBUG recv] obj_ids count=" << obj_ids.size() << std::endl;
    std::map<std::string, std::string> objects;
    bool all_objects_pulled = true;
    for (const auto &obj_id : obj_ids) {
      flwr::proto::PullObjectRequest pull_req;
      flwr::proto::PullObjectResponse pull_resp;

      auto *pull_node = new flwr::proto::Node();
      pull_node->set_node_id(node_id);
      pull_req.set_allocated_node(pull_node);
      pull_req.set_run_id(msg.metadata.run_id);
      pull_req.set_object_id(obj_id);

      if (communicator->send_pull_object(pull_req, &pull_resp)) {
        std::cerr << "[DEBUG recv] obj " << obj_id.substr(0,16)
                  << " found=" << pull_resp.object_found()
                  << " avail=" << pull_resp.object_available()
                  << " size=" << pull_resp.object_content().size() << std::endl;
        if (pull_resp.object_found() && pull_resp.object_available()) {
          objects[obj_id] = pull_resp.object_content();
        } else {
          std::cerr << "[WARN recv] Object " << obj_id.substr(0,16)
                    << " not ready (found=" << pull_resp.object_found()
                    << " avail=" << pull_resp.object_available()
                    << "); will not confirm message" << std::endl;
          all_objects_pulled = false;
        }
      } else {
        std::cerr << "[DEBUG recv] PullObject RPC failed for " << obj_id.substr(0,16) << std::endl;
        all_objects_pulled = false;
      }
    }
    std::cerr << "[DEBUG recv] pulled " << objects.size() << " objects" << std::endl;

    // If any objects are missing, do not confirm — return nullopt so the
    // message remains on the server and can be retried on the next poll.
    if (!all_objects_pulled) {
      std::cerr << "[WARN recv] Incomplete objects; not confirming message."
                << std::endl;
      return std::nullopt;
    }

    // The Message has one child: the RecordDict (content)
    if (msg_tree.children_size() > 0) {
      const std::string &rd_obj_id = msg_tree.children(0).object_id();
      try {
        msg.content = inflate_recorddict(rd_obj_id, objects);
      } catch (const std::exception &e) {
        std::cerr << "Failed to inflate RecordDict: " << e.what() << std::endl;
        // Inflation failed — don't confirm, allow retry.
        return std::nullopt;
      }
    }

    // All objects pulled and inflated successfully — confirm receipt.
    flwr::proto::ConfirmMessageReceivedRequest confirm_req;
    flwr::proto::ConfirmMessageReceivedResponse confirm_resp;
    auto *confirm_node = new flwr::proto::Node();
    confirm_node->set_node_id(node_id);
    confirm_req.set_allocated_node(confirm_node);
    confirm_req.set_run_id(msg.metadata.run_id);
    confirm_req.set_message_object_id(msg.metadata.message_id);
    communicator->send_confirm_message_received(confirm_req, &confirm_resp);
  }

  current_message = msg;
  return msg;
}

void send(Communicator *communicator, const flwr_local::Message &message) {
  uint64_t node_id;
  {
    std::lock_guard<std::mutex> lock(node_mutex);
    if (!stored_node_id) {
      return;
    }
    node_id = *stored_node_id;
  }

  flwr::proto::PushMessagesRequest request;
  flwr::proto::PushMessagesResponse response;

  auto *node = new flwr::proto::Node();
  node->set_node_id(node_id);
  request.set_allocated_node(node);

  if (message.content) {
    // Deflate message content into inflatable objects (Flower 1.27+ protocol)
    DeflatedContent deflated = deflate_message(*message.content, message.metadata);

    // Build proto message with empty RecordDict placeholder content (Python's
    // Message constructor requires content to be set, even if empty), plus the
    // computed message_id.
    flwr_local::Message msg_no_content = message;
    msg_no_content.content = flwr_local::RecordDict{};
    msg_no_content.metadata.message_id = deflated.message_id;

    *request.add_messages_list() = message_to_proto(msg_no_content);
    *request.add_message_object_trees() = deflated.message_tree;

    if (!communicator->send_push_messages(request, &response)) {
      current_message.reset();
      return;
    }

    // Push objects that the server requested
    for (const auto &obj_id : response.objects_to_push()) {
      auto it = deflated.objects.find(obj_id);
      if (it == deflated.objects.end()) {
        std::cerr << "Server requested unknown object: " << obj_id << std::endl;
        continue;
      }

      flwr::proto::PushObjectRequest push_req;
      flwr::proto::PushObjectResponse push_resp;
      auto *push_node = new flwr::proto::Node();
      push_node->set_node_id(node_id);
      push_req.set_allocated_node(push_node);
      push_req.set_run_id(message.metadata.run_id);
      push_req.set_object_id(obj_id);
      push_req.set_object_content(it->second);

      communicator->send_push_object(push_req, &push_resp);
    }
  } else {
    // No content: send empty message
    *request.add_messages_list() = message_to_proto(message);
    communicator->send_push_messages(request, &response);
  }

  current_message.reset();
}
