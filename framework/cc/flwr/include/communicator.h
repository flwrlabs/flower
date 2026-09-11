#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include "flwr/proto/fab.pb.h"
#include "flwr/proto/fleet.pb.h"
#include "flwr/proto/heartbeat.pb.h"
#include "flwr/proto/run.pb.h"
#include "typing.h"
#include <optional>

class Communicator {
public:
  virtual ~Communicator() = default;

  virtual bool
  send_register_node(flwr::proto::RegisterNodeFleetRequest request,
                     flwr::proto::RegisterNodeFleetResponse *response) = 0;

  virtual bool
  send_activate_node(flwr::proto::ActivateNodeRequest request,
                     flwr::proto::ActivateNodeResponse *response) = 0;

  virtual bool
  send_deactivate_node(flwr::proto::DeactivateNodeRequest request,
                       flwr::proto::DeactivateNodeResponse *response) = 0;

  virtual bool
  send_unregister_node(flwr::proto::UnregisterNodeFleetRequest request,
                       flwr::proto::UnregisterNodeFleetResponse *response) = 0;

  virtual bool
  send_heartbeat(flwr::proto::SendNodeHeartbeatRequest request,
                 flwr::proto::SendNodeHeartbeatResponse *response) = 0;

  virtual bool
  send_pull_messages(flwr::proto::PullMessagesRequest request,
                     flwr::proto::PullMessagesResponse *response) = 0;

  virtual bool
  send_push_messages(flwr::proto::PushMessagesRequest request,
                     flwr::proto::PushMessagesResponse *response) = 0;

  virtual bool send_get_run(flwr::proto::GetRunRequest request,
                            flwr::proto::GetRunResponse *response) = 0;

  virtual bool send_get_fab(flwr::proto::GetFabRequest request,
                            flwr::proto::GetFabResponse *response) = 0;

  virtual bool send_pull_object(flwr::proto::PullObjectRequest request,
                                flwr::proto::PullObjectResponse *response) = 0;

  virtual bool send_push_object(flwr::proto::PushObjectRequest request,
                                flwr::proto::PushObjectResponse *response) = 0;

  virtual bool send_confirm_message_received(
      flwr::proto::ConfirmMessageReceivedRequest request,
      flwr::proto::ConfirmMessageReceivedResponse *response) = 0;

  virtual const std::string &public_key_pem() const = 0;
};

// Node lifecycle functions
void register_node(Communicator *communicator);
uint64_t activate_node(Communicator *communicator, double heartbeat_interval);
void deactivate_node(Communicator *communicator);
void unregister_node(Communicator *communicator);
bool send_heartbeat(Communicator *communicator, double heartbeat_interval);

// Message exchange functions
std::optional<flwr_local::Message> receive(Communicator *communicator);
void send(Communicator *communicator, const flwr_local::Message &message);

#endif
