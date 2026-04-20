#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include "flwr/proto/fleet.pb.h"
#include <chrono>
#include <optional>

class Communicator {
public:
  virtual bool send_create_node(flwr::proto::CreateNodeRequest request,
                                flwr::proto::CreateNodeResponse *response) = 0;

  virtual bool send_delete_node(flwr::proto::DeleteNodeRequest request,
                                flwr::proto::DeleteNodeResponse *response) = 0;

  virtual bool
  send_pull_task_ins(flwr::proto::PullMessagesRequest request,
                     flwr::proto::PullMessagesResponse *response) = 0;

  virtual bool
  send_push_task_res(flwr::proto::PushMessagesRequest request,
                     flwr::proto::PushMessagesResponse *response) = 0;
};

void create_node(Communicator *communicator);
void delete_node(Communicator *communicator);
void send(Communicator *communicator, flwr::proto::Message task_res);
std::optional<flwr::proto::Message> receive(Communicator *communicator);

#endif
