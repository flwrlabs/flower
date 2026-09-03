/*************************************************************************************************
 *
 * @file grpc-rere.h
 *
 * @brief Provide functions for establishing gRPC request-response communication
 *
 * @author The Flower Authors
 *
 * @version 1.0
 *
 * @date 06/11/2023
 *
 *************************************************************************************************/

#ifndef GRPC_RERE_H
#define GRPC_RERE_H
#pragma once
#include "communicator.h"
#include "flwr/proto/fleet.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <openssl/evp.h>
#include <string>

class gRPCRereCommunicator : public Communicator {
public:
  gRPCRereCommunicator(std::string server_address, int grpc_max_message_length);
  ~gRPCRereCommunicator();

  bool send_register_node(flwr::proto::RegisterNodeFleetRequest request,
                          flwr::proto::RegisterNodeFleetResponse *response);

  bool send_activate_node(flwr::proto::ActivateNodeRequest request,
                          flwr::proto::ActivateNodeResponse *response);

  bool send_deactivate_node(flwr::proto::DeactivateNodeRequest request,
                            flwr::proto::DeactivateNodeResponse *response);

  bool send_unregister_node(flwr::proto::UnregisterNodeFleetRequest request,
                            flwr::proto::UnregisterNodeFleetResponse *response);

  bool send_heartbeat(flwr::proto::SendNodeHeartbeatRequest request,
                      flwr::proto::SendNodeHeartbeatResponse *response);

  bool send_pull_messages(flwr::proto::PullMessagesRequest request,
                          flwr::proto::PullMessagesResponse *response);

  bool send_push_messages(flwr::proto::PushMessagesRequest request,
                          flwr::proto::PushMessagesResponse *response);

  bool send_get_run(flwr::proto::GetRunRequest request,
                    flwr::proto::GetRunResponse *response);

  bool send_get_fab(flwr::proto::GetFabRequest request,
                    flwr::proto::GetFabResponse *response);

  bool send_pull_object(flwr::proto::PullObjectRequest request,
                        flwr::proto::PullObjectResponse *response);

  bool send_push_object(flwr::proto::PushObjectRequest request,
                        flwr::proto::PushObjectResponse *response);

  bool send_confirm_message_received(
      flwr::proto::ConfirmMessageReceivedRequest request,
      flwr::proto::ConfirmMessageReceivedResponse *response);

  const std::string &public_key_pem() const { return public_key_pem_; }

private:
  std::unique_ptr<flwr::proto::Fleet::Stub> stub;
  EVP_PKEY *pkey_ = nullptr;
  std::string public_key_pem_;

  void add_auth_metadata(grpc::ClientContext &ctx);
};

#endif
