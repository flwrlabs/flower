#include "grpc_rere.h"
#include "flwr/proto/fleet.grpc.pb.h"

#include <openssl/bio.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/evp.h>
#include <openssl/pem.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <unistd.h>

gRPCRereCommunicator::gRPCRereCommunicator(std::string server_address,
                                           int grpc_max_message_length) {
  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(grpc_max_message_length);
  args.SetMaxSendMessageSize(grpc_max_message_length);

  // Establish an insecure gRPC connection to a gRPC server
  std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
      server_address, grpc::InsecureChannelCredentials(), args);

  // Create stub
  stub = flwr::proto::Fleet::NewStub(channel);

  // Generate EC key pair (SECP384R1) for node authentication.
  // Build key material by mixing /dev/urandom with container-unique sources
  // (hostname + PID + high-res timestamp) to avoid PRNG collisions between
  // containers that start simultaneously (known issue with Docker on macOS).
  uint8_t key_bytes[48] = {};

  FILE *f = fopen("/dev/urandom", "rb");
  if (f) {
    fread(key_bytes, 1, sizeof(key_bytes), f);
    fclose(f);
  }

  char hostname[256] = {};
  gethostname(hostname, sizeof(hostname));
  size_t hlen = strnlen(hostname, sizeof(hostname));

  pid_t pid = getpid();
  auto ts =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();

  for (size_t i = 0; i < sizeof(key_bytes); i++) {
    key_bytes[i] ^= static_cast<uint8_t>(hostname[i % (hlen ? hlen : 1)]);
    key_bytes[i] ^=
        reinterpret_cast<const uint8_t *>(&pid)[i % sizeof(pid)];
    key_bytes[i] ^=
        reinterpret_cast<const uint8_t *>(&ts)[i % sizeof(ts)];
  }

  // Construct the EC key from our private key scalar.
  EC_GROUP *group = EC_GROUP_new_by_curve_name(NID_secp384r1);
  EC_KEY *ec_key = EC_KEY_new();
  EC_KEY_set_group(ec_key, group);

  BIGNUM *priv_bn = BN_bin2bn(key_bytes, sizeof(key_bytes), nullptr);
  EC_KEY_set_private_key(ec_key, priv_bn);

  // Derive public key from private key scalar.
  EC_POINT *pub_pt = EC_POINT_new(group);
  EC_POINT_mul(group, pub_pt, priv_bn, nullptr, nullptr, nullptr);
  EC_KEY_set_public_key(ec_key, pub_pt);

  pkey_ = EVP_PKEY_new();
  EVP_PKEY_assign_EC_KEY(pkey_, ec_key);

  EC_POINT_free(pub_pt);
  BN_free(priv_bn);
  EC_GROUP_free(group);

  // Serialize public key to PEM format
  BIO *bio = BIO_new(BIO_s_mem());
  PEM_write_bio_PUBKEY(bio, pkey_);
  BUF_MEM *bptr;
  BIO_get_mem_ptr(bio, &bptr);
  public_key_pem_ = std::string(bptr->data, bptr->length);
  BIO_free(bio);
}

gRPCRereCommunicator::~gRPCRereCommunicator() {
  if (pkey_) {
    EVP_PKEY_free(pkey_);
    pkey_ = nullptr;
  }
}

void gRPCRereCommunicator::add_auth_metadata(grpc::ClientContext &ctx) {
  // Generate ISO 8601 UTC timestamp with microsecond precision
  auto now = std::chrono::system_clock::now();
  auto tt = std::chrono::system_clock::to_time_t(now);
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch()) %
            1000000;

  struct tm utc_tm = {};
  gmtime_r(&tt, &utc_tm);

  char date_buf[32];
  strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%S", &utc_tm);

  char us_buf[16];
  snprintf(us_buf, sizeof(us_buf), ".%06ld", us.count());

  std::string timestamp = std::string(date_buf) + us_buf + "+00:00";

  // Sign the timestamp with ECDSA/SHA256
  EVP_MD_CTX *md = EVP_MD_CTX_new();
  EVP_DigestSignInit(md, nullptr, EVP_sha256(), nullptr, pkey_);
  EVP_DigestSignUpdate(md, timestamp.data(), timestamp.size());
  size_t sig_len = 0;
  EVP_DigestSignFinal(md, nullptr, &sig_len);
  std::string signature(sig_len, '\0');
  EVP_DigestSignFinal(md, reinterpret_cast<uint8_t *>(&signature[0]), &sig_len);
  signature.resize(sig_len);
  EVP_MD_CTX_free(md);

  // Add authentication metadata (binary headers are base64-encoded by gRPC)
  ctx.AddMetadata("flwr-public-key-bin", public_key_pem_);
  ctx.AddMetadata("flwr-timestamp", timestamp);
  ctx.AddMetadata("flwr-signature-bin", signature);
}

bool gRPCRereCommunicator::send_register_node(
    flwr::proto::RegisterNodeFleetRequest request,
    flwr::proto::RegisterNodeFleetResponse *response) {
  grpc::ClientContext context;
  add_auth_metadata(context);
  grpc::Status status = stub->RegisterNode(&context, request, response);
  if (!status.ok()) {
    std::cerr << "RegisterNode RPC failed: " << status.error_message()
              << std::endl;
    return false;
  }
  return true;
}

bool gRPCRereCommunicator::send_activate_node(
    flwr::proto::ActivateNodeRequest request,
    flwr::proto::ActivateNodeResponse *response) {
  grpc::ClientContext context;
  add_auth_metadata(context);
  grpc::Status status = stub->ActivateNode(&context, request, response);
  if (!status.ok()) {
    std::cerr << "ActivateNode RPC failed: " << status.error_message()
              << std::endl;
    return false;
  }
  return true;
}

bool gRPCRereCommunicator::send_deactivate_node(
    flwr::proto::DeactivateNodeRequest request,
    flwr::proto::DeactivateNodeResponse *response) {
  grpc::ClientContext context;
  add_auth_metadata(context);
  grpc::Status status = stub->DeactivateNode(&context, request, response);
  if (!status.ok()) {
    std::cerr << "DeactivateNode RPC failed: " << status.error_message()
              << std::endl;
    return false;
  }
  return true;
}

bool gRPCRereCommunicator::send_unregister_node(
    flwr::proto::UnregisterNodeFleetRequest request,
    flwr::proto::UnregisterNodeFleetResponse *response) {
  grpc::ClientContext context;
  add_auth_metadata(context);
  grpc::Status status = stub->UnregisterNode(&context, request, response);
  if (!status.ok()) {
    std::cerr << "UnregisterNode RPC failed: " << status.error_message()
              << std::endl;
    return false;
  }
  return true;
}

bool gRPCRereCommunicator::send_heartbeat(
    flwr::proto::SendNodeHeartbeatRequest request,
    flwr::proto::SendNodeHeartbeatResponse *response) {
  grpc::ClientContext context;
  add_auth_metadata(context);
  grpc::Status status = stub->SendNodeHeartbeat(&context, request, response);
  if (!status.ok()) {
    std::cerr << "SendNodeHeartbeat RPC failed: " << status.error_message()
              << std::endl;
    return false;
  }
  return true;
}

bool gRPCRereCommunicator::send_pull_messages(
    flwr::proto::PullMessagesRequest request,
    flwr::proto::PullMessagesResponse *response) {
  grpc::ClientContext context;
  add_auth_metadata(context);
  grpc::Status status = stub->PullMessages(&context, request, response);
  if (!status.ok()) {
    std::cerr << "PullMessages RPC failed: " << status.error_message()
              << std::endl;
    return false;
  }
  return true;
}

bool gRPCRereCommunicator::send_push_messages(
    flwr::proto::PushMessagesRequest request,
    flwr::proto::PushMessagesResponse *response) {
  grpc::ClientContext context;
  add_auth_metadata(context);
  grpc::Status status = stub->PushMessages(&context, request, response);
  if (!status.ok()) {
    std::cerr << "PushMessages RPC failed: " << status.error_message()
              << std::endl;
    return false;
  }
  return true;
}

bool gRPCRereCommunicator::send_get_run(
    flwr::proto::GetRunRequest request,
    flwr::proto::GetRunResponse *response) {
  grpc::ClientContext context;
  add_auth_metadata(context);
  grpc::Status status = stub->GetRun(&context, request, response);
  if (!status.ok()) {
    std::cerr << "GetRun RPC failed: " << status.error_message() << std::endl;
    return false;
  }
  return true;
}

bool gRPCRereCommunicator::send_get_fab(
    flwr::proto::GetFabRequest request,
    flwr::proto::GetFabResponse *response) {
  grpc::ClientContext context;
  add_auth_metadata(context);
  grpc::Status status = stub->GetFab(&context, request, response);
  if (!status.ok()) {
    std::cerr << "GetFab RPC failed: " << status.error_message() << std::endl;
    return false;
  }
  return true;
}

bool gRPCRereCommunicator::send_pull_object(
    flwr::proto::PullObjectRequest request,
    flwr::proto::PullObjectResponse *response) {
  grpc::ClientContext context;
  add_auth_metadata(context);
  grpc::Status status = stub->PullObject(&context, request, response);
  if (!status.ok()) {
    std::cerr << "PullObject RPC failed: " << status.error_message()
              << std::endl;
    return false;
  }
  return true;
}

bool gRPCRereCommunicator::send_push_object(
    flwr::proto::PushObjectRequest request,
    flwr::proto::PushObjectResponse *response) {
  grpc::ClientContext context;
  add_auth_metadata(context);
  grpc::Status status = stub->PushObject(&context, request, response);
  if (!status.ok()) {
    std::cerr << "PushObject RPC failed: " << status.error_message()
              << std::endl;
    return false;
  }
  return true;
}

bool gRPCRereCommunicator::send_confirm_message_received(
    flwr::proto::ConfirmMessageReceivedRequest request,
    flwr::proto::ConfirmMessageReceivedResponse *response) {
  grpc::ClientContext context;
  add_auth_metadata(context);
  grpc::Status status =
      stub->ConfirmMessageReceived(&context, request, response);
  if (!status.ok()) {
    std::cerr << "ConfirmMessageReceived RPC failed: "
              << status.error_message() << std::endl;
    return false;
  }
  return true;
}
