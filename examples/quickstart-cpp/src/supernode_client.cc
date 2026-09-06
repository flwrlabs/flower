#include "supernode_client.h"

#include "flwr/proto/fleet.grpc.pb.h"
#include "flwr/proto/heartbeat.pb.h"
#include "message_handler.h"

#include <chrono>
#include <ctime>
#include <deque>
#include <grpcpp/grpcpp.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <openssl/evp.h>
#include <openssl/obj_mac.h>
#include <openssl/pem.h>
#include <openssl/sha.h>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace flwr_quickstart {
namespace {

constexpr double kHeartbeatIntervalSeconds = 30.0;
constexpr const char *kPublicKeyHeader = "flwr-public-key-bin";
constexpr const char *kTimestampHeader = "flwr-timestamp";
constexpr const char *kSignatureHeader = "flwr-signature-bin";

struct EvpPkeyDeleter {
  void operator()(EVP_PKEY *key) const { EVP_PKEY_free(key); }
};

using EvpPkeyPtr = std::unique_ptr<EVP_PKEY, EvpPkeyDeleter>;

struct ObjectBundle {
  flwr::proto::ObjectTree tree;
  std::map<std::string, std::string> objects;
};

EvpPkeyPtr generate_private_key() {
  EVP_PKEY_CTX *raw_ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_EC, nullptr);
  if (raw_ctx == nullptr) {
    throw std::runtime_error("EVP_PKEY_CTX_new_id failed");
  }
  std::unique_ptr<EVP_PKEY_CTX, decltype(&EVP_PKEY_CTX_free)> ctx(
      raw_ctx, EVP_PKEY_CTX_free);
  if (EVP_PKEY_keygen_init(ctx.get()) <= 0 ||
      EVP_PKEY_CTX_set_ec_paramgen_curve_nid(ctx.get(), NID_secp384r1) <= 0) {
    throw std::runtime_error("EC keygen init failed");
  }
  EVP_PKEY *raw_key = nullptr;
  if (EVP_PKEY_keygen(ctx.get(), &raw_key) <= 0) {
    throw std::runtime_error("EC keygen failed");
  }
  return EvpPkeyPtr(raw_key);
}

std::string public_key_to_pem(EVP_PKEY *key) {
  BIO *raw_bio = BIO_new(BIO_s_mem());
  if (raw_bio == nullptr) {
    throw std::runtime_error("BIO_new failed");
  }
  std::unique_ptr<BIO, decltype(&BIO_free)> bio(raw_bio, BIO_free);
  if (PEM_write_bio_PUBKEY(bio.get(), key) != 1) {
    throw std::runtime_error("PEM_write_bio_PUBKEY failed");
  }
  char *data = nullptr;
  const long len = BIO_get_mem_data(bio.get(), &data);
  return std::string(data, static_cast<size_t>(len));
}

std::string utc_iso_timestamp() {
  using clock = std::chrono::system_clock;
  const auto now = clock::now();
  const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
  const auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(now - seconds)
          .count();
  std::time_t time = clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &time);
#else
  gmtime_r(&time, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
  std::ostringstream oss;
  oss << buf << "." << std::setfill('0') << std::setw(6) << micros << "+00:00";
  return oss.str();
}

std::string sign_message(EVP_PKEY *key, const std::string &message) {
  EVP_MD_CTX *raw_ctx = EVP_MD_CTX_new();
  if (raw_ctx == nullptr) {
    throw std::runtime_error("EVP_MD_CTX_new failed");
  }
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(raw_ctx,
                                                             EVP_MD_CTX_free);
  if (EVP_DigestSignInit(ctx.get(), nullptr, EVP_sha256(), nullptr, key) <= 0 ||
      EVP_DigestSignUpdate(ctx.get(), message.data(), message.size()) <= 0) {
    throw std::runtime_error("EVP_DigestSign init/update failed");
  }
  size_t sig_len = 0;
  if (EVP_DigestSignFinal(ctx.get(), nullptr, &sig_len) <= 0) {
    throw std::runtime_error("EVP_DigestSignFinal length failed");
  }
  std::string signature(sig_len, '\0');
  if (EVP_DigestSignFinal(
          ctx.get(), reinterpret_cast<unsigned char *>(signature.data()),
          &sig_len) <= 0) {
    throw std::runtime_error("EVP_DigestSignFinal failed");
  }
  signature.resize(sig_len);
  return signature;
}

std::vector<std::string> split_csv(const std::string &value) {
  std::vector<std::string> out;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

std::string join_csv(const std::vector<std::string> &values) {
  std::ostringstream oss;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      oss << ",";
    }
    oss << values[i];
  }
  return oss.str();
}

std::string object_body(const std::string &content,
                        const std::string &expected_type = "") {
  const size_t divider = content.find('\0');
  if (divider == std::string::npos) {
    throw std::runtime_error("Invalid Flower object: missing header divider");
  }
  const std::string head = content.substr(0, divider);
  const size_t first_space = head.find(' ');
  const size_t last_space = head.rfind(' ');
  if (first_space == std::string::npos || last_space == std::string::npos ||
      first_space == last_space) {
    throw std::runtime_error("Invalid Flower object header");
  }
  const std::string type = head.substr(0, first_space);
  const size_t body_len = std::stoull(head.substr(last_space + 1));
  if (!expected_type.empty() && type != expected_type) {
    throw std::runtime_error("Unexpected Flower object type: " + type +
                             ", expected: " + expected_type);
  }
  std::string body = content.substr(divider + 1);
  if (body.size() != body_len) {
    throw std::runtime_error("Invalid Flower object body length");
  }
  return body;
}

std::vector<std::string> object_child_ids(const std::string &content) {
  const size_t divider = content.find('\0');
  if (divider == std::string::npos) {
    throw std::runtime_error("Invalid Flower object: missing header divider");
  }
  const std::string head = content.substr(0, divider);
  const size_t first_space = head.find(' ');
  const size_t last_space = head.rfind(' ');
  if (first_space == std::string::npos || last_space == std::string::npos ||
      first_space == last_space) {
    throw std::runtime_error("Invalid Flower object header");
  }
  const std::string children = head.substr(first_space + 1,
                                           last_space - first_space - 1);
  return split_csv(children);
}

std::string object_type(const std::string &content) {
  const size_t divider = content.find('\0');
  if (divider == std::string::npos) {
    throw std::runtime_error("Invalid Flower object: missing header divider");
  }
  const std::string head = content.substr(0, divider);
  const size_t first_space = head.find(' ');
  if (first_space == std::string::npos) {
    throw std::runtime_error("Invalid Flower object header");
  }
  return head.substr(0, first_space);
}

std::map<std::string, std::string> parse_json_string_map(
    const std::string &json) {
  std::map<std::string, std::string> out;
  const std::regex pair_re("\"([^\"]+)\"\\s*:\\s*\"([^\"]*)\"");
  for (std::sregex_iterator it(json.begin(), json.end(), pair_re), end; it != end;
       ++it) {
    out[(*it)[1].str()] = (*it)[2].str();
  }
  return out;
}

std::string parse_json_string_field(const std::string &json,
                                    const std::string &field) {
  const std::regex re("\"" + field + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  if (std::regex_search(json, match, re)) {
    return match[1].str();
  }
  return "";
}

std::vector<int32_t> parse_json_int_list(const std::string &json,
                                         const std::string &field) {
  const std::regex re("\"" + field + "\"\\s*:\\s*\\[([^\\]]*)\\]");
  std::smatch match;
  std::vector<int32_t> out;
  if (!std::regex_search(json, match, re)) {
    return out;
  }
  std::stringstream ss(match[1].str());
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      out.push_back(static_cast<int32_t>(std::stol(item)));
    }
  }
  return out;
}

std::string make_object_content(const std::string &type,
                                const std::vector<std::string> &children,
                                const std::string &body) {
  std::ostringstream head;
  head << type << " " << join_csv(children) << " " << body.size();
  std::string content = head.str();
  content.push_back('\0');
  content += body;
  return content;
}

std::string sha256_hex(const std::string &bytes) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size(),
         hash);
  std::ostringstream oss;
  for (unsigned char byte : hash) {
    oss << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<int>(byte);
  }
  return oss.str();
}

ObjectBundle make_leaf_object(const std::string &type, const std::string &body) {
  ObjectBundle bundle;
  const std::string content = make_object_content(type, {}, body);
  const std::string id = sha256_hex(content);
  bundle.tree.set_object_id(id);
  bundle.objects[id] = content;
  return bundle;
}

void merge_bundle(ObjectBundle *dst, const ObjectBundle &src) {
  *dst->tree.add_children() = src.tree;
  dst->objects.insert(src.objects.begin(), src.objects.end());
}

ObjectBundle make_array_object(const flwr::proto::Array &array) {
  ObjectBundle chunk = make_leaf_object("ArrayChunk", array.data());
  std::ostringstream body;
  body << "{\"dtype\":\"" << array.dtype() << "\",\"shape\":[";
  for (int i = 0; i < array.shape_size(); ++i) {
    if (i != 0) {
      body << ",";
    }
    body << array.shape(i);
  }
  body << "],\"stype\":\"" << array.stype()
       << "\",\"arraychunk_ids\":[0]}";

  ObjectBundle bundle;
  const std::string content =
      make_object_content("Array", {chunk.tree.object_id()}, body.str());
  const std::string id = sha256_hex(content);
  bundle.tree.set_object_id(id);
  merge_bundle(&bundle, chunk);
  bundle.objects[id] = content;
  return bundle;
}

ObjectBundle make_array_record_object(const flwr::proto::ArrayRecord &record) {
  ObjectBundle bundle;
  std::map<std::string, std::string> refs;
  for (const auto &item : record.items()) {
    ObjectBundle child = make_array_object(item.value());
    refs[item.key()] = child.tree.object_id();
    merge_bundle(&bundle, child);
  }
  std::ostringstream body;
  body << "{";
  bool first = true;
  for (const auto &[key, id] : refs) {
    if (!first) {
      body << ",";
    }
    first = false;
    body << "\"" << key << "\":\"" << id << "\"";
  }
  body << "}";
  std::vector<std::string> ids;
  for (const auto &child : bundle.tree.children()) {
    ids.push_back(child.object_id());
  }
  const std::string content = make_object_content("ArrayRecord", ids, body.str());
  const std::string id = sha256_hex(content);
  bundle.tree.set_object_id(id);
  bundle.objects[id] = content;
  return bundle;
}

ObjectBundle make_recorddict_object(const flwr::proto::RecordDict &recorddict) {
  ObjectBundle bundle;
  std::map<std::string, std::string> refs;
  for (const auto &item : recorddict.items()) {
    ObjectBundle child;
    if (item.has_array_record()) {
      child = make_array_record_object(item.array_record());
    } else if (item.has_config_record()) {
      std::string body;
      item.config_record().SerializeToString(&body);
      child = make_leaf_object("ConfigRecord", body);
    } else if (item.has_metric_record()) {
      std::string body;
      item.metric_record().SerializeToString(&body);
      child = make_leaf_object("MetricRecord", body);
    } else {
      continue;
    }
    refs[item.key()] = child.tree.object_id();
    merge_bundle(&bundle, child);
  }
  std::ostringstream body;
  body << "{";
  bool first = true;
  for (const auto &[key, id] : refs) {
    if (!first) {
      body << ",";
    }
    first = false;
    body << "\"" << key << "\":\"" << id << "\"";
  }
  body << "}";
  std::vector<std::string> ids;
  for (const auto &child : bundle.tree.children()) {
    ids.push_back(child.object_id());
  }
  const std::string content = make_object_content("RecordDict", ids, body.str());
  const std::string id = sha256_hex(content);
  bundle.tree.set_object_id(id);
  bundle.objects[id] = content;
  return bundle;
}

ObjectBundle make_message_object(const flwr::proto::Message &message) {
  ObjectBundle recorddict = make_recorddict_object(message.content());
  flwr::proto::Message body_message = message;
  body_message.mutable_metadata()->set_message_id("");
  body_message.clear_content();
  std::string body;
  body_message.SerializeToString(&body);

  ObjectBundle bundle;
  const std::string content =
      make_object_content("Message", {recorddict.tree.object_id()}, body);
  const std::string id = sha256_hex(content);
  bundle.tree.set_object_id(id);
  merge_bundle(&bundle, recorddict);
  bundle.objects[id] = content;
  return bundle;
}

const flwr::proto::ObjectTree *find_child_tree(const flwr::proto::ObjectTree &tree,
                                               const std::string &object_id) {
  for (const auto &child : tree.children()) {
    if (child.object_id() == object_id) {
      return &child;
    }
  }
  return nullptr;
}

class RereClient {
public:
  RereClient(const std::string &server_address, int grpc_max_message_length)
      : private_key_(generate_private_key()) {
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(grpc_max_message_length);
    args.SetMaxSendMessageSize(grpc_max_message_length);
    channel_ = grpc::CreateCustomChannel(server_address,
                                         grpc::InsecureChannelCredentials(), args);
    stub_ = flwr::proto::Fleet::NewStub(channel_);
    public_key_ = public_key_to_pem(private_key_.get());
  }

  void register_node() {
    flwr::proto::RegisterNodeFleetRequest request;
    request.set_public_key(public_key_);
    flwr::proto::RegisterNodeFleetResponse response;
    grpc::ClientContext context;
    add_auth_metadata(&context);
    const auto status = stub_->RegisterNode(&context, request, &response);
    if (!status.ok()) {
      throw std::runtime_error("RegisterNode failed: " + status.error_message());
    }
  }

  uint64_t activate_node() {
    flwr::proto::ActivateNodeRequest request;
    request.set_public_key(public_key_);
    request.set_heartbeat_interval(kHeartbeatIntervalSeconds);
    flwr::proto::ActivateNodeResponse response;
    grpc::ClientContext context;
    add_auth_metadata(&context);
    const auto status = stub_->ActivateNode(&context, request, &response);
    if (!status.ok()) {
      throw std::runtime_error("ActivateNode failed: " + status.error_message());
    }
    node_id_ = response.node_id();
    std::cout << "[flwr-cpp] activated node_id=" << node_id_ << std::endl;
    return node_id_;
  }

  void deactivate_node() {
    if (node_id_ == 0) {
      return;
    }
    flwr::proto::DeactivateNodeRequest request;
    request.set_node_id(node_id_);
    flwr::proto::DeactivateNodeResponse response;
    grpc::ClientContext context;
    add_auth_metadata(&context);
    const auto status = stub_->DeactivateNode(&context, request, &response);
    if (!status.ok()) {
      std::cerr << "[flwr-cpp] DeactivateNode failed: " << status.error_message()
                << std::endl;
    }
  }

  void unregister_node() {
    if (node_id_ == 0) {
      return;
    }
    flwr::proto::UnregisterNodeFleetRequest request;
    request.set_node_id(node_id_);
    flwr::proto::UnregisterNodeFleetResponse response;
    grpc::ClientContext context;
    add_auth_metadata(&context);
    const auto status = stub_->UnregisterNode(&context, request, &response);
    if (!status.ok()) {
      std::cerr << "[flwr-cpp] UnregisterNode failed: " << status.error_message()
                << std::endl;
    }
  }

  bool heartbeat() {
    flwr::proto::SendNodeHeartbeatRequest request;
    request.mutable_node()->set_node_id(node_id_);
    request.set_heartbeat_interval(kHeartbeatIntervalSeconds);
    flwr::proto::SendNodeHeartbeatResponse response;
    grpc::ClientContext context;
    add_auth_metadata(&context);
    const auto status = stub_->SendNodeHeartbeat(&context, request, &response);
    if (!status.ok()) {
      std::cerr << "[flwr-cpp] heartbeat failed: " << status.error_message()
                << std::endl;
      return false;
    }
    return response.success();
  }

  bool pull_message(flwr::proto::Message *message) {
    if (!pending_messages_.empty()) {
      *message = std::move(pending_messages_.front());
      pending_messages_.pop_front();
      return true;
    }

    flwr::proto::PullMessagesRequest request;
    request.mutable_node()->set_node_id(node_id_);
    flwr::proto::PullMessagesResponse response;
    grpc::ClientContext context;
    add_auth_metadata(&context);
    const auto status = stub_->PullMessages(&context, request, &response);
    if (!status.ok()) {
      throw std::runtime_error("PullMessages failed: " + status.error_message());
    }
    if (response.messages_list_size() == 0) {
      return false;
    }

    for (int idx = 0; idx < response.messages_list_size(); ++idx) {
      flwr::proto::Message pulled = response.messages_list(idx);
      if (!pulled.has_content() || pulled.content().items_size() == 0) {
        if (response.message_object_trees_size() <= idx ||
            response.message_object_trees(idx).children_size() == 0) {
          throw std::runtime_error(
              "Pulled message has no inline content or matching object tree");
        }
        const auto &message_tree = response.message_object_trees(idx);
        const auto &recorddict_tree = message_tree.children(0);
        *pulled.mutable_content() =
            inflate_recorddict(pulled.metadata().run_id(), recorddict_tree);
        confirm_message_received(pulled.metadata().run_id(),
                                 message_tree.object_id());
      }
      pending_messages_.push_back(std::move(pulled));
    }

    *message = std::move(pending_messages_.front());
    pending_messages_.pop_front();
    return true;
  }

  void push_message(const flwr::proto::Message &message, double runtime_sec) {
    ObjectBundle bundle = make_message_object(message);
    flwr::proto::Message wire_message = message;
    wire_message.mutable_metadata()->set_message_id(bundle.tree.object_id());
    wire_message.mutable_content()->Clear();

    flwr::proto::PushMessagesRequest request;
    request.mutable_node()->set_node_id(node_id_);
    *request.add_messages_list() = wire_message;
    *request.add_message_object_trees() = bundle.tree;
    request.add_clientapp_runtime_list(runtime_sec);
    flwr::proto::PushMessagesResponse response;
    grpc::ClientContext context;
    add_auth_metadata(&context);
    const auto status = stub_->PushMessages(&context, request, &response);
    if (!status.ok()) {
      throw std::runtime_error("PushMessages failed: " + status.error_message());
    }
    for (const auto &object_id : response.objects_to_push()) {
      auto it = bundle.objects.find(object_id);
      if (it == bundle.objects.end()) {
        std::cerr << "[flwr-cpp] missing requested object " << object_id
                  << std::endl;
        continue;
      }
      push_object(message.metadata().run_id(), object_id, it->second);
    }
  }

private:
  void add_auth_metadata(grpc::ClientContext *context) {
    const std::string timestamp = utc_iso_timestamp();
    const std::string signature = sign_message(private_key_.get(), timestamp);
    context->AddMetadata(kPublicKeyHeader, public_key_);
    context->AddMetadata(kTimestampHeader, timestamp);
    context->AddMetadata(kSignatureHeader, signature);
  }

  std::string pull_object(uint64_t run_id, const std::string &object_id) {
    for (int attempt = 0; attempt < 30; ++attempt) {
      flwr::proto::PullObjectRequest request;
      request.mutable_node()->set_node_id(node_id_);
      request.set_run_id(run_id);
      request.set_object_id(object_id);
      flwr::proto::PullObjectResponse response;
      grpc::ClientContext context;
      add_auth_metadata(&context);
      const auto status = stub_->PullObject(&context, request, &response);
      if (!status.ok()) {
        throw std::runtime_error("PullObject failed: " + status.error_message());
      }
      if (response.object_found() && response.object_available()) {
        return response.object_content();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    throw std::runtime_error("PullObject timed out for object_id=" + object_id);
  }

  void push_object(uint64_t run_id, const std::string &object_id,
                   const std::string &content) {
    flwr::proto::PushObjectRequest request;
    request.mutable_node()->set_node_id(node_id_);
    request.set_run_id(run_id);
    request.set_object_id(object_id);
    request.set_object_content(content);
    flwr::proto::PushObjectResponse response;
    grpc::ClientContext context;
    add_auth_metadata(&context);
    const auto status = stub_->PushObject(&context, request, &response);
    if (!status.ok()) {
      throw std::runtime_error("PushObject failed: " + status.error_message());
    }
    if (!response.stored()) {
      throw std::runtime_error("PushObject was rejected for object_id=" + object_id);
    }
  }

  void confirm_message_received(uint64_t run_id, const std::string &object_id) {
    flwr::proto::ConfirmMessageReceivedRequest request;
    request.mutable_node()->set_node_id(node_id_);
    request.set_run_id(run_id);
    request.set_message_object_id(object_id);
    flwr::proto::ConfirmMessageReceivedResponse response;
    grpc::ClientContext context;
    add_auth_metadata(&context);
    const auto status =
        stub_->ConfirmMessageReceived(&context, request, &response);
    if (!status.ok()) {
      std::cerr << "[flwr-cpp] ConfirmMessageReceived failed: "
                << status.error_message() << std::endl;
    }
  }

  flwr::proto::Array inflate_array(uint64_t run_id,
                                   const flwr::proto::ObjectTree &tree) {
    const std::string content = pull_object(run_id, tree.object_id());
    const std::string body = object_body(content, "Array");
    const std::vector<std::string> child_ids = object_child_ids(content);

    flwr::proto::Array array;
    array.set_dtype(parse_json_string_field(body, "dtype"));
    array.set_stype(parse_json_string_field(body, "stype"));
    for (int32_t dim : parse_json_int_list(body, "shape")) {
      array.add_shape(dim);
    }
    std::string data;
    for (int32_t idx : parse_json_int_list(body, "arraychunk_ids")) {
      if (idx < 0 || static_cast<size_t>(idx) >= child_ids.size()) {
        throw std::runtime_error("Invalid Array chunk index");
      }
      const flwr::proto::ObjectTree *chunk_tree =
          find_child_tree(tree, child_ids.at(static_cast<size_t>(idx)));
      if (chunk_tree == nullptr) {
        throw std::runtime_error("Array chunk object missing from ObjectTree");
      }
      const std::string chunk_content =
          pull_object(run_id, chunk_tree->object_id());
      data += object_body(chunk_content, "ArrayChunk");
    }
    array.set_data(data);
    return array;
  }

  flwr::proto::ArrayRecord
  inflate_array_record(uint64_t run_id, const flwr::proto::ObjectTree &tree) {
    const std::string content = pull_object(run_id, tree.object_id());
    const auto refs = parse_json_string_map(object_body(content, "ArrayRecord"));
    flwr::proto::ArrayRecord out;
    for (const auto &[key, object_id] : refs) {
      const flwr::proto::ObjectTree *child = find_child_tree(tree, object_id);
      if (child == nullptr) {
        throw std::runtime_error("ArrayRecord child object missing");
      }
      auto *item = out.add_items();
      item->set_key(key);
      *item->mutable_value() = inflate_array(run_id, *child);
    }
    return out;
  }

  flwr::proto::RecordDict
  inflate_recorddict(uint64_t run_id, const flwr::proto::ObjectTree &tree) {
    const std::string content = pull_object(run_id, tree.object_id());
    const auto refs = parse_json_string_map(object_body(content, "RecordDict"));
    flwr::proto::RecordDict out;
    for (const auto &[key, object_id] : refs) {
      const flwr::proto::ObjectTree *child = find_child_tree(tree, object_id);
      if (child == nullptr) {
        throw std::runtime_error("RecordDict child object missing");
      }
      const std::string child_content = pull_object(run_id, child->object_id());
      const std::string type = object_type(child_content);
      auto *item = out.add_items();
      item->set_key(key);
      if (type == "ArrayRecord") {
        *item->mutable_array_record() = inflate_array_record(run_id, *child);
      } else if (type == "ConfigRecord") {
        if (!item->mutable_config_record()->ParseFromString(
                object_body(child_content, "ConfigRecord"))) {
          throw std::runtime_error("Failed to parse ConfigRecord");
        }
      } else if (type == "MetricRecord") {
        if (!item->mutable_metric_record()->ParseFromString(
                object_body(child_content, "MetricRecord"))) {
          throw std::runtime_error("Failed to parse MetricRecord");
        }
      } else {
        throw std::runtime_error("Unsupported RecordDict child type: " + type);
      }
    }
    return out;
  }

  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<flwr::proto::Fleet::Stub> stub_;
  EvpPkeyPtr private_key_;
  std::string public_key_;
  uint64_t node_id_ = 0;
  std::deque<flwr::proto::Message> pending_messages_;
};

} // namespace

void start_client(const std::string &server_address, flwr_local::Client *client,
                  int grpc_max_message_length) {
  RereClient rere(server_address, grpc_max_message_length);
  rere.register_node();
  rere.activate_node();

  int idle_polls = 0;
  try {
    while (true) {
      rere.heartbeat();

      flwr::proto::Message in;
      if (!rere.pull_message(&in)) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (++idle_polls > 900) {
          std::cout << "[flwr-cpp] no messages for 30 minutes, exiting"
                    << std::endl;
          break;
        }
        continue;
      }
      idle_polls = 0;

      const auto start = std::chrono::steady_clock::now();
      flwr::proto::Message out = handle_message(client, in);
      const auto stop = std::chrono::steady_clock::now();
      const double runtime =
          std::chrono::duration<double>(stop - start).count();
      rere.push_message(out, runtime);
    }
  } catch (const std::exception &e) {
    std::cerr << "[flwr-cpp] fatal: " << e.what() << std::endl;
  }

  rere.deactivate_node();
  rere.unregister_node();
}

} // namespace flwr_quickstart
