#include "start.h"
#include <atomic>
#include <condition_variable>
#include <iostream>

// cppcheck-suppress unusedFunction
void start::start_client(std::string server_address, flwr_local::Client *client,
                         int grpc_max_message_length) {

  gRPCRereCommunicator communicator(server_address, grpc_max_message_length);

  // 1. Register node
  register_node(&communicator);

  // 2. Activate node (gets node_id)
  double heartbeat_interval = 30.0;
  uint64_t node_id = activate_node(&communicator, heartbeat_interval);
  std::cout << "Node activated with id: " << node_id << std::endl;
  if (node_id == 0) {
    std::cerr << "Failed to activate node; exiting." << std::endl;
    return;
  }

  // 3. Start heartbeat background thread
  std::atomic<bool> running{true};
  std::mutex cv_mutex;
  std::condition_variable cv;

  std::thread heartbeat_thread([&]() {
    while (running) {
      {
        std::unique_lock<std::mutex> lock(cv_mutex);
        cv.wait_for(lock, std::chrono::seconds((int)heartbeat_interval),
                    [&] { return !running.load(); });
      }
      if (!running)
        break;
      send_heartbeat(&communicator, heartbeat_interval);
    }
  });

  // 4. Message loop
  while (true) {
    std::optional<flwr_local::Message> message;
    try {
      message = receive(&communicator);
    } catch (const std::exception &e) {
      std::cerr << "[CRASH] receive() threw: " << e.what() << std::endl;
      break;
    }
    if (!message) {
      std::this_thread::sleep_for(std::chrono::seconds(3));
      continue;
    }

    std::cerr << "[DEBUG] Received msg type='" << message->metadata.message_type
              << "' has_content=" << (message->content ? "yes" : "no") << std::endl;

    std::tuple<flwr_local::Message, int, bool> handle_result;
    try {
      handle_result = handle_message(client, *message);
    } catch (const std::exception &e) {
      std::cerr << "[CRASH] handle_message() threw: " << e.what() << std::endl;
      break;
    }
    auto [reply, sleep_duration, keep_going] = handle_result;

    std::cerr << "[DEBUG] Sending reply type='" << reply.metadata.message_type
              << "' has_content=" << (reply.content ? "yes" : "no") << std::endl;

    try {
      send(&communicator, reply);
    } catch (const std::exception &e) {
      std::cerr << "[CRASH] send() threw: " << e.what() << std::endl;
      break;
    }

    if (!keep_going) {
      if (sleep_duration > 0) {
        std::cout << "Reconnect requested; sleeping " << sleep_duration
                  << "s before shutdown." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(sleep_duration));
      }
      break;
    }
  }

  // 5. Cleanup
  running = false;
  cv.notify_all();
  if (heartbeat_thread.joinable()) {
    heartbeat_thread.join();
  }

  deactivate_node(&communicator);
  unregister_node(&communicator);

  std::cout << "Disconnect and shut down." << std::endl;
}
