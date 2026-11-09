#pragma once

#include "client.h"
#include "flwr/proto/message.pb.h"

namespace flwr_quickstart {

flwr::proto::Message handle_message(flwr_local::Client *client,
                                    const flwr::proto::Message &message);

} // namespace flwr_quickstart
