/*************************************************************************************************
 *
 * @file message_handler.h
 *
 * @brief Handle server messages by calling appropriate client methods
 *
 * @author Lekang Jiang
 *
 * @version 1.0
 *
 * @date 04/09/2021
 *
 *************************************************************************************************/

#pragma once
#include "client.h"
#include "serde.h"

// Returns (reply_message, sleep_duration, keep_going)
std::tuple<flwr_local::Message, int, bool>
handle_message(flwr_local::Client *client, const flwr_local::Message &message);
