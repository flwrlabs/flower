use crate::client;
use crate::flwr_proto as proto;
use crate::serde;
use crate::task_handler;

fn reconnect(reconnect_msg: proto::server_message::ReconnectIns) -> (proto::ClientMessage, i64) {
    let mut reason = proto::Reason::Ack;
    let mut sleep_duration = 0;
    if reconnect_msg.seconds != 0 {
        reason = proto::Reason::Reconnect;
        sleep_duration = reconnect_msg.seconds;
    }

    let disconnect_res = proto::client_message::DisconnectRes {
        reason: reason.into(),
    };

    return (
        proto::ClientMessage {
            msg: Some(proto::client_message::Msg::DisconnectRes(disconnect_res)),
        },
        sleep_duration,
    );
}

fn get_properties(
    client: &dyn client::Client,
    get_properties_msg: proto::server_message::GetPropertiesIns,
) -> proto::ClientMessage {
    let get_properties_res = serde::get_properties_res_to_proto(
        client.get_properties(serde::get_properties_ins_from_proto(get_properties_msg)),
    );

    proto::ClientMessage {
        msg: Some(proto::client_message::Msg::GetPropertiesRes(
            get_properties_res,
        )),
    }
}

fn get_parameters(
    client: &dyn client::Client,
    // get_parameters_msg: proto::server_message::GetParametersIns,
) -> proto::ClientMessage {
    let res = serde::parameter_res_to_proto(client.get_parameters());
    proto::ClientMessage {
        msg: Some(proto::client_message::Msg::GetParametersRes(res)),
    }
}

fn fit(
    client: &dyn client::Client,
    fit_msg: proto::server_message::FitIns,
) -> proto::ClientMessage {
    let res = serde::fit_res_to_proto(client.fit(serde::fit_ins_from_proto(fit_msg)));
    proto::ClientMessage {
        msg: Some(proto::client_message::Msg::FitRes(res)),
    }
}

fn evaluate(
    client: &dyn client::Client,
    evaluate_msg: proto::server_message::EvaluateIns,
) -> proto::ClientMessage {
    let res =
        serde::evaluate_res_to_proto(client.evaluate(serde::evaluate_ins_from_proto(evaluate_msg)));
    proto::ClientMessage {
        msg: Some(proto::client_message::Msg::EvaluateRes(res)),
    }
}

fn handle_legacy_message(
    client: &dyn client::Client,
    server_msg: proto::ServerMessage,
) -> Result<(proto::ClientMessage, i64, bool), &str> {
    match server_msg.msg {
        Some(proto::server_message::Msg::ReconnectIns(reconnect_ins)) => {
            let rec = reconnect(reconnect_ins);
            Ok((rec.0, rec.1, false))
        }
        Some(proto::server_message::Msg::GetParametersIns(_)) => {
            Ok((get_parameters(client), 0, true))
        }
        Some(proto::server_message::Msg::FitIns(fit_ins)) => Ok((fit(client, fit_ins), 0, true)),
        Some(proto::server_message::Msg::EvaluateIns(evaluate_ins)) => {
            Ok((evaluate(client, evaluate_ins), 0, true))
        }
        _ => Err("Unknown server message"),
    }
}

pub fn handle(
    client: &dyn client::Client,
    task_ins: proto::TaskIns,
) -> Result<(proto::TaskRes, i64, bool), &str> {
    let server_msg = task_handler::get_server_message_from_task_ins(&task_ins, false);
    if server_msg.is_none() {
        return Err("Not implemented");
    }
    let (client_msg, sleep_duration, keep_going) =
        handle_legacy_message(client, server_msg.unwrap())?;
    let task_res = task_handler::wrap_client_message_in_task_res(client_msg);
    Ok((task_res, sleep_duration, keep_going))
}
