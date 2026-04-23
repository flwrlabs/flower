use crate::flwr_proto as proto;

pub fn validate_task_ins(task_ins: &proto::TaskIns, discard_reconnect_ins: bool) -> bool {
    match &task_ins.task {
        Some(task) => {
            let has_legacy_server_msg = task.legacy_server_message.as_ref().map_or(false, |lsm| {
                if discard_reconnect_ins {
                    !matches!(lsm.msg, Some(proto::server_message::Msg::ReconnectIns(..)))
                } else {
                    true
                }
            });

            has_legacy_server_msg || task.sa.is_some()
        }
        None => false,
    }
}

pub fn validate_task_res(task_res: &proto::TaskRes) -> bool {
    // Check for initialization of fields in TaskRes
    let task_res_is_uninitialized =
        task_res.task_id.is_empty() && task_res.group_id.is_empty() && task_res.workload_id == 0;

    // Check for initialization of fields in Task, if Task is present
    let task_is_uninitialized = task_res.task.as_ref().map_or(true, |task| {
        task.producer.is_none() && task.consumer.is_none() && task.ancestry.is_empty()
    });

    task_res_is_uninitialized && task_is_uninitialized
}

pub fn get_task_ins(pull_task_ins_response: &proto::PullTaskInsResponse) -> Option<proto::TaskIns> {
    if pull_task_ins_response.task_ins_list.is_empty() {
        return None;
    }

    let task_ins = pull_task_ins_response.task_ins_list.first();
    return task_ins.cloned();
}

pub fn get_server_message_from_task_ins(
    task_ins: &proto::TaskIns,
    exclude_reconnect_ins: bool,
) -> Option<proto::ServerMessage> {
    if !validate_task_ins(task_ins, exclude_reconnect_ins) {
        return None;
    }

    match &task_ins.task {
        Some(task) => {
            if let Some(legacy_server_message) = &task.legacy_server_message {
                Some(legacy_server_message.clone())
            } else {
                None
            }
        }
        None => None,
    }
}

pub fn wrap_client_message_in_task_res(client_message: proto::ClientMessage) -> proto::TaskRes {
    return proto::TaskRes {
        task_id: "".to_string(),
        group_id: "".to_string(),
        workload_id: 0,
        task: Some(proto::Task {
            ancestry: vec![],
            legacy_client_message: Some(client_message),
            ..Default::default()
        }),
    };
}

pub fn configure_task_res(
    mut task_res: proto::TaskRes,
    ref_task_ins: &proto::TaskIns,
    producer: proto::Node,
) -> proto::TaskRes {
    // Set group_id and workload_id
    task_res.group_id = ref_task_ins.group_id.clone();
    task_res.workload_id = ref_task_ins.workload_id;

    // Check if task_res has a task field set; if not, initialize it.
    if task_res.task.is_none() {
        task_res.task = Some(proto::Task::default());
    }

    // Assuming the task is now Some, unwrap it and set its fields.
    if let Some(ref mut task) = task_res.task {
        task.producer = Some(producer);
        task.consumer = ref_task_ins.task.as_ref().and_then(|t| t.producer.clone());

        // Set ancestry to contain just ref_task_ins.task_id
        task.ancestry = vec![ref_task_ins.task_id.clone()];
    }

    task_res
}
