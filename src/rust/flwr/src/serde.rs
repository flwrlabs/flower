use crate::flwr_proto as proto;
use crate::typing as local;

pub fn parameters_to_proto(parameters: local::Parameters) -> proto::Parameters {
    return proto::Parameters {
        tensors: parameters.tensors,
        tensor_type: parameters.tensor_type,
    };
}

pub fn parameters_from_proto(params_msg: proto::Parameters) -> local::Parameters {
    return local::Parameters {
        tensors: params_msg.tensors,
        tensor_type: params_msg.tensor_type,
    };
}
pub fn scalar_to_proto(scalar: local::Scalar) -> proto::Scalar {
    match scalar {
        local::Scalar::Bool(value) => proto::Scalar {
            scalar: Some(proto::scalar::Scalar::Bool(value)),
        },
        local::Scalar::Bytes(value) => proto::Scalar {
            scalar: Some(proto::scalar::Scalar::Bytes(value)),
        },
        local::Scalar::Float(value) => proto::Scalar {
            scalar: Some(proto::scalar::Scalar::Double(value as f64)),
        },
        local::Scalar::Int(value) => proto::Scalar {
            scalar: Some(proto::scalar::Scalar::Sint64(value as i64)),
        },
        local::Scalar::Str(value) => proto::Scalar {
            scalar: Some(proto::scalar::Scalar::String(value)),
        },
    }
}

pub fn scalar_from_proto(scalar_msg: proto::Scalar) -> local::Scalar {
    match &scalar_msg.scalar {
        Some(proto::scalar::Scalar::Double(value)) => local::Scalar::Float(*value as f32),
        Some(proto::scalar::Scalar::Sint64(value)) => local::Scalar::Int(*value as i32),
        Some(proto::scalar::Scalar::Bool(value)) => local::Scalar::Bool(*value),
        Some(proto::scalar::Scalar::String(value)) => local::Scalar::Str(value.clone()),
        Some(proto::scalar::Scalar::Bytes(value)) => local::Scalar::Bytes(value.clone()),
        None => panic!("Error scalar type"),
    }
}

pub fn metrics_to_proto(
    metrics: local::Metrics,
) -> std::collections::HashMap<String, proto::Scalar> {
    let mut proto_metrics = std::collections::HashMap::new();

    for (key, value) in metrics.iter() {
        proto_metrics.insert(key.clone(), scalar_to_proto(value.clone()));
    }

    return proto_metrics;
}

pub fn metrics_from_proto(
    proto_metrics: std::collections::HashMap<String, proto::Scalar>,
) -> local::Metrics {
    let mut metrics = local::Metrics::new();

    for (key, value) in proto_metrics.iter() {
        metrics.insert(key.clone(), scalar_from_proto(value.clone()));
    }

    return metrics;
}

pub fn parameter_res_to_proto(
    res: local::GetParametersRes,
) -> proto::client_message::GetParametersRes {
    return proto::client_message::GetParametersRes {
        parameters: Some(parameters_to_proto(res.parameters)),
        status: Some(status_to_proto(res.status)),
    };
}

pub fn fit_ins_from_proto(fit_ins_msg: proto::server_message::FitIns) -> local::FitIns {
    local::FitIns {
        parameters: parameters_from_proto(fit_ins_msg.parameters.unwrap()),
        config: metrics_from_proto(fit_ins_msg.config.into_iter().collect()),
    }
}

pub fn fit_res_to_proto(res: local::FitRes) -> proto::client_message::FitRes {
    return proto::client_message::FitRes {
        parameters: Some(parameters_to_proto(res.parameters)),
        num_examples: res.num_examples.into(),
        metrics: if res.metrics.len() > 0 {
            metrics_to_proto(res.metrics)
        } else {
            Default::default()
        },
        status: Some(status_to_proto(res.status)),
    };
}

pub fn evaluate_ins_from_proto(
    evaluate_ins_msg: proto::server_message::EvaluateIns,
) -> local::EvaluateIns {
    local::EvaluateIns {
        parameters: parameters_from_proto(evaluate_ins_msg.parameters.unwrap()),
        config: metrics_from_proto(evaluate_ins_msg.config.into_iter().collect()),
    }
}

pub fn evaluate_res_to_proto(res: local::EvaluateRes) -> proto::client_message::EvaluateRes {
    return proto::client_message::EvaluateRes {
        loss: res.loss.into(),
        num_examples: res.num_examples.into(),
        metrics: if res.metrics.len() > 0 {
            metrics_to_proto(res.metrics)
        } else {
            Default::default()
        },
        status: Default::default(),
    };
}

fn status_to_proto(status: local::Status) -> proto::Status {
    return proto::Status {
        code: status.code as i32,
        message: status.message,
    };
}

pub fn get_properties_ins_from_proto(
    get_properties_msg: proto::server_message::GetPropertiesIns,
) -> local::GetPropertiesIns {
    return local::GetPropertiesIns {
        config: properties_from_proto(get_properties_msg.config),
    };
}

pub fn get_properties_res_to_proto(
    res: local::GetPropertiesRes,
) -> proto::client_message::GetPropertiesRes {
    return proto::client_message::GetPropertiesRes {
        properties: properties_to_proto(res.properties),
        status: Some(status_to_proto(res.status)),
    };
}

fn properties_from_proto(
    proto: std::collections::HashMap<String, proto::Scalar>,
) -> local::Properties {
    let mut properties = std::collections::HashMap::new();
    for (k, v) in proto.iter() {
        properties.insert(k.clone(), scalar_from_proto(v.clone()));
    }
    return properties;
}

fn properties_to_proto(
    properties: local::Properties,
) -> std::collections::HashMap<String, proto::Scalar> {
    let mut proto = std::collections::HashMap::new();
    for (k, v) in properties.iter() {
        proto.insert(k.clone(), scalar_to_proto(v.clone()));
    }
    return proto;
}
