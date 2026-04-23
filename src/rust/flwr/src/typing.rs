use std::collections::HashMap;

// Scalar and Value types as described for ProtoBuf
pub type Metrics = HashMap<String, Scalar>;
pub type MetricsAggregationFn = fn(Vec<(i32, Metrics)>) -> Metrics;
pub type Config = HashMap<String, Scalar>;
pub type Properties = HashMap<String, Scalar>;

#[derive(Debug, Clone)]
pub enum Scalar {
    Bool(bool),
    Bytes(Vec<u8>),
    Float(f32),
    Int(i32),
    Str(String),
}

#[derive(Debug, Clone)]
pub enum Value {
    Bool(bool),
    Bytes(Vec<u8>),
    Float(f32),
    Int(i32),
    Str(String),
    ListBool(Vec<bool>),
    ListBytes(Vec<Vec<u8>>),
    ListFloat(Vec<f32>),
    ListInt(Vec<i32>),
    ListStr(Vec<String>),
}

#[derive(Debug, Clone, Copy)]
pub enum Code {
    OK = 0,
    GET_PROPERTIES_NOT_IMPLEMENTED = 1,
    GET_PARAMETERS_NOT_IMPLEMENTED = 2,
    FIT_NOT_IMPLEMENTED = 3,
    EVALUATE_NOT_IMPLEMENTED = 4,
}

#[derive(Debug, Clone)]
pub struct Status {
    pub code: Code,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct Parameters {
    pub tensors: Vec<Vec<u8>>,
    pub tensor_type: String,
}

#[derive(Debug, Clone)]
pub struct GetParametersIns {
    pub config: Config,
}

#[derive(Debug, Clone)]
pub struct GetParametersRes {
    pub status: Status,
    pub parameters: Parameters,
}

#[derive(Debug, Clone)]
pub struct FitIns {
    pub parameters: Parameters,
    pub config: Config,
}

#[derive(Debug, Clone)]
pub struct FitRes {
    pub status: Status,
    pub parameters: Parameters,
    pub num_examples: i32,
    pub metrics: Metrics,
}

#[derive(Debug, Clone)]
pub struct EvaluateIns {
    pub parameters: Parameters,
    pub config: Config,
}

#[derive(Debug, Clone)]
pub struct EvaluateRes {
    pub status: Status,
    pub loss: f32,
    pub num_examples: i32,
    pub metrics: Metrics,
}

#[derive(Debug, Clone)]
pub struct GetPropertiesIns {
    pub config: Config,
}

#[derive(Debug, Clone)]
pub struct GetPropertiesRes {
    pub status: Status,
    pub properties: Properties,
}

#[derive(Debug, Clone)]
pub struct ReconnectIns {
    pub seconds: Option<i32>,
}

#[derive(Debug, Clone)]
pub struct DisconnectRes {
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct ServerMessage {
    pub get_properties_ins: Option<GetPropertiesIns>,
    pub get_parameters_ins: Option<GetParametersIns>,
    pub fit_ins: Option<FitIns>,
    pub evaluate_ins: Option<EvaluateIns>,
}

#[derive(Debug, Clone)]
pub struct ClientMessage {
    pub get_properties_res: Option<GetPropertiesRes>,
    pub get_parameters_res: Option<GetParametersRes>,
    pub fit_res: Option<FitRes>,
    pub evaluate_res: Option<EvaluateRes>,
}
