#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Node {
    #[prost(sint64, tag = "1")]
    pub node_id: i64,
    #[prost(bool, tag = "2")]
    pub anonymous: bool,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Status {
    #[prost(enumeration = "Code", tag = "1")]
    pub code: i32,
    #[prost(string, tag = "2")]
    pub message: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Parameters {
    #[prost(bytes = "vec", repeated, tag = "1")]
    pub tensors: ::prost::alloc::vec::Vec<::prost::alloc::vec::Vec<u8>>,
    #[prost(string, tag = "2")]
    pub tensor_type: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ServerMessage {
    #[prost(oneof = "server_message::Msg", tags = "1, 2, 3, 4, 5")]
    pub msg: ::core::option::Option<server_message::Msg>,
}
/// Nested message and enum types in `ServerMessage`.
pub mod server_message {
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct ReconnectIns {
        #[prost(int64, tag = "1")]
        pub seconds: i64,
    }
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct GetPropertiesIns {
        #[prost(map = "string, message", tag = "1")]
        pub config: ::std::collections::HashMap<
            ::prost::alloc::string::String,
            super::Scalar,
        >,
    }
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct GetParametersIns {
        #[prost(map = "string, message", tag = "1")]
        pub config: ::std::collections::HashMap<
            ::prost::alloc::string::String,
            super::Scalar,
        >,
    }
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct FitIns {
        #[prost(message, optional, tag = "1")]
        pub parameters: ::core::option::Option<super::Parameters>,
        #[prost(map = "string, message", tag = "2")]
        pub config: ::std::collections::HashMap<
            ::prost::alloc::string::String,
            super::Scalar,
        >,
    }
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct EvaluateIns {
        #[prost(message, optional, tag = "1")]
        pub parameters: ::core::option::Option<super::Parameters>,
        #[prost(map = "string, message", tag = "2")]
        pub config: ::std::collections::HashMap<
            ::prost::alloc::string::String,
            super::Scalar,
        >,
    }
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Msg {
        #[prost(message, tag = "1")]
        ReconnectIns(ReconnectIns),
        #[prost(message, tag = "2")]
        GetPropertiesIns(GetPropertiesIns),
        #[prost(message, tag = "3")]
        GetParametersIns(GetParametersIns),
        #[prost(message, tag = "4")]
        FitIns(FitIns),
        #[prost(message, tag = "5")]
        EvaluateIns(EvaluateIns),
    }
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ClientMessage {
    #[prost(oneof = "client_message::Msg", tags = "1, 2, 3, 4, 5")]
    pub msg: ::core::option::Option<client_message::Msg>,
}
/// Nested message and enum types in `ClientMessage`.
pub mod client_message {
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct DisconnectRes {
        #[prost(enumeration = "super::Reason", tag = "1")]
        pub reason: i32,
    }
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct GetPropertiesRes {
        #[prost(message, optional, tag = "1")]
        pub status: ::core::option::Option<super::Status>,
        #[prost(map = "string, message", tag = "2")]
        pub properties: ::std::collections::HashMap<
            ::prost::alloc::string::String,
            super::Scalar,
        >,
    }
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct GetParametersRes {
        #[prost(message, optional, tag = "1")]
        pub status: ::core::option::Option<super::Status>,
        #[prost(message, optional, tag = "2")]
        pub parameters: ::core::option::Option<super::Parameters>,
    }
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct FitRes {
        #[prost(message, optional, tag = "1")]
        pub status: ::core::option::Option<super::Status>,
        #[prost(message, optional, tag = "2")]
        pub parameters: ::core::option::Option<super::Parameters>,
        #[prost(int64, tag = "3")]
        pub num_examples: i64,
        #[prost(map = "string, message", tag = "4")]
        pub metrics: ::std::collections::HashMap<
            ::prost::alloc::string::String,
            super::Scalar,
        >,
    }
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct EvaluateRes {
        #[prost(message, optional, tag = "1")]
        pub status: ::core::option::Option<super::Status>,
        #[prost(float, tag = "2")]
        pub loss: f32,
        #[prost(int64, tag = "3")]
        pub num_examples: i64,
        #[prost(map = "string, message", tag = "4")]
        pub metrics: ::std::collections::HashMap<
            ::prost::alloc::string::String,
            super::Scalar,
        >,
    }
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Msg {
        #[prost(message, tag = "1")]
        DisconnectRes(DisconnectRes),
        #[prost(message, tag = "2")]
        GetPropertiesRes(GetPropertiesRes),
        #[prost(message, tag = "3")]
        GetParametersRes(GetParametersRes),
        #[prost(message, tag = "4")]
        FitRes(FitRes),
        #[prost(message, tag = "5")]
        EvaluateRes(EvaluateRes),
    }
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Scalar {
    /// The following `oneof` contains all types that ProtoBuf considers to be
    /// "Scalar Value Types". Commented-out types are listed for reference and
    /// might be enabled in future releases. Source:
    /// <https://developers.google.com/protocol-buffers/docs/proto3#scalar>
    #[prost(oneof = "scalar::Scalar", tags = "1, 8, 13, 14, 15")]
    pub scalar: ::core::option::Option<scalar::Scalar>,
}
/// Nested message and enum types in `Scalar`.
pub mod scalar {
    /// The following `oneof` contains all types that ProtoBuf considers to be
    /// "Scalar Value Types". Commented-out types are listed for reference and
    /// might be enabled in future releases. Source:
    /// <https://developers.google.com/protocol-buffers/docs/proto3#scalar>
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Scalar {
        #[prost(double, tag = "1")]
        Double(f64),
        /// float float = 2;
        /// int32 int32 = 3;
        /// int64 int64 = 4;
        /// uint32 uint32 = 5;
        /// uint64 uint64 = 6;
        /// sint32 sint32 = 7;
        #[prost(sint64, tag = "8")]
        Sint64(i64),
        /// fixed32 fixed32 = 9;
        /// fixed64 fixed64 = 10;
        /// sfixed32 sfixed32 = 11;
        /// sfixed64 sfixed64 = 12;
        #[prost(bool, tag = "13")]
        Bool(bool),
        #[prost(string, tag = "14")]
        String(::prost::alloc::string::String),
        #[prost(bytes, tag = "15")]
        Bytes(::prost::alloc::vec::Vec<u8>),
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum Code {
    Ok = 0,
    GetPropertiesNotImplemented = 1,
    GetParametersNotImplemented = 2,
    FitNotImplemented = 3,
    EvaluateNotImplemented = 4,
}
impl Code {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            Code::Ok => "OK",
            Code::GetPropertiesNotImplemented => "GET_PROPERTIES_NOT_IMPLEMENTED",
            Code::GetParametersNotImplemented => "GET_PARAMETERS_NOT_IMPLEMENTED",
            Code::FitNotImplemented => "FIT_NOT_IMPLEMENTED",
            Code::EvaluateNotImplemented => "EVALUATE_NOT_IMPLEMENTED",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "OK" => Some(Self::Ok),
            "GET_PROPERTIES_NOT_IMPLEMENTED" => Some(Self::GetPropertiesNotImplemented),
            "GET_PARAMETERS_NOT_IMPLEMENTED" => Some(Self::GetParametersNotImplemented),
            "FIT_NOT_IMPLEMENTED" => Some(Self::FitNotImplemented),
            "EVALUATE_NOT_IMPLEMENTED" => Some(Self::EvaluateNotImplemented),
            _ => None,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum Reason {
    Unknown = 0,
    Reconnect = 1,
    PowerDisconnected = 2,
    WifiUnavailable = 3,
    Ack = 4,
}
impl Reason {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            Reason::Unknown => "UNKNOWN",
            Reason::Reconnect => "RECONNECT",
            Reason::PowerDisconnected => "POWER_DISCONNECTED",
            Reason::WifiUnavailable => "WIFI_UNAVAILABLE",
            Reason::Ack => "ACK",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "UNKNOWN" => Some(Self::Unknown),
            "RECONNECT" => Some(Self::Reconnect),
            "POWER_DISCONNECTED" => Some(Self::PowerDisconnected),
            "WIFI_UNAVAILABLE" => Some(Self::WifiUnavailable),
            "ACK" => Some(Self::Ack),
            _ => None,
        }
    }
}
/// Generated client implementations.
pub mod flower_service_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    use tonic::codegen::http::Uri;
    #[derive(Debug, Clone)]
    pub struct FlowerServiceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl FlowerServiceClient<tonic::transport::Channel> {
        /// Attempt to create a new client by connecting to a given endpoint.
        pub async fn connect<D>(dst: D) -> Result<Self, tonic::transport::Error>
        where
            D: TryInto<tonic::transport::Endpoint>,
            D::Error: Into<StdError>,
        {
            let conn = tonic::transport::Endpoint::new(dst)?.connect().await?;
            Ok(Self::new(conn))
        }
    }
    impl<T> FlowerServiceClient<T>
    where
        T: tonic::client::GrpcService<tonic::body::BoxBody>,
        T::Error: Into<StdError>,
        T::ResponseBody: Body<Data = Bytes> + Send + 'static,
        <T::ResponseBody as Body>::Error: Into<StdError> + Send,
    {
        pub fn new(inner: T) -> Self {
            let inner = tonic::client::Grpc::new(inner);
            Self { inner }
        }
        pub fn with_origin(inner: T, origin: Uri) -> Self {
            let inner = tonic::client::Grpc::with_origin(inner, origin);
            Self { inner }
        }
        pub fn with_interceptor<F>(
            inner: T,
            interceptor: F,
        ) -> FlowerServiceClient<InterceptedService<T, F>>
        where
            F: tonic::service::Interceptor,
            T::ResponseBody: Default,
            T: tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
                Response = http::Response<
                    <T as tonic::client::GrpcService<tonic::body::BoxBody>>::ResponseBody,
                >,
            >,
            <T as tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
            >>::Error: Into<StdError> + Send + Sync,
        {
            FlowerServiceClient::new(InterceptedService::new(inner, interceptor))
        }
        /// Compress requests with the given encoding.
        ///
        /// This requires the server to support it otherwise it might respond with an
        /// error.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.send_compressed(encoding);
            self
        }
        /// Enable decompressing responses.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.accept_compressed(encoding);
            self
        }
        /// Limits the maximum size of a decoded message.
        ///
        /// Default: `4MB`
        #[must_use]
        pub fn max_decoding_message_size(mut self, limit: usize) -> Self {
            self.inner = self.inner.max_decoding_message_size(limit);
            self
        }
        /// Limits the maximum size of an encoded message.
        ///
        /// Default: `usize::MAX`
        #[must_use]
        pub fn max_encoding_message_size(mut self, limit: usize) -> Self {
            self.inner = self.inner.max_encoding_message_size(limit);
            self
        }
        pub async fn join(
            &mut self,
            request: impl tonic::IntoStreamingRequest<Message = super::ClientMessage>,
        ) -> std::result::Result<
            tonic::Response<tonic::codec::Streaming<super::ServerMessage>>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/flwr.proto.FlowerService/Join",
            );
            let mut req = request.into_streaming_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("flwr.proto.FlowerService", "Join"));
            self.inner.streaming(req, path, codec).await
        }
    }
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Task {
    #[prost(message, optional, tag = "1")]
    pub producer: ::core::option::Option<Node>,
    #[prost(message, optional, tag = "2")]
    pub consumer: ::core::option::Option<Node>,
    #[prost(string, tag = "3")]
    pub created_at: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub delivered_at: ::prost::alloc::string::String,
    #[prost(string, tag = "5")]
    pub ttl: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "6")]
    pub ancestry: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(message, optional, tag = "7")]
    pub sa: ::core::option::Option<SecureAggregation>,
    #[deprecated]
    #[prost(message, optional, tag = "101")]
    pub legacy_server_message: ::core::option::Option<ServerMessage>,
    #[deprecated]
    #[prost(message, optional, tag = "102")]
    pub legacy_client_message: ::core::option::Option<ClientMessage>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TaskIns {
    #[prost(string, tag = "1")]
    pub task_id: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub group_id: ::prost::alloc::string::String,
    #[prost(sint64, tag = "3")]
    pub workload_id: i64,
    #[prost(message, optional, tag = "4")]
    pub task: ::core::option::Option<Task>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TaskRes {
    #[prost(string, tag = "1")]
    pub task_id: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub group_id: ::prost::alloc::string::String,
    #[prost(sint64, tag = "3")]
    pub workload_id: i64,
    #[prost(message, optional, tag = "4")]
    pub task: ::core::option::Option<Task>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Value {
    #[prost(oneof = "value::Value", tags = "1, 2, 3, 4, 5, 21, 22, 23, 24, 25")]
    pub value: ::core::option::Option<value::Value>,
}
/// Nested message and enum types in `Value`.
pub mod value {
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct DoubleList {
        #[prost(double, repeated, tag = "1")]
        pub vals: ::prost::alloc::vec::Vec<f64>,
    }
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Sint64List {
        #[prost(sint64, repeated, tag = "1")]
        pub vals: ::prost::alloc::vec::Vec<i64>,
    }
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct BoolList {
        #[prost(bool, repeated, tag = "1")]
        pub vals: ::prost::alloc::vec::Vec<bool>,
    }
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct StringList {
        #[prost(string, repeated, tag = "1")]
        pub vals: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    }
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct BytesList {
        #[prost(bytes = "vec", repeated, tag = "1")]
        pub vals: ::prost::alloc::vec::Vec<::prost::alloc::vec::Vec<u8>>,
    }
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Value {
        /// Single element
        #[prost(double, tag = "1")]
        Double(f64),
        #[prost(sint64, tag = "2")]
        Sint64(i64),
        #[prost(bool, tag = "3")]
        Bool(bool),
        #[prost(string, tag = "4")]
        String(::prost::alloc::string::String),
        #[prost(bytes, tag = "5")]
        Bytes(::prost::alloc::vec::Vec<u8>),
        /// List types
        #[prost(message, tag = "21")]
        DoubleList(DoubleList),
        #[prost(message, tag = "22")]
        Sint64List(Sint64List),
        #[prost(message, tag = "23")]
        BoolList(BoolList),
        #[prost(message, tag = "24")]
        StringList(StringList),
        #[prost(message, tag = "25")]
        BytesList(BytesList),
    }
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SecureAggregation {
    #[prost(map = "string, message", tag = "1")]
    pub named_values: ::std::collections::HashMap<::prost::alloc::string::String, Value>,
}
/// CreateNode messages
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreateNodeRequest {}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreateNodeResponse {
    #[prost(message, optional, tag = "1")]
    pub node: ::core::option::Option<Node>,
}
/// DeleteNode messages
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeleteNodeRequest {
    #[prost(message, optional, tag = "1")]
    pub node: ::core::option::Option<Node>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeleteNodeResponse {}
/// PullTaskIns messages
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PullTaskInsRequest {
    #[prost(message, optional, tag = "1")]
    pub node: ::core::option::Option<Node>,
    #[prost(string, repeated, tag = "2")]
    pub task_ids: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PullTaskInsResponse {
    #[prost(message, optional, tag = "1")]
    pub reconnect: ::core::option::Option<Reconnect>,
    #[prost(message, repeated, tag = "2")]
    pub task_ins_list: ::prost::alloc::vec::Vec<TaskIns>,
}
/// PushTaskRes messages
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PushTaskResRequest {
    #[prost(message, repeated, tag = "1")]
    pub task_res_list: ::prost::alloc::vec::Vec<TaskRes>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PushTaskResResponse {
    #[prost(message, optional, tag = "1")]
    pub reconnect: ::core::option::Option<Reconnect>,
    #[prost(map = "string, uint32", tag = "2")]
    pub results: ::std::collections::HashMap<::prost::alloc::string::String, u32>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Reconnect {
    #[prost(uint64, tag = "1")]
    pub reconnect: u64,
}
/// Generated client implementations.
pub mod fleet_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    use tonic::codegen::http::Uri;
    #[derive(Debug, Clone)]
    pub struct FleetClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl FleetClient<tonic::transport::Channel> {
        /// Attempt to create a new client by connecting to a given endpoint.
        pub async fn connect<D>(dst: D) -> Result<Self, tonic::transport::Error>
        where
            D: TryInto<tonic::transport::Endpoint>,
            D::Error: Into<StdError>,
        {
            let conn = tonic::transport::Endpoint::new(dst)?.connect().await?;
            Ok(Self::new(conn))
        }
    }
    impl<T> FleetClient<T>
    where
        T: tonic::client::GrpcService<tonic::body::BoxBody>,
        T::Error: Into<StdError>,
        T::ResponseBody: Body<Data = Bytes> + Send + 'static,
        <T::ResponseBody as Body>::Error: Into<StdError> + Send,
    {
        pub fn new(inner: T) -> Self {
            let inner = tonic::client::Grpc::new(inner);
            Self { inner }
        }
        pub fn with_origin(inner: T, origin: Uri) -> Self {
            let inner = tonic::client::Grpc::with_origin(inner, origin);
            Self { inner }
        }
        pub fn with_interceptor<F>(
            inner: T,
            interceptor: F,
        ) -> FleetClient<InterceptedService<T, F>>
        where
            F: tonic::service::Interceptor,
            T::ResponseBody: Default,
            T: tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
                Response = http::Response<
                    <T as tonic::client::GrpcService<tonic::body::BoxBody>>::ResponseBody,
                >,
            >,
            <T as tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
            >>::Error: Into<StdError> + Send + Sync,
        {
            FleetClient::new(InterceptedService::new(inner, interceptor))
        }
        /// Compress requests with the given encoding.
        ///
        /// This requires the server to support it otherwise it might respond with an
        /// error.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.send_compressed(encoding);
            self
        }
        /// Enable decompressing responses.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.accept_compressed(encoding);
            self
        }
        /// Limits the maximum size of a decoded message.
        ///
        /// Default: `4MB`
        #[must_use]
        pub fn max_decoding_message_size(mut self, limit: usize) -> Self {
            self.inner = self.inner.max_decoding_message_size(limit);
            self
        }
        /// Limits the maximum size of an encoded message.
        ///
        /// Default: `usize::MAX`
        #[must_use]
        pub fn max_encoding_message_size(mut self, limit: usize) -> Self {
            self.inner = self.inner.max_encoding_message_size(limit);
            self
        }
        pub async fn create_node(
            &mut self,
            request: impl tonic::IntoRequest<super::CreateNodeRequest>,
        ) -> std::result::Result<
            tonic::Response<super::CreateNodeResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/flwr.proto.Fleet/CreateNode",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("flwr.proto.Fleet", "CreateNode"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn delete_node(
            &mut self,
            request: impl tonic::IntoRequest<super::DeleteNodeRequest>,
        ) -> std::result::Result<
            tonic::Response<super::DeleteNodeResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/flwr.proto.Fleet/DeleteNode",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("flwr.proto.Fleet", "DeleteNode"));
            self.inner.unary(req, path, codec).await
        }
        /// Retrieve one or more tasks, if possible
        ///
        /// HTTP API path: /api/v1/fleet/pull-task-ins
        pub async fn pull_task_ins(
            &mut self,
            request: impl tonic::IntoRequest<super::PullTaskInsRequest>,
        ) -> std::result::Result<
            tonic::Response<super::PullTaskInsResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/flwr.proto.Fleet/PullTaskIns",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("flwr.proto.Fleet", "PullTaskIns"));
            self.inner.unary(req, path, codec).await
        }
        /// Complete one or more tasks, if possible
        ///
        /// HTTP API path: /api/v1/fleet/push-task-res
        pub async fn push_task_res(
            &mut self,
            request: impl tonic::IntoRequest<super::PushTaskResRequest>,
        ) -> std::result::Result<
            tonic::Response<super::PushTaskResResponse>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/flwr.proto.Fleet/PushTaskRes",
            );
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("flwr.proto.Fleet", "PushTaskRes"));
            self.inner.unary(req, path, codec).await
        }
    }
}
