pub mod client;
pub mod grpc_bidi;
pub mod grpc_rere;
pub mod message_handler;
pub mod serde;
pub mod start;
pub mod task_handler;
pub mod typing;

pub mod flwr_proto {
    include!("flwr.proto.rs");
}
