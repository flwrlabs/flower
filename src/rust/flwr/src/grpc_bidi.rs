use crate::flwr_proto as proto;
use std::path::Path;

use async_channel::Sender;
use futures::StreamExt;
use tonic::transport::channel::ClientTlsConfig;
use tonic::transport::{Certificate, Channel};
use tonic::{Request, Streaming};

use uuid::Uuid;

pub struct GrpcConnection {
    channel: tonic::transport::Channel,
    stub: proto::flower_service_client::FlowerServiceClient<tonic::transport::Channel>,
    queue: Sender<proto::ClientMessage>,
    server_message_iterator: Streaming<proto::ServerMessage>,
}

impl GrpcConnection {
    pub async fn new(
        server_address: &str,
        root_certificates: Option<&Path>,
    ) -> Result<GrpcConnection, Box<dyn std::error::Error>> {
        let mut builder = Channel::builder(server_address.parse()?);

        // For now, skipping max_message_length because it's not directly supported in Tonic
        // Check Tonic's documentation or source for workarounds or updates regarding this

        if let Some(root_certificates) = root_certificates {
            let pem = tokio::fs::read(root_certificates).await?;
            let cert = Certificate::from_pem(pem);
            let tls_config = ClientTlsConfig::new().ca_certificate(cert);
            builder = builder.tls_config(tls_config)?;
        }

        let channel = builder.connect().await?;
        let mut stub = proto::flower_service_client::FlowerServiceClient::new(channel.clone());

        let (tx, rx) = async_channel::bounded(1); // This is our queue equivalent in async Rust

        let response = stub.join(Request::new(rx)).await?;
        let server_message_stream = response.into_inner();

        Ok(GrpcConnection {
            channel,
            stub,
            queue: tx,
            server_message_iterator: server_message_stream,
        })
    }

    pub async fn receive(&mut self) -> Result<proto::TaskIns, Box<dyn std::error::Error>> {
        if let Some(Ok(server_message)) = self.server_message_iterator.next().await {
            let task_ins = proto::TaskIns {
                group_id: "".to_string(),
                workload_id: 0,
                task_id: Uuid::new_v4().to_string(),
                task: Some(proto::Task {
                    producer: Some(proto::Node {
                        node_id: 0,
                        anonymous: true,
                    }),
                    consumer: Some(proto::Node {
                        node_id: 0,
                        anonymous: true,
                    }),
                    ancestry: vec![],
                    legacy_server_message: Some(server_message),
                    ..Default::default()
                }),
            };
            Ok(task_ins)
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to get server message.",
            )))
        }
    }

    pub async fn send(&self, task_res: proto::TaskRes) -> Result<(), Box<dyn std::error::Error>> {
        let client_message = task_res
            .task
            .unwrap_or_default()
            .legacy_client_message
            .unwrap();
        self.queue.send(client_message).await?;
        Ok(())
    }
}
