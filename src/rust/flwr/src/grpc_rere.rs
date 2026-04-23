use crate::flwr_proto as proto;
use crate::task_handler;
use std::path::Path;

use tonic::transport::channel::ClientTlsConfig;
use tonic::transport::{Certificate, Channel};

const KEY_TASK_INS: &str = "current_task_ins";
const KEY_NODE: &str = "node";

pub struct GrpcRereConnection {
    stub: proto::fleet_client::FleetClient<tonic::transport::Channel>,
    state: std::collections::HashMap<String, Option<proto::TaskIns>>,
    node_store: std::collections::HashMap<String, Option<proto::Node>>,
}

impl GrpcRereConnection {
    pub async fn new(
        server_address: &str,
        root_certificates: Option<&Path>,
    ) -> Result<GrpcRereConnection, Box<dyn std::error::Error>> {
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
        let stub = proto::fleet_client::FleetClient::new(channel.clone());
        let state = std::collections::HashMap::from([(KEY_TASK_INS.to_string(), None)]);
        let node_store = std::collections::HashMap::from([(KEY_NODE.to_string(), None)]);

        Ok(GrpcRereConnection {
            stub,
            state,
            node_store,
        })
    }

    pub async fn create_node(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let create_node_request = proto::CreateNodeRequest::default();
        let create_node_response = self
            .stub
            .create_node(create_node_request)
            .await?
            .into_inner();
        self.node_store
            .insert(KEY_NODE.to_string(), create_node_response.node);
        Ok(())
    }

    pub async fn delete_node(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let node = match self.node_store.get(&KEY_NODE.to_string()) {
            Some(Some(n)) => n.clone(),
            _ => {
                eprintln!("Node instance missing");
                return Err("Node instance missing".into());
            }
        };

        let delete_node_request = proto::DeleteNodeRequest { node: Some(node) };
        self.stub.delete_node(delete_node_request).await?;
        Ok(())
    }

    pub async fn receive(&mut self) -> Result<Option<proto::TaskIns>, Box<dyn std::error::Error>> {
        let node = match self.node_store.get(KEY_NODE) {
            Some(Some(n)) => n.clone(),
            _ => {
                eprintln!("Node instance missing");
                return Err("Node instance missing".into());
            }
        };

        let request = proto::PullTaskInsRequest {
            node: Some(node),
            ..Default::default()
        };
        let response = self.stub.pull_task_ins(request).await?.into_inner();

        let mut task_ins = task_handler::get_task_ins(&response);
        if let Some(ref ti) = task_ins {
            if !task_handler::validate_task_ins(ti, true) {
                task_ins = None;
            }
        }

        self.state
            .insert(KEY_TASK_INS.to_string(), task_ins.clone());

        Ok(task_ins)
    }

    pub async fn send(
        &mut self,
        task_res: proto::TaskRes,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node = match self.node_store.get(KEY_NODE) {
            Some(Some(n)) => n.clone(),
            _ => {
                eprintln!("Node instance missing");
                return Err("Node instance missing".into());
            }
        };

        let task_ins = match self.state.get(KEY_TASK_INS) {
            Some(Some(ti)) => ti.clone(),
            _ => {
                eprintln!("No current TaskIns");
                return Err("No current TaskIns".into());
            }
        };

        if !task_handler::validate_task_res(&task_res) {
            self.state.insert(KEY_TASK_INS.to_string(), None);
            eprintln!("TaskRes is invalid");
            return Err("TaskRes is invalid".into());
        }

        let task_res = task_handler::configure_task_res(task_res, &task_ins, node);

        let request = proto::PushTaskResRequest {
            task_res_list: vec![task_res],
        };
        self.stub.push_task_res(request).await?;

        self.state.insert(KEY_TASK_INS.to_string(), None);

        Ok(())
    }
}
