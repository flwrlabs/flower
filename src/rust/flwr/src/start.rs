use std::path::Path;
use std::time::Duration;

use crate::grpc_bidi as bidi;
use crate::grpc_rere as rere;
use crate::message_handler as handler;

use crate::client;

pub async fn start_client<C>(
    address: &str,
    client: &C,
    root_certificates: Option<&Path>,
    transport: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>>
where
    C: client::Client,
{
    loop {
        let mut sleep_duration: i64 = 0;
        if transport.is_some() && transport == Some("rere") {
            let mut conn = rere::GrpcRereConnection::new(address, root_certificates).await?;
            // Register node
            conn.create_node().await?;
            loop {
                match conn.receive().await {
                    Ok(Some(task_ins)) => match handler::handle(client, task_ins) {
                        Ok((task_res, new_sleep_duration, keep_going)) => {
                            println!("Task received! {}", task_res.task.is_some());
                            sleep_duration = new_sleep_duration;
                            conn.send(task_res).await?;
                            if !keep_going {
                                break;
                            }
                        }
                        Err(e) => {
                            eprintln!("Error: {}", e);
                            return Err("Couldn't handle task".into());
                        }
                    },
                    Ok(None) => {
                        println!("No task received");
                        tokio::time::sleep(Duration::from_secs(3)).await; // Wait for 3s before asking again
                    }
                    Err(e) => {
                        eprintln!("Error: {}", e);
                        return Err("Couldn't receive task".into());
                    }
                }
            }
            // Unregister node
            conn.delete_node().await?;
        } else {
            let mut conn = bidi::GrpcConnection::new(address, root_certificates).await?;
            loop {
                match conn.receive().await {
                    Ok(task_ins) => match handler::handle(client, task_ins) {
                        Ok((task_res, new_sleep_duration, keep_going)) => {
                            sleep_duration = new_sleep_duration;
                            conn.send(task_res).await?;
                            if !keep_going {
                                break;
                            }
                        }
                        Err(e) => {
                            eprintln!("Error: {}", e);
                        }
                    },
                    Err(_) => {
                        tokio::time::sleep(Duration::from_secs(3)).await; // Wait for 3s before asking again
                    }
                }
            }
        }

        if sleep_duration == 0 {
            println!("Disconnect and shut down");
            break;
        }

        println!(
            "Disconnect, then re-establish connection after {} second(s)",
            sleep_duration
        );
        tokio::time::sleep(Duration::from_secs(sleep_duration as u64)).await;
    }
    Ok(())
}
