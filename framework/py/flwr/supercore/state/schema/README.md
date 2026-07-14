# State Entity Relationship Diagram

## Schema

<!-- BEGIN_SQLALCHEMY_DOCS -->
```mermaid

---
    config:
        layout: elk
---
erDiagram
  automation {
    INTEGER automation_id PK
    TIMESTAMP created_at
    VARCHAR fab_hash "nullable"
    VARCHAR fab_id "nullable"
    VARCHAR fab_version "nullable"
    VARCHAR federation_config "nullable"
    VARCHAR federation_id
    BIGINT fixed_interval "nullable"
    VARCHAR flwr_aid
    TIMESTAMP next_run_at
    VARCHAR override_config
    VARCHAR primary_task_type
    INTEGER remaining_runs "nullable"
    BIGINT series_id
    VARCHAR status
    TIMESTAMP stopped_at "nullable"
    TIMESTAMP updated_at
  }

  connector {
    VARCHAR connector_ref PK
    VARCHAR flwr_aid PK
    VARCHAR config_json
    VARCHAR credentials_json
  }

  connector_oauth_session {
    VARCHAR oauth_session_id PK
    TIMESTAMP completed_at "nullable"
    VARCHAR connector_ref
    TIMESTAMP created_at
    TIMESTAMP expires_at
    VARCHAR flwr_aid
    VARCHAR pkce_verifier "nullable"
    VARCHAR redirect_uri
    VARCHAR state
  }

  context {
    BIGINT run_id FK "nullable"
    BLOB context "nullable"
  }

  fab {
    VARCHAR fab_hash PK
    BLOB content
    VARCHAR verifications
  }

  logs {
    BIGINT run_id FK "nullable"
    VARCHAR log "nullable"
    BIGINT node_id "nullable"
    FLOAT timestamp "nullable"
  }

  message_ins {
    BIGINT run_id FK "nullable"
    BLOB content "nullable"
    FLOAT created_at "nullable"
    VARCHAR delivered_at "nullable"
    BIGINT dst_node_id "nullable"
    BLOB error "nullable"
    VARCHAR group_id "nullable"
    VARCHAR message_id UK "nullable"
    VARCHAR message_type "nullable"
    VARCHAR reply_to_message_id "nullable"
    BIGINT src_node_id "nullable"
    FLOAT ttl "nullable"
  }

  message_res {
    BIGINT run_id FK "nullable"
    BLOB content "nullable"
    FLOAT created_at "nullable"
    VARCHAR delivered_at "nullable"
    BIGINT dst_node_id "nullable"
    BLOB error "nullable"
    VARCHAR group_id "nullable"
    VARCHAR message_id UK "nullable"
    VARCHAR message_type "nullable"
    VARCHAR reply_to_message_id "nullable"
    BIGINT src_node_id "nullable"
    FLOAT ttl "nullable"
  }

  node {
    FLOAT heartbeat_interval "nullable"
    VARCHAR last_activated_at "nullable"
    VARCHAR last_deactivated_at "nullable"
    BIGINT node_id UK "nullable"
    FLOAT online_until "nullable"
    VARCHAR owner_aid "nullable"
    VARCHAR owner_name "nullable"
    BLOB public_key UK "nullable"
    VARCHAR registered_at "nullable"
    VARCHAR status "nullable"
    VARCHAR unregistered_at "nullable"
  }

  nonce_store {
    VARCHAR namespace PK
    VARCHAR nonce PK
    FLOAT expires_at
  }

  object_children {
    VARCHAR child_id PK,FK
    VARCHAR parent_id PK,FK
  }

  object_push_session_pending {
    VARCHAR object_id PK
    VARCHAR session_id PK,FK
  }

  object_push_session_roots {
    VARCHAR root_object_id PK
    VARCHAR session_id FK
  }

  object_push_sessions {
    VARCHAR session_id PK
    TIMESTAMP expires_at
    INTEGER pending_count
    BIGINT run_id
  }

  objects {
    VARCHAR object_id PK "nullable"
    BLOB content "nullable"
    INTEGER is_available
    INTEGER ref_count
  }

  objectstore_locks {
    VARCHAR lock_id PK
    INTEGER lock_value
  }

  run {
    BIGINT bytes_recv "nullable"
    BIGINT bytes_sent "nullable"
    FLOAT clientapp_runtime "nullable"
    VARCHAR fab_hash "nullable"
    VARCHAR fab_id "nullable"
    VARCHAR fab_version "nullable"
    VARCHAR federation_config "nullable"
    VARCHAR federation_id "nullable"
    VARCHAR flwr_aid "nullable"
    VARCHAR override_config "nullable"
    BIGINT primary_task_id
    BIGINT run_id UK "nullable"
    BIGINT series_id "nullable"
    VARCHAR usage_reported_at
  }

  run_connector {
    VARCHAR connector_ref PK
    BIGINT run_id PK
  }

  run_objects {
    VARCHAR object_id PK,FK
    BIGINT run_id PK
  }

  run_series {
    BIGINT series_id PK
    TIMESTAMP created_at
    VARCHAR description "nullable"
    VARCHAR federation_id
    TIMESTAMP updated_at
  }

  series_context {
    BIGINT series_id PK
    BLOB context "nullable"
  }

  series_runs {
    INTEGER id PK
    BIGINT run_id UK
    BIGINT series_id
  }

  task {
    TIMESTAMP active_until "nullable"
    VARCHAR connector_ref "nullable"
    VARCHAR details
    VARCHAR fab_hash "nullable"
    TIMESTAMP finished_at "nullable"
    VARCHAR model_ref "nullable"
    TIMESTAMP pending_at
    BIGINT run_id
    TIMESTAMP running_at "nullable"
    TIMESTAMP starting_at "nullable"
    VARCHAR sub_status
    BIGINT task_id UK
    VARCHAR token "nullable"
    VARCHAR type
  }

  task_event {
    INTEGER id PK
    BIGINT task_id FK
    VARCHAR data
    VARCHAR event
    BIGINT run_id
    TIMESTAMP timestamp
  }

  task_logs {
    BIGINT task_id FK
    VARCHAR log
    FLOAT timestamp
  }

  task_message {
    VARCHAR message_id PK
    BIGINT dst_task_id FK
    BIGINT src_task_id FK
    BLOB content "nullable"
    FLOAT created_at
    BLOB error "nullable"
    VARCHAR message_type
    VARCHAR reply_to_message_id "nullable"
    BIGINT run_id
    FLOAT ttl
  }

  task_usage {
    INTEGER id PK
    BIGINT task_id FK
    TIMESTAMP created_at
    BIGINT input_tokens "nullable"
    BIGINT output_tokens "nullable"
    TIMESTAMP reported_at "nullable"
    BIGINT run_id
    BIGINT total_tokens "nullable"
    VARCHAR usage_type
  }

  run ||--o| context : run_id
  run ||--o{ logs : run_id
  run ||--o{ message_ins : run_id
  run ||--o{ message_res : run_id
  objects ||--o| object_children : parent_id
  objects ||--o| object_children : child_id
  object_push_sessions ||--o| object_push_session_pending : session_id
  object_push_sessions ||--o{ object_push_session_roots : session_id
  objects ||--o| run_objects : object_id
  task ||--o{ task_event : task_id
  task ||--o{ task_logs : task_id
  task ||--o{ task_message : src_task_id
  task ||--o{ task_message : dst_task_id
  task ||--o{ task_usage : task_id

```
<!-- END_SQLALCHEMY_DOCS -->
