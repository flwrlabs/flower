---
tags: [tabular, federated analytics, omop]
dataset: [artificial]
framework: [pandas]
---

# Federated Analytics with OMOP CDM using Flower

This example will show you how you can use Flower to run federated analytics workloads on distributed SQL databases, which are applicable to many biostatistics and healthcare use cases. You will use an artificial dataset generated in adherence to the [OMOP Common Data Model](https://www.ohdsi.org/data-standardization/), which uses the OHDSI standardized vocabularies widely adopted in clinical domains. You will also run this example with Flower's Deployment Engine to demonstrate how each SuperNode can be configured to connect to different PostgreSQL databases, respectively, while the database connection will be handled using SQLAlchemy and the `psycopg` adapter which is the latest implementation of the PostgreSQL adapter for Python.

## Set up the project

### Prerequisites

This example will make use of PostgreSQL databases running in containers. Make sure you have Docker and Docker Compose v2 installed.

### Clone the project

After cloning the project, this will create a new directory called `federated-analytics` containing the following files:

```shell
federated-analytics
├── federated-analytics
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your database connection and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `federated_analytics` package.

```shell
# From a new python environment, run:
pip install -e .
```

### Start PostgreSQL databases

First, create a database initialization script that defines the OMOP CDM table structure and generates sample data. Create a file named `db_init.sh`:

```bash
#!/usr/bin/env bash

SEED=${DB_SEED:-0.42}

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE TABLE person_measurements (
        person_id         INTEGER PRIMARY KEY,
        age               INTEGER,
        bmi               FLOAT,
        systolic_bp       INTEGER,
        diastolic_bp      INTEGER,
        ldl_cholesterol   FLOAT,
        hba1c             FLOAT
    );

    SELECT setseed($SEED);

    INSERT INTO person_measurements
    SELECT
        gs AS person_id,
        20 + (random() * 60)::INT        AS age,
        18 + (random() * 15)             AS bmi,
        100 + (random() * 40)::INT       AS systolic_bp,
        60 + (random() * 25)::INT        AS diastolic_bp,
        70 + (random() * 120)            AS ldl_cholesterol,
        4.5 + (random() * 4)             AS hba1c
    FROM generate_series(1, 100) gs;
EOSQL
```

Make it executable:

```shell
chmod +x db_init.sh
```

Next, create a script to start the PostgreSQL containers. Create a file named `db_start.sh`:

```bash
#!/usr/bin/env bash

set -e

N=${1:-2}   # number of PostgreSQL databases (default = 2)
BASE_PORT=5433

{
    echo "services:"

    for i in $(seq 1 "$N"); do
        PORT=$((BASE_PORT + i - 1))
        # Set a seed for each of the database for producing different random data
        SEED=$(echo "scale=2; $i / 100" | bc)
        cat <<EOF
  postgres_$i:
    image: postgres:18
    container_name: postgres_$i
    environment:
      POSTGRES_USER: flwrlabs
      POSTGRES_PASSWORD: flwrlabs
      POSTGRES_DB: flwrlabs
      DB_SEED: $SEED
    ports:
      - "$PORT:5432"
    volumes:
      - ./db_init.sh:/docker-entrypoint-initdb.d/init.sh:ro

EOF
    done
} | docker compose -f - up -d
```

Make it executable and run it to start two PostgreSQL databases:

```shell
chmod +x db_start.sh
./db_start.sh
```

> [!NOTE]
> To start more than two databases, pass the desired number as the first argument to the script, e.g. `./db_start.sh 3`.

### Update Flower Config

SuperLink connections are defined in the [Flower Configuration](https://flower.ai/docs/framework/main/en/ref-flower-configuration.html) file. If it doesn't exist, this file is created automatically for you when you use the Flower CLI command. Open the `config.toml` file (usually located at `$HOME/.flwr`) and add a new SuperLink connection at the end:

```toml
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
```

### Run with the Deployment Engine

For a basic execution of this federated analytics app, activate your environment and start the SuperLink process in insecure mode:

```shell
flower-superlink --insecure
```

Next, start two Supernodes and connect them to the SuperLink. You will need to specify different `--node-config` values so that each SuperNode will connect to different PostgreSQL databases.

The `db-url` parameter accepts a standard PostgreSQL connection string in the format `postgresql+psycopg://username:password@host:port/database`. Each SuperNode should point to a different database port (5433 and 5434) corresponding to the containers started by `db_start.sh`. In this example, the database URLs are set individually and passed to the `--node-config` parameter:

```shell
flower-supernode \
     --insecure \
     --superlink 127.0.0.1:9092 \
     --clientappio-api-address 127.0.0.1:9094 \
     --node-config="db-url='postgresql+psycopg://flwrlabs:flwrlabs@localhost:5433/flwrlabs'"
```

```shell
flower-supernode \
     --insecure \
     --superlink 127.0.0.1:9092 \
     --clientappio-api-address 127.0.0.1:9095 \
     --node-config="db-url='postgresql+psycopg://flwrlabs:flwrlabs@localhost:5434/flwrlabs'"
```

Finally, run the Flower App and follow the `ServerApp` logs to track the execution of the run:

```shell
flwr run . local-deployment --stream
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```shell
flwr run . local-deployment --run-config "selected-features='age,bmi'" --stream
```

The steps above are adapted from this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html). After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
