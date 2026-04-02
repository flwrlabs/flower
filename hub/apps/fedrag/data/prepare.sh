#!/bin/bash

# Default values
datasets=("statpearls" "textbooks")
index_num_chunks=100  # Default value for number of chunks

# This script will download all the corpus we need for the FedRAG workflow
# and also prepare the indices using the FAISS library for document retrieval

set -e
# Change directory to the script's directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --datasets)
            shift
            datasets=()  # Clear default datasets when --datasets is provided
            while [[ "$#" -gt 0 && ! "$1" =~ ^-- ]]; do
                datasets+=("$1")
                shift
            done
            ;;
        --index_num_chunks)
            index_num_chunks="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done


# Construct the command to pass arguments to Python script
cmd=("python" "./prepare.py")

# Add datasets to the command
cmd+=("--datasets")
for dataset in "${datasets[@]}"; do
    cmd+=("$dataset")
done

# Add number of chunks to consider for the index of each corpus
cmd+=("--index_num_chunks")
cmd+=("$index_num_chunks")

# Run the Python script
echo "Executing:" "${cmd[@]}"
"${cmd[@]}"
