from flwr.app import ArrayRecord, RecordDict
import torch.nn as nn

classification_head_name = "classification-head"
personal_model_name = "personal_net"

def save_layer_weights_to_state(state: RecordDict, head: nn.Module):
    """Save last layer weights to state."""
    state[classification_head_name] = ArrayRecord(head.state_dict())

def load_layer_weights_from_state(state: RecordDict, head: nn.Module):
    """Load last layer weights from state and applies them to the model."""
    if classification_head_name not in state:
        return
    # Restore this client's saved classification head
    state_dict = state[classification_head_name].to_torch_state_dict()
    head.load_state_dict(state_dict, strict=True)

def save_model_from_to_state(state: RecordDict, net: nn.Module):
    """Save last weights to state."""
    state[personal_model_name] = ArrayRecord(net.state_dict())

def load_model_from_state(state: RecordDict, net: nn.Module):
    """Load last weights from state and applies them to the model."""
    if personal_model_name not in state:
        return

    # Restore this client's saved classification head
    state_dict = state[personal_model_name].to_torch_state_dict()
    net.load_state_dict(state_dict, strict=True)

# def save_personal_params(run_id, client_id, params, round_number, lr, init_epochs):
#     """Save model parameters with metadata in filename."""
#     os.makedirs(f"runs/{run_id}/clientmodels", exist_ok=True)
#     filename = (
#         f"runs/{run_id}/clientmodels/"
#         f"local_model_c{client_id}_r{round_number:03d}_lr{lr:.3f}_init{init_epochs}.pkl"
#     )
#     with open(filename, "wb") as f:
#         pickle.dump(params, f)
#     return filename

# def load_personal_params(run_id, client_id, round_number, lr, init_epochs):
#     """Load a specific version of a client's personal model, or return None if not found."""
#     filename = (
#         f"runs/{run_id}/clientmodels/"
#         f"local_model_c{client_id}_r{round_number:03d}_lr{lr:.3f}_init{init_epochs}.pkl"
#     )
#     if not os.path.exists(filename):
#         return None 
#     with open(filename, "rb") as f:
#         return pickle.load(f)