"""Client-side algorithm state for GHBM-family methods."""

from dataclasses import dataclass

from flwr.app import ArrayRecord, Context, Message

from ghbm.config import AlgorithmName
from ghbm.utils import StateDict, clone_state_dict

ANCHOR_MODEL_KEY = "anchor_model"


@dataclass(frozen=True)
class TrainingModifiers:
    """Per-round modifiers applied by the local trainer."""

    server_momentum: StateDict | None = None
    anchor_model: StateDict | None = None
    anchor_scale: float = 0.0


def prepare_training_modifiers(
    context: Context,
    algorithm_name: AlgorithmName,
    fraction_train: float,
    current_global_model: StateDict,
    message: Message,
) -> TrainingModifiers:
    """Resolve the client-side correction state for one training round."""
    if algorithm_name is AlgorithmName.FEDAVG:
        return TrainingModifiers()

    if algorithm_name is AlgorithmName.GHBM:
        return TrainingModifiers(
            server_momentum=_resolve_server_momentum(message),
        )

    if algorithm_name is AlgorithmName.LOCAL_GHBM:
        return TrainingModifiers(
            server_momentum=_prepare_localghbm_momentum(
                context=context,
                fraction_train=fraction_train,
                current_global_model=current_global_model,
            ),
        )

    if algorithm_name is AlgorithmName.FED_HBM:
        return TrainingModifiers(
            anchor_model=_resolve_persisted_anchor(context),
            anchor_scale=fraction_train,
        )

    raise ValueError(f"Unsupported algorithm-name: {algorithm_name}")


def finalize_training_state(
    context: Context,
    algorithm_name: AlgorithmName,
    trained_model_state: StateDict,
) -> None:
    """Persist any client-local state needed by future participations."""
    if algorithm_name is AlgorithmName.FED_HBM:
        context.state.array_records[ANCHOR_MODEL_KEY] = ArrayRecord(
            torch_state_dict=clone_state_dict(trained_model_state)
        )


def _resolve_server_momentum(message: Message) -> StateDict | None:
    """Read the server-sent GHBM momentum from the incoming message."""
    momentum_record = message.content.array_records.get("server_momentum")
    if momentum_record is None:
        return None
    return momentum_record.to_torch_state_dict()


def _prepare_localghbm_momentum(
    context: Context,
    fraction_train: float,
    current_global_model: StateDict,
) -> StateDict:
    """Compute LocalGHBM's fixed round-level correction."""
    anchor_state = _resolve_persisted_anchor(context)
    context.state.array_records[ANCHOR_MODEL_KEY] = ArrayRecord(
        torch_state_dict=clone_state_dict(current_global_model)
    )
    if anchor_state is None:
        return type(current_global_model)(
            (name, tensor.new_zeros(tensor.shape))
            for name, tensor in current_global_model.items()
        )
    return type(current_global_model)(
        (
            name,
            (anchor_state[name] - current_global_model[name]) * fraction_train,
        )
        for name in current_global_model
    )


def _resolve_persisted_anchor(context: Context) -> StateDict | None:
    """Load the client-local anchor model from Flower context state."""
    anchor_record = context.state.array_records.get(ANCHOR_MODEL_KEY)
    if anchor_record is None:
        return None
    return anchor_record.to_torch_state_dict()
