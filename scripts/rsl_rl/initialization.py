"""Initialize new skill models from trusted project policies or checkpoints."""

import math

import torch


def actor_from_state_dict(state, device="cpu"):
    if any(not name.startswith(("mlp.", "distribution.")) for name in state):
        raise ValueError("Only the project's unnormalized ELU actor is supported")
    weights = sorted(
        (int(name.split(".")[1]), value)
        for name, value in state.items() if name.startswith("mlp.") and name.endswith(".weight")
    )
    if not weights or [index for index, _value in weights] != list(range(0, 2 * len(weights), 2)):
        raise ValueError("Unsupported actor layer layout")
    layers = []
    for position, (_index, weight) in enumerate(weights):
        layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
        if position + 1 < len(weights):
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers)
    actor.load_state_dict({name.removeprefix("mlp."): value for name, value in state.items() if name.startswith("mlp.")})
    return actor.to(device).eval()


def initialize_models(actor, critic, *, checkpoint_path=None, actor_path=None, initial_std=0.25, action_indices=None):
    if (checkpoint_path is None) == (actor_path is None):
        raise ValueError("Choose exactly one initialization source")
    if not math.isfinite(initial_std) or initial_std <= 0.0:
        raise ValueError("initial_std must be positive and finite")
    if action_indices is not None and actor_path is None:
        raise ValueError("Action selection requires an exported actor")
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        critic.load_state_dict(checkpoint["critic_state_dict"])
        source_kind = "actor_and_critic_checkpoint"
    else:
        source = torch.jit.load(str(actor_path), map_location="cpu").state_dict()
        target = actor.state_dict()
        if action_indices is not None:
            last_layer = max(int(name.split(".")[1]) for name in source if name.startswith("mlp.") and name.endswith(".weight"))
            prefix = f"mlp.{last_layer}"
            outputs = source[f"{prefix}.weight"].shape[0]
            if not action_indices or len(set(action_indices)) != len(action_indices) or any(index < 0 or index >= outputs for index in action_indices):
                raise ValueError("Invalid actor action indices")
            for name in (f"{prefix}.weight", f"{prefix}.bias", "distribution.log_std_param"):
                if name in source:
                    source[name] = source[name][action_indices]
        first_weight = "mlp.0.weight"
        if first_weight not in source or first_weight not in target:
            raise ValueError("Expected the project's exported MLP actor")
        source_inputs = source[first_weight].shape[1]
        if target[first_weight].shape[1] < source_inputs:
            raise ValueError("New actor cannot drop pretrained input features")
        for name, value in source.items():
            if name not in target:
                raise ValueError(f"Unexpected exported actor tensor: {name}")
            if name == first_weight and target[name].shape[0] == value.shape[0]:
                target[name] = torch.zeros_like(target[name])
                target[name][:, :source_inputs] = value.to(target[name].device)
            elif target[name].shape == value.shape:
                target[name] = value
            else:
                raise ValueError(f"Actor tensor shape differs: {name}")
        actor.load_state_dict(target)
        source_kind = "exported_actor_with_input_prefix"
    with torch.no_grad():
        actor.distribution.log_std_param.fill_(math.log(initial_std))
    return {"source": source_kind, "input_features": actor.state_dict()["mlp.0.weight"].shape[1], "initial_std": initial_std, "action_indices": action_indices}