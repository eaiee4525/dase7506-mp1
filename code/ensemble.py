import torch
import math
from torch import nn
from torch.nn import functional as F
from student import StudentGPT


class ProbabilityEnsemble(nn.Module):
    def __init__(self, config):
        super().__init__()

        component_configs = config["components"]
        weights = torch.tensor(config["weights"], dtype=torch.float32)

        if len(component_configs) != len(weights):
            raise ValueError(
                "The number of component configs and weights must match."
            )

        if len(component_configs) < 2:
            raise ValueError("An ensemble needs at least two models.")

        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Ensemble weights must be finite and positive.")

        for component_config in component_configs:
            if component_config["vocab"] != config["vocab"]:
                raise ValueError("All models must use the same vocabulary.")

            if component_config["context"] != config["context"]:
                raise ValueError("All models must use the same context.")

        weights = weights / weights.sum()
        temperature = float(config.get("temperature", 1.0))

        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("Temperature must be finite and positive.")

        self.context = config["context"]
        self.temperature = temperature
        self.models = nn.ModuleList([
            StudentGPT(component_config)
            for component_config in component_configs
        ])
        self.register_buffer("log_weights", weights.log())

    def predict_log_probs(self, ids):
        combined = None

        for index, model in enumerate(self.models):
            weighted_log_probs = (
                model.predict_log_probs(ids)
                + self.log_weights[index]
            )

            if combined is None:
                combined = weighted_log_probs
            else:
                combined = torch.logaddexp(
                    combined,
                    weighted_log_probs,
                )

        return F.log_softmax(
            combined / self.temperature,
            dim=-1,
        )

    def forward(self, ids):
        return self.predict_log_probs(ids)


def build_model(config):
    return ProbabilityEnsemble(config)