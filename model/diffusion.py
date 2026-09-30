"""Flow matching sampler used by the induced IGD rotation decoder."""

import torch
import torch.nn.functional as F
from torch import nn
from einops import reduce


class FlowMatching(nn.Module):
    """Independent conditional flow matching with Euler sampling."""

    def __init__(self, condition_mask, sigma=0.0, denosing_steps=1):
        super().__init__()
        self.sigma = sigma
        self.denosing_steps = denosing_steps
        self.register_buffer("condition_mask", torch.as_tensor(condition_mask, dtype=torch.bool))

    def loss_fn(self, model, model_input):
        data = model_input["data"]
        context = model_input["context"]
        source = torch.randn_like(data)
        source[..., self.condition_mask] = data[..., self.condition_mask]
        time = torch.rand(data.shape[0], device=data.device, dtype=data.dtype)
        interpolation = time.reshape(-1, *([1] * (data.ndim - 1)))
        sample = interpolation * data + (1.0 - interpolation) * source
        if self.sigma:
            sample = sample + self.sigma * torch.randn_like(sample)
        sample[..., self.condition_mask] = data[..., self.condition_mask]
        target = data - source
        prediction = model(sample, context, time)
        loss = F.mse_loss(prediction[..., ~self.condition_mask], target[..., ~self.condition_mask], reduction="none")
        return reduce(loss, "b ... -> b", "mean")

    def sample_data(self, source, context, model):
        trajectory = source.clone()
        for step in range(self.denosing_steps):
            trajectory[..., self.condition_mask] = source[..., self.condition_mask]
            time = torch.full(
                (trajectory.shape[0],), step / self.denosing_steps,
                device=trajectory.device, dtype=trajectory.dtype,
            )
            trajectory = trajectory + model(trajectory, context, time) / self.denosing_steps
        trajectory[..., self.condition_mask] = source[..., self.condition_mask]
        return trajectory
