import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput, randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from typing import List, Optional, Tuple, Union
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import time
from einops import rearrange, reduce
try:
    from torchcfm.optimal_transport import OTPlanSampler
    from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
except:
    pass


class Diffusion(nn.Module):
    def __init__(self, schedulers="DDPM", condition_mask=[0,0,0,0,0,0], beta_schedule='squaredcos_cap_v2', prediction_type='epsilon', num_inference_steps=100):
        super().__init__()
        self.schedulers = schedulers
        if schedulers == "DDPM":
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule=beta_schedule, prediction_type=prediction_type)
        elif schedulers == "DPM":
            self.noise_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=100, beta_schedule=beta_schedule, prediction_type=prediction_type)
        elif schedulers == "DDIM":
            self.noise_scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule=beta_schedule, prediction_type=prediction_type)
        else:
            raise NotImplementedError("not implemented.")
        self.num_inference_steps = num_inference_steps
        condition_mask = torch.FloatTensor(condition_mask).bool()
        self.viz_trajectory = []

        self.register_buffer('condition_mask', condition_mask)
    
    
    def loss_fn(self, model, model_input):
        """
        data: torch.tensor(bs, ns, dim)
        
        """
        data = model_input['data']
        c = model_input['context']

        batch_size, sample_num = data.shape[0], data.shape[1]
   
        noise = torch.randn(data.shape, device=data.device) # bs, ns, dim
        
        #### ot plan ####
        # with torch.no_grad():
        #     cost_matrix = compute_cost_matrix(data, noise)
        #     transport_plan = sinkhorn_algorithm(cost_matrix)
        #     re_idx = torch.argmax(transport_plan, dim=1).unsqueeze(2).expand(-1, -1, data.shape[2])
        #     noise = torch.gather(noise, dim=1, index=re_idx)
        #### ot plan ####

        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        timesteps = torch.randint(
            1, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=data.device
        ).long()

        noisy_data = self.noise_scheduler.add_noise(
            data, noise, timesteps)
        
        ##### debug #####
        anchor = torch.linspace(0, sample_num-1, sample_num).unsqueeze(0).repeat(batch_size, 1).to(data.device)
        new_noise = noise.clone()
        while True:
            noise_data_ = noisy_data.clone()
            noise_data_[..., self.condition_mask] = data[..., self.condition_mask]
            min_dist = (noise_data_[:,:,None,:] - data[:,None,:,:]).pow(2).sum(-1).pow(0.5) # (bs, ns, ns)
            idx = torch.argmin(min_dist, dim=-1)
            renoise_mask = (idx != anchor)
            if renoise_mask.sum()< 10:
                break
            # re-generate noise
            new_noise = torch.randn(data.shape, device=data.device)
            # new_noise = new_noise + (data-new_noise) * 0.1
            new_noisy_data = self.noise_scheduler.add_noise(
                data, new_noise, timesteps)
            noise[renoise_mask] = new_noise[renoise_mask]
            noisy_data[renoise_mask] = new_noisy_data[renoise_mask]
        ##### debug #####
        
        
        # compute loss mask
        loss_mask = ~self.condition_mask

        # apply conditioning
        noisy_data[..., self.condition_mask] = data[..., self.condition_mask]
        pred = model(noisy_data, c, timesteps)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = data
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss[..., loss_mask]
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean(-1)
        
        return loss

    def sample_data(self, data, c, model):
        batch_size = data.size(0)
        sample_num = data.size(1)
        dim = data.size(2)
        condition_data = data

        scheduler = self.noise_scheduler
        
        data = torch.randn((batch_size, sample_num, dim), device=data.device)

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            data[..., self.condition_mask] = condition_data[..., self.condition_mask]
            # 2. predict model output
            model_output = model(data, c, t[None].to(data.device).expand(data.shape[0]))

            # 3. compute previous image: x_t -> x_t-1
            data = scheduler.step(
                model_output, t, data, 
                generator=None,
                ).prev_sample
            data = data.reshape(batch_size, sample_num, -1)
            # viz_traj.append(data[0,0,3:])
        
        # finally make sure conditioning is enforced
        data[..., self.condition_mask] = condition_data[..., self.condition_mask]
        # self.viz_trajectory.append(torch.stack(viz_traj, dim=0).detach().cpu().numpy())

        return data


class FlowMatching(nn.Module):
    def __init__(self,condition_mask, sigma=0.0, denosing_steps=1, ot=True):
        super().__init__()
        self.ot_sampler = OTPlanSampler(method="exact")
        condition_mask = torch.FloatTensor(condition_mask).bool()
        self.viz_trajectory = []

        self.register_buffer('condition_mask', condition_mask)
        self.FM = ConditionalFlowMatcher(sigma=sigma)
        self.denosing_steps = denosing_steps
    
    def ot_sample_plan(self, x0, x1):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return torch.stack([x0, x1], dim=0)
        
    
    def loss_fn(self, model, model_input):
        data = model_input['data']
        c = model_input['context']
        
        batch_size = data.size(0)
        sample_num = data.size(1)
        
        x0 = torch.randn(data.shape, device=data.device) # (bs, num_samples, 7)
        x0[...,self.condition_mask] = data[...,self.condition_mask]
        
        anchor = torch.linspace(0, sample_num-1, sample_num).unsqueeze(0).repeat(batch_size, 1).to(data.device)
        while True:
            x0_ = x0.clone()
            min_dist = (x0_[:,:,None,:] - data[:,None,:,:]).pow(2).sum(-1).pow(0.5) # (bs, ns, ns)
            idx = torch.argmin(min_dist, dim=-1)
            renoise_mask = (idx != anchor)
            if renoise_mask.sum()< 10:
                break
            # re-generate noise
            new_x0 = torch.randn(data.shape, device=data.device) # (bs, num_samples, 7)
            new_x0[...,self.condition_mask] = data[...,self.condition_mask]
            # new_noise = new_noise + (data-new_noise) * 0.1
            x0[renoise_mask] = new_x0[renoise_mask]
        
        timestep, xt, ut = self.FM.sample_location_and_conditional_flow(x0, data)
        
        loss_mask = ~self.condition_mask
        
        xt[...,self.condition_mask] = data[...,self.condition_mask]
        vt = model(xt, c, timestep)
        loss = F.mse_loss(vt, ut, reduction='none')
        loss = loss[..., loss_mask]
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean(-1)
        
        return loss
    
    def sample_data(self, x0, c, model):
        batch_size = x0.size(0)
        sample_num = x0.size(1)
        
        
        condition_data = x0
        traj = x0
        
        for i in range(self.denosing_steps):
            traj[..., self.condition_mask] = condition_data[..., self.condition_mask]
            timestep = torch.tensor([i / self.denosing_steps]).to(x0.device)
            vt =  model(traj, c, timestep.expand(traj.shape[0]))# t[None].to(data.device).expand(data.shape[0])
            traj = (vt * 1 / self.denosing_steps + traj)
        traj[..., self.condition_mask] = condition_data[..., self.condition_mask]
        # traj[...,3:] = nn.functional.normalize(traj[...,3:], dim=2)
        return traj
    
    
def compute_cost_matrix(x0, x1):
    # x0, x1: 形状为 (bs, ns, dim) 的张量
    # 返回成本矩阵 cost_matrix: 形状为 (bs, ns, ns)
    bs, ns, dim = x0.shape
    x0_expanded = x0.unsqueeze(2)  # (bs, ns, 1, dim)
    x1_expanded = x1.unsqueeze(1)  # (bs, 1, ns, dim)
    cost_matrix = torch.norm(x0_expanded - x1_expanded, p=2, dim=3)  # (bs, ns, ns)
    return cost_matrix

def sinkhorn_algorithm(cost_matrix, epsilon=0.1, max_iter=100, tol=1e-9):
    bs, ns, _ = cost_matrix.shape
    mu = torch.full((bs, ns), 1.0 / ns, dtype=torch.float32, device=cost_matrix.device)
    nu = torch.full((bs, ns), 1.0 / ns, dtype=torch.float32, device=cost_matrix.device)

    K = torch.exp(-cost_matrix / epsilon)  # (bs, ns, ns)

    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    for _ in range(max_iter):
        u_prev = u.clone()
        u = mu / (K @ v.unsqueeze(2)).squeeze(2)  # (bs, ns)
        v = nu / (K.transpose(1, 2) @ u.unsqueeze(2)).squeeze(2)  # (bs, ns)

        if torch.max(torch.abs(u - u_prev)) < tol:
            break

    transport_plan = u.unsqueeze(2) * K * v.unsqueeze(1)  # (bs, ns, ns)
    return transport_plan

if __name__ == "__main__":
    # 示例
    bs, ns, dim = 2, 100, 3
    x0 = torch.randn(bs, ns, dim).cuda()
    x1 = torch.randn(bs, ns, dim).cuda()

    cost_matrix = compute_cost_matrix(x0, x1)
    transport_plan = sinkhorn_algorithm(cost_matrix)
    re_idx = torch.argmax(transport_plan, dim=1).unsqueeze(2).expand(-1, -1, x1.shape[2])
    x1_new = torch.gather(x1, dim=1, index=re_idx)
    # x1_new = x1[re_idx]
    print(re_idx)
    print()
    # transport_plan 是形状为 (bs, ns, ns) 的张量，表示每个批次的传输计划

