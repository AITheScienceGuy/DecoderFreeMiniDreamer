#!/usr/bin/env python

import os
import io
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.distributions import Categorical, OneHotCategorical
from torch.amp import GradScaler, autocast 

from utils import make_vector_env, get_device
from models import WorldModel, ActorCritic, ACTION_DIM, EMBED_DIM, STOCH_DIM, CLASS_DIM
from buffer import ReplayBuffer

NUM_ENVS = 8                
TOTAL_ITERATIONS = 2000     
STEPS_PER_ITER = 1000       
WM_EPOCHS = 1               
AGENT_EPOCHS = 25           
BATCH_SIZE = 16             
SEQ_LEN = 50                
BURN_IN_STEPS = 5  
IMAGINE_HORIZON = 25         
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_SCALE = 5e-2         

W_BARLOW = 1.0  
W_REWARD = 10.0
W_DONE = 1.0
BARLOW_LAMBDA = 5e-3 

EPS_START       = 1.0
EPS_END         = 0.2
EPS_DECAY_ITERS = 1500 
IMAGINATION_TEMPERATURE = 1.5

writer = SummaryWriter(log_dir="runs/SpaceInvaders_Barlow")
global_step_wm = 0
global_step_agent = 0
global_step_env = 0

def get_epsilon(iteration,
                eps_start=EPS_START,
                eps_end=EPS_END,
                decay_iters=EPS_DECAY_ITERS):
    # linear decay from eps_start to eps_end over decay_iters
    frac = min(1.0, iteration / decay_iters)
    return eps_start - frac * (eps_start - eps_end)

def get_kl_weight(iteration):
    if iteration < 50: return 0.0
    return min(0.1, 0.01 + (iteration - 50) * 0.01)

# --- VISUALIZATION HELPERS ---
def plot_correlation_matrix(z1, z2):
    """
    Generates a heatmap of the cross-correlation matrix between Prior and Posterior.
    Executed on CPU to save VRAM.
    """
    with torch.no_grad():
        # Normalize
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)
        
        # Compute Matrix (c is D x D)
        if z1.shape[1] > 64:
            z1 = z1[:, :64]
            z2 = z2[:, :64]
            
        c = torch.matmul(z1.T, z2) / z1.shape[0]
        c_np = c.detach().cpu().numpy()
        
        # Plot
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(c_np, cmap='viridis', vmin=-1, vmax=1)
        # plt.colorbar(im) # Colorbar takes space, removing for small thumbnails
        ax.set_title("Barlow Correlation (Top 64)")
        ax.axis('off')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        image = Image.open(buf)
        return np.array(image)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def compute_barlow_loss(z1, z2):
    z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
    z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)
    
    N = z1.shape[0]
    c = torch.matmul(z1_norm.T, z2_norm) / N

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    
    loss = on_diag + BARLOW_LAMBDA * off_diag
    return loss, on_diag, off_diag

def collect_data(buffer, steps, wm=None, agent=None, epsilon=1.0, device="cpu"):
    global global_step_env
    envs = make_vector_env(NUM_ENVS)
    obs, _ = envs.reset()
    
    prev_h = torch.zeros(NUM_ENVS, EMBED_DIM, device=device)
    prev_z_flat = torch.zeros(NUM_ENVS, STOCH_DIM * CLASS_DIM, device=device)

    episode_returns = np.zeros(NUM_ENVS, dtype=np.float32)
    episode_lengths = np.zeros(NUM_ENVS, dtype=np.int32)
    
    if wm: wm.eval()
    if agent: agent.eval()
    
    use_amp = (device.type == "cuda")
    loops = steps // NUM_ENVS
    
    for _ in range(loops):
        if np.random.rand() < epsilon or agent is None:
            actions = envs.action_space.sample() 
        else:
            with torch.no_grad():
                with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp): 
                    obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
                    embed = wm.encoder(obs_tensor)
                    
                    action_one_hot = F.one_hot(torch.tensor(prev_actions, device=device), ACTION_DIM).float() if 'prev_actions' in locals() else torch.zeros(NUM_ENVS, ACTION_DIM, device=device)
                    
                    h, logits = wm.rssm.step(prev_h, prev_z_flat, action_one_hot, embed)
                    z, z_flat = wm.get_stochastic_state(logits, hard=True)
                    feat = wm.rssm.get_feat(h, z_flat)
                    
                    action_probs = agent.get_action(feat, temperature=1.0)
                    dist = Categorical(probs=action_probs)
                    actions = dist.sample().cpu().numpy()
                    
                    prev_h = h
                    prev_z_flat = z_flat
        
        prev_actions = actions 
        next_obs, rewards, terms, truncs, _ = envs.step(actions)
        dones = np.logical_or(terms, truncs).astype(np.float32)

        episode_returns += rewards
        episode_lengths += 1
        
        if np.any(dones):
            mask = torch.tensor(1.0 - dones, device=device).unsqueeze(1)
            prev_h = prev_h * mask
            prev_z_flat = prev_z_flat * mask

            for env_idx, done in enumerate(dones):
                if done:
                    writer.add_scalar(
                        "Env/Episode_Reward",
                        episode_returns[env_idx],
                        global_step_env,
                    )
                    writer.add_scalar(
                        "Env/Episode_Length",
                        episode_lengths[env_idx],
                        global_step_env,
                    )
                    global_step_env += 1
                    episode_returns[env_idx] = 0.0
                    episode_lengths[env_idx] = 0

        buffer.add_batch(obs, actions, rewards, dones)
        obs = next_obs
    envs.close()

def train_world_model(buffer, model, optimizer, scaler, device, epochs, kl_scale): 
    global global_step_wm
    model.train()
    steps_per_epoch = min(50, (len(buffer) // BATCH_SIZE)) 
    
    use_amp = (device.type == "cuda")
    
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            batch = buffer.sample_sequence(BATCH_SIZE, recent_only_pct=0.5)
            if batch is None: break
            b_obs, b_act, b_rew, b_don = batch
            
            b_obs, b_act = b_obs.to(device), b_act.to(device)
            b_rew, b_don = b_rew.to(device), b_don.to(device)
            
            optimizer.zero_grad() 
            
            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp): 
                B, T, C, H, W = b_obs.shape
                flat_obs = b_obs.view(B*T, C, H, W)
                embeds = model.encoder(flat_obs).view(B, T, -1)
                
                prev_h = torch.zeros(B, EMBED_DIM, device=device)
                prev_z_flat = torch.zeros(B, STOCH_DIM * CLASS_DIM, device=device)
                
                loss_kl = 0; loss_rew = 0; loss_done = 0
                avg_entropy = 0
                
                post_feats_list = []
                prior_feats_list = []
                
                for t in range(T):
                    act = torch.zeros(B, ACTION_DIM, device=device) if t==0 else F.one_hot(b_act[:, t-1], ACTION_DIM).float()
                    
                    # Posterior (Reality)
                    h_post, post_logits = model.rssm.step(prev_h, prev_z_flat, act, embeds[:, t])
                    z_post, z_flat_post = model.get_stochastic_state(post_logits)
                    
                    # Prior (Prediction)
                    _, prior_logits = model.rssm.step(prev_h, prev_z_flat, act, embed=None)
                    z_prior, z_flat_prior = model.get_stochastic_state(prior_logits)
                    
                    feat_post = model.rssm.get_feat(h_post, z_flat_post)
                    feat_prior = model.rssm.get_feat(h_post, z_flat_prior)
                    
                    post_feats_list.append(feat_post)
                    prior_feats_list.append(feat_prior)

                    q = OneHotCategorical(logits=post_logits)
                    p = OneHotCategorical(logits=prior_logits)
                    kl = torch.distributions.kl_divergence(q, p)
                    loss_kl += torch.maximum(kl, torch.tensor(1.0, device=device)).mean()
                    
                    # Track Entropy to detect collapse
                    avg_entropy += q.entropy().mean()

                    pred_r = model.reward_head(feat_post)
                    pred_d = model.done_head(feat_post)
                    loss_rew += F.mse_loss(pred_r, b_rew[:, t])
                    loss_done += F.binary_cross_entropy_with_logits(pred_d, b_don[:, t])
                    
                    prev_h = h_post
                    prev_z_flat = z_flat_post
                
                # BARLOW LOSS
                all_post = torch.cat(post_feats_list, dim=0) 
                all_prior = torch.cat(prior_feats_list, dim=0)
                
                proj_post = model.projector(all_post)
                proj_prior = model.projector(all_prior)
                
                loss_barlow, on_diag, off_diag = compute_barlow_loss(proj_prior, proj_post)

                loss_kl /= T; loss_rew /= T; loss_done /= T
                avg_entropy /= T
                
                loss = (W_BARLOW * loss_barlow + kl_scale * loss_kl + W_REWARD * loss_rew + W_DONE * loss_done)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            scaler.step(optimizer)
            scaler.update()
            
            global_step_wm += 1
            if global_step_wm % 500 == 0:
                print(f"WM Step {global_step_wm} | Loss: {loss.item():.3f} | Diag: {on_diag.item():.3f} | Off: {off_diag.item():.3f}")
                
                # --- DETAILED LOGGING ---
                writer.add_scalar("WM/Total_Loss", loss.item(), global_step_wm)
                writer.add_scalar("WM/Barlow_OnDiag", on_diag.item(), global_step_wm)
                writer.add_scalar("WM/Barlow_OffDiag", off_diag.item(), global_step_wm)
                writer.add_scalar("WM/KL_Divergence", loss_kl.item(), global_step_wm)
                writer.add_scalar("WM/Latent_Entropy", avg_entropy.item(), global_step_wm)
                writer.add_scalar("WM/Reward_MSE", loss_rew.item(), global_step_wm)
                writer.add_scalar("WM/Done_BCE", loss_done.item(), global_step_wm)
                writer.add_scalar("WM/Done_Pred_Mean", torch.sigmoid(pred_d).mean().item(), global_step_wm)
                
                proj_std = proj_post.std(0).mean()
                writer.add_scalar("WM/Projector_Std", proj_std.item(), global_step_wm)
                
                # Visual Heatmap
                if global_step_wm % 1000 == 0:
                    img = plot_correlation_matrix(proj_prior, proj_post)
                    writer.add_image("WM/Correlation_Matrix", img, global_step_wm, dataformats='HWC')

    torch.save(model.state_dict(), "world_model.pth")

def compute_lambda_returns(rewards, values, dones, gamma=0.99, lambda_=0.95):
    returns = []
    last_lambda_ret = values[-1] 
    for t in reversed(range(len(rewards))):
        r_t = rewards[t]
        v_next = values[t+1]
        d_t = dones[t]
        one_step = r_t + gamma * (1.0 - d_t) * v_next
        lambda_ret = (1 - lambda_) * one_step + lambda_ * (r_t + gamma * (1.0 - d_t) * last_lambda_ret)
        returns.insert(0, lambda_ret)
        last_lambda_ret = lambda_ret
    return torch.stack(returns)

def train_policy_dreamer(buffer, wm, agent, optimizer, scaler, device, epochs): 
    global global_step_agent
    wm.eval()
    agent.train()
    wm.requires_grad_(False) 
    
    use_amp = (device.type == "cuda")

    for _ in range(epochs):
        batch = buffer.sample_sequence(BATCH_SIZE, recent_only_pct=0.5)
        if batch is None: continue
        b_obs, b_act, _, _ = batch 
        
        context_obs = b_obs[:, :BURN_IN_STEPS].to(device)
        context_act = b_act[:, :BURN_IN_STEPS].to(device)
        
        optimizer.zero_grad()

        with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp): 
            with torch.no_grad():
                B, T_ctx, C, H, W = context_obs.shape
                flat_obs = context_obs.view(B*T_ctx, C, H, W)
                embeds = wm.encoder(flat_obs).view(B, T_ctx, -1)
                
                h = torch.zeros(B, EMBED_DIM, device=device)
                z_flat = torch.zeros(B, STOCH_DIM * CLASS_DIM, device=device)
                
                for t in range(T_ctx):
                    act = torch.zeros(B, ACTION_DIM, device=device) if t==0 else F.one_hot(context_act[:, t-1], ACTION_DIM).float()
                    h, logits = wm.rssm.step(h, z_flat, act, embeds[:, t])
                    z, z_flat = wm.get_stochastic_state(logits, hard=True)
            
            list_rewards, list_values, list_dones = [], [], []
            list_log_probs, list_entropies = [], []
            
            curr_h, curr_z_flat = h, z_flat
            
            feat_start = wm.rssm.get_feat(curr_h, curr_z_flat).detach()
            list_values.append(agent.get_value(feat_start))
            
            for t in range(IMAGINE_HORIZON):
                feat = wm.rssm.get_feat(curr_h, curr_z_flat).detach() 
                logits_act = agent.get_action_logits(feat) / IMAGINATION_TEMPERATURE
                dist = OneHotCategorical(logits=logits_act)
                action = dist.sample() 
                action_idx = action.argmax(dim=-1)
                
                log_prob = dist.log_prob(action)
                list_log_probs.append(log_prob)
                list_entropies.append(dist.entropy())
                
                curr_h, logits_next = wm.rssm.step(curr_h, curr_z_flat, action, embed=None)
                z_next, z_flat_next = wm.get_stochastic_state(logits_next, hard=True)
                
                feat_next = wm.rssm.get_feat(curr_h, z_flat_next)
                
                pred_rew = wm.reward_head(feat_next)
                pred_done = wm.done_head(feat_next) 
                pred_val = agent.get_value(feat_next)
                
                list_rewards.append(pred_rew)
                list_values.append(pred_val)
                done_prob = torch.sigmoid(pred_done)
                list_dones.append(done_prob.detach())
                
                curr_z_flat = z_flat_next
                
            rewards = torch.stack(list_rewards) 
            values = torch.stack(list_values)   
            dones = torch.stack(list_dones)     
            log_probs = torch.stack(list_log_probs) 
            entropies = torch.stack(list_entropies) 
            
            lambda_targets = compute_lambda_returns(rewards, values, dones, GAMMA, LAMBDA).detach()
            critic_loss = 0.5 * F.mse_loss(values[:-1], lambda_targets)
            
            advantage = (lambda_targets - values[:-1]).detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            advantage = advantage.clamp(-5.0, 5.0)
            
            actor_loss = - (log_probs * advantage.squeeze(-1)).mean() 
            entropy_loss = - entropies.mean()
            
            total_loss = actor_loss + critic_loss + (ENTROPY_SCALE * entropy_loss)
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 100.0)
        scaler.step(optimizer)
        scaler.update()
        
        global_step_agent += 1
        
        if global_step_agent % 200 == 0:
             writer.add_scalar("Agent/Actor_Loss", actor_loss.item(), global_step_agent)
             writer.add_scalar("Agent/Critic_Loss", critic_loss.item(), global_step_agent)
             writer.add_scalar("Agent/Entropy", entropies.mean().item(), global_step_agent)
             writer.add_scalar("Agent/Value_Mean", values.mean().item(), global_step_agent)
             writer.add_histogram("Agent/Action_Index", action_idx.cpu(), global_step_agent)

    wm.requires_grad_(True)
    torch.save(agent.state_dict(), "policy.pth")

def main():
    device = get_device()
    print(f"Using Device: {device}")
    buffer = ReplayBuffer(capacity=5000, num_envs=NUM_ENVS, seq_len=SEQ_LEN)
    
    wm = WorldModel().to(device)
    agent = ActorCritic().to(device)
    
    opt_wm = optim.Adam(wm.parameters(), lr=2e-4) 
    opt_agent = optim.Adam(agent.parameters(), lr=1e-4)
    
    use_cuda = (device.type == "cuda")
    scaler_wm = GradScaler(enabled=use_cuda)
    scaler_agent = GradScaler(enabled=use_cuda)
    
    collect_data(buffer, steps=5000, device=device) 
    
    for i in tqdm(range(TOTAL_ITERATIONS), desc="Training"):
        epsilon = get_epsilon(i)
        
        collect_data(buffer, STEPS_PER_ITER, wm, agent, epsilon, device)
        train_world_model(buffer, wm, opt_wm, scaler_wm, device, WM_EPOCHS, get_kl_weight(i))
        train_policy_dreamer(buffer, wm, agent, opt_agent, scaler_agent, device, AGENT_EPOCHS)

if __name__ == "__main__":
    main()
