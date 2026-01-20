#!/usr/bin/env python

import os
import io
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.distributions import Categorical, OneHotCategorical
from torch.amp import GradScaler, autocast 

from utils import make_vector_env, get_device, make_single_env
from models import WorldModel, ActorCritic, EMBED_DIM, STOCH_DIM, CLASS_DIM, TwoHotDist, symlog, symexp, RetNorm
from buffer import ReplayBuffer

# --- ENVIRONMENT CONFIGURATION ---
ENV_NAME = "ALE/MsPacman-v5"

# Training Hyperparameters
NUM_ENVS = 8                
TOTAL_ITERATIONS = 4000     
STEPS_PER_ITER = 1000       
WM_EPOCHS = 32               
AGENT_EPOCHS = 32           
BATCH_SIZE = 16             
SEQ_LEN = 64                
BURN_IN_STEPS = 5  
IMAGINE_HORIZON = 15        
GAMMA = 0.997
LAMBDA = 0.95
ENTROPY_SCALE = 1e-2        
W_BARLOW = 1.0  
W_REWARD = 1.0              
W_DONE = 1.0
BARLOW_LAMBDA = 5e-4 
BARLOW_SCALE = 0.05  

WARMUP_STEPS = 20000 

ACTOR_GRAD_CLIP = 100.0

# DreamerV3 specific constants
FREE_NATS = 1.0
BETA_DYN = 0.5
BETA_REP = 0.1
LATENT_UNIMIX = 0.01

# Initialize TensorBoard Writer
writer = None
global_step_wm = 0
global_step_agent = 0
global_step_env = 0
global_step_eval = 0 

def save_model(model, path):
    state_dict = model.state_dict()
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    torch.save(clean_state_dict, path)

def compute_barlow_loss(z1, z2):
    # z1: prediction (k), z2: target (e)
    z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
    z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)
    c = torch.matmul(z1_norm.T, z2_norm) / z1.shape[0]
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum())
    return on_diag + BARLOW_LAMBDA * off_diag, on_diag, off_diag

def unimix_probs(logits, mix=0.01):
    probs = torch.softmax(logits, dim=-1)
    K = probs.shape[-1]
    return (1.0 - mix) * probs + mix / K

def onehot_dist_from_logits(logits, mix=0.01):
    return torch.distributions.OneHotCategorical(probs=unimix_probs(logits, mix))

def twohot_loss_per_sample(logits, targets, low=-20.0, high=20.0, num_buckets=255):
    # logits: (B, num_buckets)
    # targets: (B,)
    B = logits.shape[0]
    bin_size = (high - low) / (num_buckets - 1)
    
    t = targets.clamp(low, high)
    pos = (t - low) / bin_size
    idx0 = pos.floor().long().clamp(0, num_buckets - 1)
    idx1 = (idx0 + 1).clamp(0, num_buckets - 1)
    w1 = (pos - idx0.float()).clamp(0.0, 1.0)
    w0 = 1.0 - w1

    logp = F.log_softmax(logits, dim=-1)
    lp0 = logp[torch.arange(B, device=logits.device), idx0]
    lp1 = logp[torch.arange(B, device=logits.device), idx1]
    return -(w0 * lp0 + w1 * lp1)

def check_model_rollout(wm, buffer, action_dim, device, horizon=10):
    wm.eval()
    batch = buffer.sample_sequence(BATCH_SIZE, recent_only_pct=0.3)
    if batch is None: return
    
    # UNPACK 6 ITEMS
    b_obs, b_next_obs, b_act, b_rew, b_bound, b_term = [x.to(device) for x in batch]
    
    if b_obs.shape[1] < BURN_IN_STEPS + horizon + 1: return

    with torch.inference_mode():
        B = b_obs.shape[0]
        h = torch.zeros(B, EMBED_DIM, device=device)
        z_f = torch.zeros(B, STOCH_DIM * CLASS_DIM, device=device)
        
        context_obs = b_obs[:, :BURN_IN_STEPS] 
        embeds = wm.encoder(context_obs.reshape(-1, 4, 64, 64)).view(B, BURN_IN_STEPS, -1)
        
        for t in range(BURN_IN_STEPS):
            if t == 0:
                act = torch.zeros(B, action_dim, device=device)
            else:
                mask = (1.0 - b_bound[:, t-1]).float()
                h   = h   * mask
                z_f = z_f * mask
                act = F.one_hot(b_act[:, t-1], action_dim).float() * mask
        
            h, logits = wm.rssm.step(h, z_f, act, embeds[:, t])
            _, z_f = wm.get_stochastic_state(logits, hard=True)

        rew_errs = []
        done_accs = []
        
        for t in range(horizon):
            idx = BURN_IN_STEPS + t
        
            mask = (1.0 - b_bound[:, idx-1]).float()
            h   = h   * mask
            z_f = z_f * mask
        
            real_act = F.one_hot(b_act[:, idx-1], action_dim).float() * mask
        
            h, prior_logits = wm.rssm.step(h, z_f, real_act, embed=None)
            _, z_f = wm.get_stochastic_state(prior_logits, hard=True)
            feat = wm.rssm.get_feat(h, z_f)
        
            pred_rew = TwoHotDist(logits=wm.reward_head(feat)).mean_linear()
            pred_done_prob = torch.sigmoid(wm.done_head(feat))
        
            real_rew = b_rew[:, idx-1]
            real_done = b_term[:, idx-1]
        
            denom = mask.sum().clamp(min=1.0)
            rew_err  = ((pred_rew - real_rew).abs() * mask).sum() / denom
            done_err = ((pred_done_prob - real_done).abs() * mask).sum() / denom
        
            rew_errs.append(rew_err.item())
            done_accs.append(1.0 - done_err.item())

        step_idx = global_step_wm
        writer.add_scalar("sanity/openloop_rew_err_step1", rew_errs[0], step_idx)
        writer.add_scalar("sanity/openloop_rew_err_step5", rew_errs[min(4, horizon-1)], step_idx)
        writer.add_scalar("sanity/openloop_done_acc_step5", done_accs[min(4, horizon-1)], step_idx)

def evaluate_agent(wm, agent, action_dim, device, num_episodes=10):
    global global_step_eval
    wm.eval(); agent.eval()
    
    episode_returns = []
    
    for _ in range(num_episodes):
        env = make_single_env(ENV_NAME, mode='eval')
        obs, info = env.reset()
        current_lives = info["lives"]
        
        h = torch.zeros(1, EMBED_DIM, device=device)
        z_flat = torch.zeros(1, STOCH_DIM * CLASS_DIM, device=device)
        prev_action = torch.zeros(1, device=device, dtype=torch.long)
        
        total_reward = 0
        done = False
        steps_since_life_start = 0
        
        with torch.inference_mode():
            while not done:
                obs_t = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                act_oh = F.one_hot(prev_action, action_dim).float()
                
                embed = wm.encoder(obs_t)
                h, logits = wm.rssm.step(h, z_flat, act_oh, embed)
                _, z_flat = wm.get_stochastic_state(logits, hard=True)
                feat = wm.rssm.get_feat(h, z_flat)
                
                probs = agent.get_action(feat, hard=False) 
                action = Categorical(probs=probs).sample().item() 

                next_obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
                
                new_lives = info["lives"]
                life_loss = (new_lives < current_lives) and (new_lives > 0)
                
                steps_since_life_start += 1
                
                if life_loss:
                    h = torch.zeros(1, EMBED_DIM, device=device)
                    z_flat = torch.zeros(1, STOCH_DIM * CLASS_DIM, device=device)
                    prev_action = torch.zeros(1, device=device, dtype=torch.long)
                    steps_since_life_start = 0
                else:
                    prev_action = torch.tensor([action], device=device)
                    
                current_lives = new_lives
                obs = next_obs
                
                if term or trunc:
                    done = True
        
        env.close()
        episode_returns.append(total_reward)
    
    mean_ret = np.mean(episode_returns)
    std_ret = np.std(episode_returns)
    
    writer.add_scalar("eval/mean_return", mean_ret, global_step_eval)
    writer.add_scalar("eval/std_return", std_ret, global_step_eval)
    
    print(f"[Eval {global_step_eval}] Mean: {mean_ret:.1f} +/- {std_ret:.1f}")
    
    global_step_eval += 1
    return mean_ret

def collect_data(envs, buffer, steps, wm, agent, action_dim, device, runner_state, random_actions=False):
    global global_step_env
    obs = runner_state["obs"]
    prev_lives = runner_state["lives"]
    h = runner_state["h"]
    z_flat = runner_state["z_flat"]
    prev_actions = runner_state["prev_actions"]
    
    running_return = np.zeros(NUM_ENVS)
    
    wm.eval()
    if not random_actions:
        agent.eval()
    
    use_amp = (device.type == "cuda")
    
    for _ in range(steps // NUM_ENVS):
        global_step_env += NUM_ENVS
        
        with torch.inference_mode(), autocast(device_type=device.type, enabled=use_amp):
            obs_t = torch.tensor(obs, device=device, dtype=torch.float32)
            embed = wm.encoder(obs_t)
            act_oh = F.one_hot(torch.tensor(prev_actions, device=device), action_dim).float()
            h, logits = wm.rssm.step(h, z_flat, act_oh, embed)
            _, z_flat = wm.get_stochastic_state(logits, hard=True)
            feat = wm.rssm.get_feat(h, z_flat)

            if random_actions:
                actions = np.random.randint(0, action_dim, size=NUM_ENVS)
            else:
                probs = agent.get_action(feat)
                actions = Categorical(probs=probs).sample().cpu().numpy()
        
        next_obs, rewards, terms, truncs, infos = envs.step(actions)
        
        running_return += rewards
        real_dones = terms | truncs
        if real_dones.any():
            avg_ret = running_return[real_dones].mean()
            writer.add_scalar("train/avg_return", avg_ret, global_step_env)
            running_return[real_dones] = 0
            
        lives = infos["lives"]
        life_loss = (lives < prev_lives) & (lives > 0)
        
        episode_boundary = (terms | truncs | life_loss).astype(np.float32)
        rl_terminals = (terms | life_loss).astype(np.float32)

        # Retrieve true final observation
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
             if "_final_observation" in infos:
                 for i, has_final in enumerate(infos["_final_observation"]):
                     if has_final:
                         real_next_obs[i] = infos["final_observation"][i]
        
        buffer.add_batch(obs, real_next_obs, actions, rewards, episode_boundary, rl_terminals)
        
        mask = torch.tensor(1.0 - episode_boundary, device=device).unsqueeze(1)
        h = h * mask
        z_flat = z_flat * mask
        prev_actions = actions * (1 - episode_boundary).astype(np.int64)
        
        obs = next_obs
        prev_lives = lives

    runner_state["obs"] = obs
    runner_state["lives"] = prev_lives
    runner_state["h"] = h
    runner_state["z_flat"] = z_flat
    runner_state["prev_actions"] = prev_actions

def train_world_model(buffer, model, optimizer, scaler, action_dim, device, epochs): 
    global global_step_wm
    model.train()
    use_amp = (device.type == "cuda")
    
    for _ in range(epochs):
        batch = buffer.sample_sequence(BATCH_SIZE, recent_only_pct=0.2, 
                                       force_terminal_rate=0.05, 
                                       force_reward_rate=0.5)
        if batch is None: continue
        
        # UNPACK 6 items
        b_obs, b_next_obs, b_act, b_rew, b_bound, b_term = [x.to(device) for x in batch]
        b_rew_sym = symlog(b_rew)
        
        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=use_amp):
            B, T = b_obs.shape[:2]
            
            # --- FIX: Encoder Gradients ---
            # Do NOT use torch.no_grad() here. We need gradients for the posterior path.
            # We reshape to (Batch * Time, ...) for the encoder
            embeds_next = model.encoder(b_next_obs.reshape(-1, 4, 64, 64)).view(B, T, -1)
            
            # For Barlow Twins target (e), we detach to stop gradients
            embeds_target = embeds_next.detach()
            
            h = torch.zeros(B, EMBED_DIM, device=device)
            z_f = torch.zeros(B, STOCH_DIM * CLASS_DIM, device=device)
            
            loss_kl, loss_rew, loss_done = 0, 0, 0
            state_feats_list = []
            emb_target_list = []
            
            mae_rew_list = []
            mae_rew_nz_list = [] # Metric: Non-Zero Reward MAE
            all_pred_done_probs = []
            all_target_dones = []
            
            # For reward classification metrics
            rew_pred_pos_list = []
            rew_target_pos_list = []

            # Pos Weight for Done Loss
            flat_targets = b_term.float()
            n_valid = flat_targets.numel()
            n_pos = flat_targets.sum()
            n_neg = (n_valid - n_pos).clamp(min=1.0)
            pos_weight = (n_neg / (n_pos + 1e-6)).clamp(1.0, 50.0)

            for t in range(T):
                # 1. Reset Hidden State if PREVIOUS step was boundary
                if t > 0:
                    mask = (1.0 - b_bound[:, t-1]).float()
                    h = h * mask
                    z_f = z_f * mask
                
                # 2. Action for CURRENT step
                act = F.one_hot(b_act[:, t], action_dim).float()

                # 3. Step RSSM
                h, prio_logits = model.rssm.step(h, z_f, act, embed=None)
                
                # 4. Posterior using TARGET (Next Obs) - WITH GRAD
                # We use the attached embeds_next to allow gradients to flow back to encoder
                post_in = torch.cat([h, embeds_next[:, t]], dim=-1)
                post_logits = model.rssm.posterior_net(post_in).view(-1, STOCH_DIM, CLASS_DIM)
                
                _, z_f = model.get_stochastic_state(post_logits, hard=True) 
                
                # 5. KL Balancing with Gradient Stop
                q_stop = onehot_dist_from_logits(post_logits.detach(), mix=LATENT_UNIMIX)
                p      = onehot_dist_from_logits(prio_logits, mix=LATENT_UNIMIX)
                kl_dyn = torch.distributions.kl_divergence(q_stop, p).sum(-1)
                
                q      = onehot_dist_from_logits(post_logits, mix=LATENT_UNIMIX)
                p_stop = onehot_dist_from_logits(prio_logits.detach(), mix=LATENT_UNIMIX)
                kl_rep = torch.distributions.kl_divergence(q, p_stop).sum(-1)
                
                loss_kl += BETA_DYN * torch.clamp(kl_dyn, min=FREE_NATS).mean() + \
                           BETA_REP * torch.clamp(kl_rep, min=FREE_NATS).mean()
                
                feat_post = model.rssm.get_feat(h, z_f)
                state_feats_list.append(feat_post)
                
                # Append detached target for Barlow Twins
                emb_target_list.append(embeds_target[:, t])
                
                # 6. Reconstruction Losses
                # Valid mask: 1.0 (We want to train on terminals too)
                valid = torch.ones(B, device=device)
                
                pred_rew_logits = model.reward_head(feat_post)
                rew_loss_elem = twohot_loss_per_sample(pred_rew_logits, b_rew_sym[:, t].squeeze(-1))
                loss_rew += (rew_loss_elem * valid).mean()
                
                pred_done_logits = model.done_head(feat_post)
                done_loss_elem = F.binary_cross_entropy_with_logits(
                    pred_done_logits, b_term[:, t], pos_weight=pos_weight, reduction='none'
                ).squeeze(-1)
                loss_done += (done_loss_elem * valid).mean()

                with torch.no_grad():
                    pred_rew_lin = TwoHotDist(logits=pred_rew_logits).mean_linear()
                    mae_rew = (pred_rew_lin - b_rew[:, t]).abs()
                    mae_rew_list.append(mae_rew.mean())
                    
                    # --- Metric: MAE on Non-Zero Rewards ---
                    mask_nz = (b_rew[:, t].abs() > 0.01)
                    if mask_nz.any():
                        mae_rew_nz_list.append((pred_rew_lin - b_rew[:, t]).abs()[mask_nz].mean())
                    
                    # --- Metric: Reward Detection (Classify > 0.1) ---
                    # Just to see if it predicts *something*
                    rew_pred_pos_list.append((pred_rew_lin.abs() > 0.1).float())
                    rew_target_pos_list.append((b_rew[:, t].abs() > 0.01).float())

                    all_pred_done_probs.append(torch.sigmoid(pred_done_logits))
                    all_target_dones.append(b_term[:, t])

            # R2-Dreamer Barlow Loss
            all_states = torch.cat(state_feats_list, dim=0) 
            # all_embs is detached because we used append(embeds_target)
            all_embs = torch.cat(emb_target_list, dim=0)
            
            k = model.projector(all_states) 
            e = model.embed_proj(all_embs)
            
            loss_barlow, _, _ = compute_barlow_loss(k, e)
            
            l_kl = loss_kl / T
            l_rew = W_REWARD * (loss_rew/T)
            l_done = W_DONE * (loss_done/T)
            l_barlow = BARLOW_SCALE * loss_barlow 
            
            loss = l_barlow + l_kl + l_rew + l_done
            
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        global_step_wm += 1
        
        if global_step_wm % 10 == 0:
            writer.add_scalar("train/wm_total_loss", loss.item(), global_step_wm)
            writer.add_scalar("train/wm_kl", l_kl.item(), global_step_wm)
            writer.add_scalar("train/wm_barlow", l_barlow.item(), global_step_wm)
            
            avg_rew_mae = torch.stack(mae_rew_list).mean().item()
            writer.add_scalar("sanity/rew_mae_global", avg_rew_mae, global_step_wm)
            
            # --- Log Reward NZ Metrics ---
            if len(mae_rew_nz_list) > 0:
                avg_rew_mae_nz = torch.stack(mae_rew_nz_list).mean().item()
                writer.add_scalar("sanity/rew_mae_nonzero", avg_rew_mae_nz, global_step_wm)

            with torch.no_grad():
                # Reward Classification Metrics
                flat_rew_pred = torch.cat(rew_pred_pos_list).view(-1)
                flat_rew_target = torch.cat(rew_target_pos_list).view(-1)
                tp_r = (flat_rew_pred * flat_rew_target).sum()
                fn_r = ((1 - flat_rew_pred) * flat_rew_target).sum()
                fp_r = (flat_rew_pred * (1 - flat_rew_target).float()).sum()
                rec_r = tp_r / (tp_r + fn_r + 1e-8)
                prec_r = tp_r / (tp_r + fp_r + 1e-8)
                writer.add_scalar("sanity/rew_recall_nz", rec_r.item(), global_step_wm)
                writer.add_scalar("sanity/rew_precision_nz", prec_r.item(), global_step_wm)
                
                # Done Metrics
                flat_preds = torch.cat(all_pred_done_probs).view(-1)
                flat_targets = torch.cat(all_target_dones).view(-1)
                
                mean_target_rate = flat_targets.mean()
                mean_pred_rate = flat_preds.mean()
                
                pred_pos = (flat_preds > 0.5).float()
                true_pos = (flat_targets > 0.5).float()
                
                tp = (pred_pos * true_pos).sum()
                fn = ((1 - pred_pos) * true_pos).sum()
                fp = (pred_pos * (1 - true_pos)).sum()
                
                recall = tp / (tp + fn + 1e-8)
                precision = tp / (tp + fp + 1e-8)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                
                writer.add_scalar("sanity/done_pos_rate_target", mean_target_rate.item(), global_step_wm)
                writer.add_scalar("sanity/done_pos_rate_pred", mean_pred_rate.item(), global_step_wm)
                writer.add_scalar("sanity/done_recall", recall.item(), global_step_wm)
                writer.add_scalar("sanity/done_precision", precision.item(), global_step_wm)
                writer.add_scalar("sanity/done_f1", f1.item(), global_step_wm)

def compute_lambda_returns(rewards, values, continues, gamma=0.99, lamb=0.95):
    """
    Compute lambda-returns with expected continuation (discount)
    rewards: [T, B, 1]
    values: [T+1, B, 1]
    continues: [T, B, 1] - Probability of continuing (1.0 - done_prob)
    """
    returns = []
    last = values[-1]
    for t in reversed(range(len(rewards))):
        # Effective discount = gamma * P(continue)
        disc = gamma * continues[t]
        
        # Standard Lambda-Return recursion with variable discount
        # R_t = r_t + disc * ( (1-lambda)*v_{t+1} + lambda*R_{t+1} )
        
        next_val = values[t+1]
        
        # "Expected continuation" formulation
        # This softly discounts the future value by the probability of termination
        term = rewards[t] + disc * next_val + disc * lamb * (last - next_val)
        
        last = term
        returns.insert(0, last)
    return torch.stack(returns)

def train_policy_dreamer(buffer, wm, agent, opt_act, opt_crit, scaler_a, scaler_c, action_dim, device, epochs, retnorm): 
    global global_step_agent
    wm.eval(); agent.train()
    wm.requires_grad_(False)
    
    use_amp = (device.type == "cuda")

    for _ in range(epochs):
        # Sample with reward forcing
        batch = buffer.sample_sequence(BATCH_SIZE, recent_only_pct=0.2, 
                                       force_terminal_rate=0.05, 
                                       force_reward_rate=0.5)
        if batch is None: continue
        
        b_obs, b_next_obs, b_act, b_rew, b_bound, b_term = [x.to(device) for x in batch]
        
        with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
            B, T_ctx = b_obs[:, :BURN_IN_STEPS].shape[:2]
            # Use b_obs for context (history)
            embeds = wm.encoder(
                b_obs[:, :BURN_IN_STEPS].reshape(-1, 4, 64, 64)
            ).view(B, T_ctx, -1)
        
            h = torch.zeros(B, EMBED_DIM, device=device)
            z_f = torch.zeros(B, STOCH_DIM * CLASS_DIM, device=device)
        
            for t in range(T_ctx):
                if t == 0:
                    act = torch.zeros(B, action_dim, device=device)
                else:
                    mask = (1.0 - b_bound[:, t-1]).float()
                    h = h * mask
                    z_f = z_f * mask
                    act = F.one_hot(b_act[:, t-1], action_dim).float() * mask
        
                h, logits = wm.rssm.step(h, z_f, act, embeds[:, t])
                _, z_f = wm.get_stochastic_state(logits, hard=True)

        opt_act.zero_grad(); opt_crit.zero_grad()
        with autocast(device_type=device.type, enabled=use_amp):
            l_rew_lin, l_val_lin, l_continue_probs, l_ent = [], [], [], []
            l_val_logits = [] 
            l_logp = []   # Store log π(a|s)
            
            curr_h, curr_z = h, z_f
            
            v_logits = agent.critic(wm.rssm.get_feat(h, z_f))
            l_val_logits.append(v_logits)
            l_val_lin.append(TwoHotDist(logits=v_logits).mean_linear())

            for _ in range(IMAGINE_HORIZON):
                feat = wm.rssm.get_feat(curr_h, curr_z)
                
                probs = agent.get_action(feat, hard=False)
                dist = Categorical(probs=probs) 
                
                action_idx = dist.sample()
                action_oh = F.one_hot(action_idx, action_dim).float()
                
                l_logp.append(dist.log_prob(action_idx).unsqueeze(-1)) # [B, 1]
                l_ent.append(dist.entropy().mean())
                
                curr_h, logits = wm.rssm.step(curr_h, curr_z, action_oh, None)
                _, curr_z = wm.get_stochastic_state(logits, hard=True)
                feat_next = wm.rssm.get_feat(curr_h, curr_z)
                
                r_logits = wm.reward_head(feat_next)
                l_rew_lin.append(TwoHotDist(logits=r_logits).mean_linear())
                
                # --- Expected Continuation Logic ---
                # "Done" logits -> probability of terminating
                # "Continue" probability = 1.0 - sigmoid(done_logits)
                d_logits = wm.done_head(feat_next)
                cont_prob = (1.0 - torch.sigmoid(d_logits)).clamp(0.0, 1.0)
                l_continue_probs.append(cont_prob)

                v_logits_next = agent.critic(feat_next)
                l_val_logits.append(v_logits_next)
                l_val_lin.append(TwoHotDist(logits=v_logits_next).mean_linear())
                
            rets_linear = compute_lambda_returns(
                torch.stack(l_rew_lin), 
                torch.stack(l_val_lin).detach(), 
                torch.stack(l_continue_probs), # Passing continuous probs
                gamma=GAMMA,
                lamb=LAMBDA
            )
            
            target_v_symlog = symlog(rets_linear.detach())
            pred_logits = torch.stack(l_val_logits)[:-1] 
            
            # Use the new element-wise loss from models.py to preserve dimensions
            crit_loss_elem = TwoHotDist.compute_loss_elem(pred_logits, target_v_symlog) # [T, B, 1]
            
            entropy = torch.stack(l_ent).mean()
            
            # Baseline values at s_t (not s_{t+1})
            values_t = torch.stack(l_val_lin)[:-1].detach()          # [T, B, 1]
            logp_t   = torch.stack(l_logp)                           # [T, B, 1]

            # Return normalization scale (Dreamer-style: actor uses normalized advantage)
            scale = retnorm.update(rets_linear)                      # scalar tensor
            scale_safe = torch.clamp(scale, min=1.0)

            adv = ((rets_linear - values_t) / scale_safe).detach()   # [T, B, 1], stop-grad
            
            # --- Weighted Loss by Cumulative Continuation ---
            # Future steps matter less if the model predicts termination earlier
            # weights[t] = product_{i=0}^{t-1} (gamma * continue_prob[i])
            discounts = torch.stack(l_continue_probs).detach() * GAMMA
            ones = torch.ones_like(discounts[:1])
            # Cumulative product along Time dimension (dim=0)
            weights = torch.cumprod(torch.cat([ones, discounts[:-1]], dim=0), dim=0).detach()
            
            act_loss = -(weights * logp_t * adv).mean() - ENTROPY_SCALE * entropy
            crit_loss = (crit_loss_elem * weights).mean()
        
        scaler_c.scale(crit_loss).backward()
        scaler_a.scale(act_loss).backward()

        scaler_a.unscale_(opt_act)
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), ACTOR_GRAD_CLIP)
        
        scaler_c.step(opt_crit); scaler_a.step(opt_act)
        scaler_c.update(); scaler_a.update()
        global_step_agent += 1
        
        if global_step_agent % 10 == 0:
            writer.add_scalar("train/critic_loss", crit_loss.item(), global_step_agent)
            writer.add_scalar("train/actor_loss", act_loss.item(), global_step_agent)
            writer.add_scalar("train/entropy", entropy.item(), global_step_agent)
            writer.add_scalar("train/retnorm_scale", scale.mean().item(), global_step_agent)
    
    wm.requires_grad_(True)

def main():
    torch.backends.cudnn.benchmark = True
    
    device = get_device()
    
    print(f"Checking environment: {ENV_NAME}")
    tmp_env = make_single_env(ENV_NAME)
    action_dim = tmp_env.action_space.n
    tmp_env.close()
    print(f"Detected Action Dim: {action_dim}")

    buffer = ReplayBuffer(capacity=50000, num_envs=NUM_ENVS, seq_len=SEQ_LEN)
    
    train_envs = make_vector_env(NUM_ENVS, ENV_NAME)
    
    obs_init, infos_init = train_envs.reset()
    runner_state = {
        "obs": obs_init,
        "lives": infos_init["lives"],
        "h": torch.zeros(NUM_ENVS, EMBED_DIM, device=device),
        "z_flat": torch.zeros(NUM_ENVS, STOCH_DIM * CLASS_DIM, device=device),
        "prev_actions": np.zeros(NUM_ENVS, dtype=np.int64)
    }

    wm = WorldModel(action_dim).to(device)
    agent = ActorCritic(action_dim).to(device)
    
    try:
        wm, agent = torch.compile(wm), torch.compile(agent)
    except: pass

    opt_wm = optim.Adam(wm.parameters(), lr=4e-5)
    opt_act = optim.Adam(agent.actor.parameters(), lr=4e-5)
    opt_crit = optim.Adam(agent.critic.parameters(), lr=4e-5)
    scaler_wm, scaler_a, scaler_c = GradScaler(), GradScaler(), GradScaler()
    
    retnorm = RetNorm()
    
    current_steps_counter = 0

    best_eval_return = -float('inf')

    try:
        for i in tqdm(range(TOTAL_ITERATIONS)):
            is_warmup = (current_steps_counter < WARMUP_STEPS)
            collect_data(train_envs, buffer, STEPS_PER_ITER, wm, agent, action_dim, device, runner_state, random_actions=is_warmup)
            
            current_steps_counter += STEPS_PER_ITER
            
            train_world_model(buffer, wm, opt_wm, scaler_wm, action_dim, device, WM_EPOCHS) 
            
            if current_steps_counter > WARMUP_STEPS:
                train_policy_dreamer(buffer, wm, agent, opt_act, opt_crit, scaler_a, scaler_c, action_dim, device, AGENT_EPOCHS, retnorm)
            
            if i % 100 == 0:
                current_eval_ret = evaluate_agent(wm, agent, action_dim, device)
                if current_eval_ret > best_eval_return:
                    best_eval_return = current_eval_ret
                    print(f"New Record: {best_eval_return:.2f}! Saving best models...")
                    save_model(wm, "best_world_model.pth")
                    save_model(agent, "best_policy.pth")
                check_model_rollout(wm, buffer, action_dim, device)

            if i % 200 == 0:
                save_model(wm, "world_model.pth")
                save_model(agent, "policy.pth")
    finally:
        train_envs.close()

if __name__ == "__main__":
    writer = SummaryWriter(log_dir=f"runs/Atari")
    main()
