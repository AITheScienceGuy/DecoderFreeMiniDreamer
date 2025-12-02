#!/usr/bin/env python

import gymnasium as gym
import ale_py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models import WorldModel, ActorCritic, ACTION_DIM, EMBED_DIM, STOCH_DIM, CLASS_DIM
from utils import get_device, PreprocessAtari

def make_eval_env():
    gym.register_envs(ale_py)
    env = gym.make("ALE/SpaceInvaders-v5", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, 
                                          screen_size=84, terminal_on_life_loss=False, 
                                          grayscale_obs=True)
    env = gym.wrappers.FrameStackObservation(env, 4) 
    env = PreprocessAtari(env, size=(64, 64))
    return env

def run_benchmark(n_episodes=50):
    device = get_device()
    env = make_eval_env() 
    
    print("Loading models...")
    wm = WorldModel().to(device)
    try:
        wm.load_state_dict(torch.load("world_model.pth", map_location=device))
    except FileNotFoundError:
        print("Error: world_model.pth not found. Train first!")
        return
    wm.eval()

    agent = ActorCritic().to(device)
    try:
        agent.load_state_dict(torch.load("policy.pth", map_location=device))
    except FileNotFoundError:
        print("Error: policy.pth not found. Train first!")
        return
    agent.eval()

    scores = []
    
    print(f"Running benchmark over {n_episodes} episodes...")
    for i in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False
        score = 0
        
        h = torch.zeros(1, EMBED_DIM, device=device)
        z_flat = torch.zeros(1, STOCH_DIM * CLASS_DIM, device=device)
        action_idx = 0
        
        while not done:
            obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)
            
            with torch.no_grad():
                embed = wm.encoder(obs_tensor)
                action_one_hot = F.one_hot(torch.tensor([action_idx], device=device), ACTION_DIM).float()
                
                h, logits = wm.rssm.step(h, z_flat, action_one_hot, embed)
                z, z_flat = wm.get_stochastic_state(logits, hard=True)
                
                feat = wm.rssm.get_feat(h, z_flat)
                
                action_probs = agent.get_action(feat, temperature=0.1)
                action_idx = torch.argmax(action_probs).item()
            
            obs, reward, term, trunc, _ = env.step(action_idx)
            done = term or trunc
            score += reward
            
        scores.append(score)

    avg_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"\n--- Benchmark Results ({n_episodes} Episodes) ---")
    print(f"Average Score: {avg_score:.2f} +/- {std_score:.2f}")

if __name__ == "__main__":
    run_benchmark()
