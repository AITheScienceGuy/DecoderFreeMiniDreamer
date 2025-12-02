#!/usr/bin/env python

import gymnasium as gym
import torch
import numpy as np
import os
from PIL import Image  
from models import WorldModel, ActorCritic, EMBED_DIM, STOCH_DIM, CLASS_DIM
from utils import get_device, PreprocessAtari

def save_gif(frames, filename="gameplay.gif"):
    print(f"Saving {filename}...")
    imgs = [Image.fromarray(frame) for frame in frames]
    imgs[0].save(filename, save_all=True, append_images=imgs[1:], optimize=False, duration=33, loop=0)
    print("Saved!")

def watch_agent():
    device = get_device()

    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array", frameskip=1)
    
    env = gym.wrappers.AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=84, 
        terminal_on_life_loss=False, grayscale_obs=True
    )
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = PreprocessAtari(env, size=(64, 64))
    
    print(f"Loading models (Embed: {EMBED_DIM}, Stoch: {STOCH_DIM*CLASS_DIM})...")
    wm = WorldModel().to(device)
    try:
        wm.load_state_dict(torch.load("world_model.pth", map_location=device))
    except FileNotFoundError:
        print("world_model.pth not found!")
        return
    wm.eval()

    agent = ActorCritic().to(device)
    try:
        agent.load_state_dict(torch.load("policy.pth", map_location=device))
    except FileNotFoundError:
        print("policy.pth not found!")
        return
    agent.eval()

    print("Starting Recording...")
    
    for episode in range(3): 
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        h = torch.zeros(1, EMBED_DIM, device=device)
        z_flat = torch.zeros(1, STOCH_DIM * CLASS_DIM, device=device) 
        
        action = 0
        
        frames = [] 
        
        while not done:
            frame = env.render() 
            frames.append(frame)

            obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)
            
            with torch.no_grad():
                embed = wm.encoder(obs_tensor)
                
                action_t = torch.tensor([action], device=device)
                action_one_hot = torch.nn.functional.one_hot(action_t, 6).float()
                
                h, logits = wm.rssm.step(h, z_flat, action_one_hot, embed=embed)
                z, z_flat = wm.get_stochastic_state(logits, hard=True)
                
                feat = wm.rssm.get_feat(h, z_flat)
                
                # Low temperature for deterministic play
                action_probs = agent.get_action(feat, temperature=0.1)
                action = torch.argmax(action_probs).item()
            
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            total_reward += reward

        print(f"Episode {episode+1}: Total Reward {total_reward}")
        
        save_gif(frames, filename=f"episode_{episode+1}.gif")

    env.close()

if __name__ == "__main__":
    watch_agent()
