#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 256          
STOCH_DIM = 8            
CLASS_DIM = 16           
ACTION_DIM = 6
INPUT_CHANNELS = 4 

PROJECTOR_DIM = 256      

class LayerNormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.ln_ih = nn.LayerNorm(3 * hidden_size)
        self.ln_hh = nn.LayerNorm(3 * hidden_size)
        
    def forward(self, x, h):
        ih = self.ln_ih(self.weight_ih(x))
        hh = self.ln_hh(self.weight_hh(h))
        i_r, i_z, i_n = ih.chunk(3, dim=-1)
        h_r, h_z, h_n = hh.chunk(3, dim=-1)
        
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        return (1 - update_gate) * new_gate + update_gate * h

class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 16, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten() 
        )
        self.out = nn.Linear(1024, EMBED_DIM)
    
    def forward(self, x):
        y = self.net(x)
        return self.out(y)

class BarlowProjector(nn.Module):
    def __init__(self, input_dim, output_dim=PROJECTOR_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = LayerNormGRUCell(ACTION_DIM + STOCH_DIM * CLASS_DIM, EMBED_DIM)
        
        self.posterior_net = nn.Sequential(
            nn.Linear(EMBED_DIM + EMBED_DIM, 256),
            nn.ELU(),
            nn.Linear(256, STOCH_DIM * CLASS_DIM)
        )
        
        self.prior_net = nn.Sequential(
            nn.Linear(EMBED_DIM, 256),
            nn.ELU(),
            nn.Linear(256, STOCH_DIM * CLASS_DIM)
        )

    def get_feat(self, h, z_flat):
        return torch.cat([h, z_flat], dim=-1)

    def step(self, prev_h, prev_z_flat, action, embed=None):
        gru_input = torch.cat([prev_z_flat, action], dim=-1)
        h = self.gru(gru_input, prev_h)
        
        if embed is not None:
            post_in = torch.cat([h, embed], dim=-1)
            logits = self.posterior_net(post_in)
        else:
            logits = self.prior_net(h)
            
        logits = logits.view(-1, STOCH_DIM, CLASS_DIM)
        return h, logits

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VisualEncoder()
        self.rssm = RSSM()
        
        feat_dim = EMBED_DIM + STOCH_DIM * CLASS_DIM
        self.projector = BarlowProjector(feat_dim)
        
        self.reward_head = nn.Linear(feat_dim, 1)
        self.done_head = nn.Linear(feat_dim, 1)

    def get_stochastic_state(self, logits, temperature=1.0, hard=False):
        z = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        z_flat = z.view(z.size(0), -1)
        return z, z_flat

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        feat_dim = EMBED_DIM + STOCH_DIM * CLASS_DIM
        
        self.actor = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, ACTION_DIM)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def get_action_logits(self, feature):
        return self.actor(feature)

    def get_action(self, feature, temperature=1.0):
        logits = self.actor(feature)
        probs = F.softmax(logits / temperature, dim=-1)
        return probs

    def get_value(self, feature):
        return self.critic(feature)
