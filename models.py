#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

EMBED_DIM = 512          
STOCH_DIM = 32            
CLASS_DIM = 32           
INPUT_CHANNELS = 4 
PROJECTOR_DIM = 256

# DreamerV3 Two-Hot Bucketing
NUM_BUCKETS = 255
BUCKET_LOW = -20.0
BUCKET_HIGH = 20.0
LATENT_UNIMIX = 0.01
ACTOR_UNIMIX = 0.01

HL_GAUSS_SIGMA_RATIO = 0.25  # good default to start with (tune later)

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

class RetNorm:
    def __init__(self, decay=0.99, limit=1.0, lo=0.05, hi=0.95):
        self.decay = decay
        self.limit = limit
        self.lo = lo
        self.hi = hi
        self.scale = torch.ones(1)

    def update(self, returns):
        returns = returns.detach()
        lo = torch.quantile(returns, self.lo)
        hi = torch.quantile(returns, self.hi)
        s = torch.clamp(hi - lo, min=self.limit)
        self.scale = self.scale.to(returns.device)
        self.scale = self.decay * self.scale + (1 - self.decay) * s
        return self.scale

class HLGaussDist:
    def __init__(self, logits=None, low=BUCKET_LOW, high=BUCKET_HIGH, bins=NUM_BUCKETS, sigma_ratio=HL_GAUSS_SIGMA_RATIO):
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1) if logits is not None else None
        self.low, self.high, self.bins = low, high, bins

        # Edges (num_bins + 1), centers (num_bins)
        device = logits.device if logits is not None else 'cpu'
        self.support = torch.linspace(low, high, bins + 1, device=device, dtype=torch.float32)
        self.centers = (self.support[:-1] + self.support[1:]) / 2.0

        # sigma as a fraction of bin width
        bin_width = (high - low) / bins
        self.sigma = sigma_ratio * bin_width

    def mean_linear(self):
        # centers are in symlog-space in your pipeline, so map back with symexp
        return (self.probs * symexp(self.centers)).sum(dim=-1, keepdim=True)

    def to_symlog_scalar(self):
        return (self.probs * self.centers).sum(dim=-1, keepdim=True)

    def transform_to_probs(self, target_symlog):
        # target_symlog: shape (...,) or (..., 1) -> squeeze last dim if present
        t = target_symlog.squeeze(-1)
        t = torch.clamp(t, self.low, self.high)

        # Match Stop Regressing listing: erf((support - target) / (sqrt(2)*sigma))
        denom = torch.sqrt(torch.tensor(2.0, device=t.device, dtype=torch.float32)) * self.sigma
        cdf_evals = torch.special.erf((self.support - t.unsqueeze(-1)) / denom)  # (..., bins+1)

        z = cdf_evals[..., -1] - cdf_evals[..., 0]                              # (...)
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]                    # (..., bins)

        # Normalize for truncation at edges (as in the paper)
        return bin_probs / (z.unsqueeze(-1) + 1e-8)

    @staticmethod
    def compute_loss_elem(logits, targets_symlog, low=BUCKET_LOW, high=BUCKET_HIGH, bins=NUM_BUCKETS, sigma_ratio=HL_GAUSS_SIGMA_RATIO):
        # logits: (..., bins), targets_symlog: (..., 1) or (...)
        support = torch.linspace(low, high, bins + 1, device=logits.device, dtype=torch.float32)
        bin_width = (high - low) / bins
        sigma = sigma_ratio * bin_width

        t = targets_symlog.squeeze(-1)
        t = torch.clamp(t, low, high)

        denom = torch.sqrt(torch.tensor(2.0, device=logits.device, dtype=torch.float32)) * sigma
        cdf_evals = torch.special.erf((support - t.unsqueeze(-1)) / denom)
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        target_probs = (cdf_evals[..., 1:] - cdf_evals[..., :-1]) / (z.unsqueeze(-1) + 1e-8)

        logp = F.log_softmax(logits, dim=-1)
        ce = -(target_probs * logp).sum(dim=-1, keepdim=True)  # (..., 1)
        return ce

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.scale

class MlpBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            RMSNorm(output_dim),
            nn.SiLU()
        )
        self.has_residual = (input_dim == output_dim)

    def forward(self, x):
        out = self.net(x)
        if self.has_residual:
            return x + out
        return out

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
            nn.Conv2d(INPUT_CHANNELS, 32, 4, stride=2, padding=1), 
            nn.GroupNorm(1, 32), 
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), 
            nn.GroupNorm(1, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), 
            nn.GroupNorm(1, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.GroupNorm(1, 128),
            nn.SiLU(),
            nn.Flatten() 
        )
        self.out = nn.Sequential(
            nn.Linear(2048, EMBED_DIM),
            RMSNorm(EMBED_DIM),
            nn.SiLU()
        )
    
    def forward(self, x):
        y = self.net(x)
        return self.out(y)

class BarlowProjector(nn.Module):
    def __init__(self, input_dim, output_dim=PROJECTOR_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class RSSM(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.gru = LayerNormGRUCell(action_dim + STOCH_DIM * CLASS_DIM, EMBED_DIM)
        
        self.posterior_net = nn.Sequential(
            MlpBlock(EMBED_DIM + EMBED_DIM, 512),
            nn.Linear(512, STOCH_DIM * CLASS_DIM)
        )
        
        self.prior_net = nn.Sequential(
            MlpBlock(EMBED_DIM, 512),
            nn.Linear(512, STOCH_DIM * CLASS_DIM)
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
    def __init__(self, action_dim):
        super().__init__()
        self.encoder = VisualEncoder()
        self.rssm = RSSM(action_dim)
        
        feat_dim = EMBED_DIM + STOCH_DIM * CLASS_DIM
        self.projector = BarlowProjector(feat_dim)
        self.embed_proj = nn.Linear(EMBED_DIM, PROJECTOR_DIM, bias=False)
        
        self.reward_head = nn.Sequential(
            MlpBlock(feat_dim, 256),
            nn.Linear(256, NUM_BUCKETS)
        )
        self.done_head = nn.Sequential(
            MlpBlock(feat_dim, 256),
            nn.Linear(256, 1)
        )
        
        nn.init.zeros_(self.reward_head[-1].weight)
        nn.init.zeros_(self.reward_head[-1].bias)

    def get_stochastic_state(self, logits, temperature=1.0, hard=False):
        probs = torch.softmax(logits, dim=-1)
        probs = (1.0 - LATENT_UNIMIX) * probs + LATENT_UNIMIX / probs.shape[-1]
        logits_unimix = torch.log(probs.clamp(min=1e-8))
        
        z = F.gumbel_softmax(logits_unimix, tau=temperature, hard=hard, dim=-1)
        z_flat = z.view(z.size(0), -1)
        return z, z_flat

class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        feat_dim = EMBED_DIM + STOCH_DIM * CLASS_DIM
        hidden_dim = 512
        
        self.actor = nn.Sequential(
            MlpBlock(feat_dim, hidden_dim),
            MlpBlock(hidden_dim, hidden_dim),
            MlpBlock(hidden_dim, hidden_dim),
            MlpBlock(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.critic = nn.Sequential(
            MlpBlock(feat_dim, hidden_dim),
            MlpBlock(hidden_dim, hidden_dim),
            MlpBlock(hidden_dim, hidden_dim),
            MlpBlock(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, NUM_BUCKETS)
        )
        
        self.actor[-1].weight.data.uniform_(-1e-3, 1e-3)
        self.actor[-1].bias.data.zero_()
        
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def get_action_logits(self, feature):
        return self.actor(feature)

    def get_action(self, feature, temperature=1.0, hard=False):
        logits = self.actor(feature)
        probs = torch.softmax(logits / temperature, dim=-1)
        
        probs = (1.0 - ACTOR_UNIMIX) * probs + ACTOR_UNIMIX / probs.shape[-1]
        
        if hard:
            logits_u = torch.log(probs.clamp(min=1e-8))
            return F.gumbel_softmax(logits_u, tau=temperature, hard=True, dim=-1)
        else:
            return probs
