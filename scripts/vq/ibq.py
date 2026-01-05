import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from torch import einsum
from einops import rearrange

import torch.distributed as dist
from torch.distributed import get_rank, barrier, is_initialized


class IBQ(nn.Module):
    def __init__(self, n_e, e_dim, quantization_temp=2.0, beta=0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.quantization_temp = quantization_temp
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.init = False
        self.codebook_loss_weight = 0.25
        # self.codebook_loss_weight = 100
    
    def forward(self, z, is_image=None):
        # z = F.normalize(z, p=2, dim=-1)
        # embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        embedding = self.embedding.weight

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('b d,d n->b n', z, torch.einsum('n d -> d n', embedding))
        
        if self.training:
            logits = -d / self.quantization_temp
            soft_one_hot = F.softmax(logits, dim=1)
            min_encoding_indices = soft_one_hot.max(1, keepdim=True)[1]
            hard_one_hot = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(1, min_encoding_indices, 1.0)
            one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot

            z_q = torch.einsum('b n, n d -> b d', one_hot, self.embedding.weight)
            z_q_2 = torch.einsum('b n, n d -> b d', hard_one_hot, self.embedding.weight)

            # compute loss for embedding
            commit_loss = torch.mean((z_q - z) ** 2) + torch.mean((z_q_2.detach() - z) ** 2) + self.beta * torch.mean((z_q_2 - z.detach()) ** 2)
            commit_loss = self.codebook_loss_weight * commit_loss
        else:
            min_encoding_indices = torch.argmin(d, dim=1)
            z_q = embedding[min_encoding_indices].view(z.shape)
            commit_loss = torch.tensor(0.0)

        num_codes = min_encoding_indices[:, 0].unique().numel() if min_encoding_indices.ndim > 1 else min_encoding_indices.unique().numel()
        if dist.get_rank() == 0 and self.training:
            print(f"[IBQ Train log] is {'image' if is_image else 'video'} | Sequence length={z.shape[0]:4d} | Unique codes={num_codes:4d} | "f"Commit loss={commit_loss.item():.6f}")
        else:
            print(f"[IBQ Infer log] is {'image' if is_image else 'video'} | Sequence length={z.shape[0]:4d} | Unique codes={num_codes:4d} | "f"Commit loss={commit_loss.item():.6f}")

        return z_q, dict(loss=commit_loss)


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"



class Recon_Head(nn.Module):
    def __init__(self, codebook_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(codebook_dim, codebook_dim),
            nn.SiLU(),
            nn.Linear(codebook_dim, codebook_dim),
        )
        self.ln_q = Qwen2RMSNorm(codebook_dim, eps=1e-6)

    def forward(self, x):
        x = self.ln_q(x)
        x_in = x
        x = self.mlp(x)
        x = x + x_in
        return x


class VQ(nn.Module):
    def __init__(
        self, 
        z_channels=2048, 
        codebook_size=16384, 
        codebook_dim=2048, 
        transformer_out_layers=4,
        use_transformer=True,
        config=None,
    ):
        super().__init__()
        self.quantize = IBQ(codebook_size, codebook_dim)
        self.config = config
    
    
    def forward(self, x, cu_seqlens=None, position_embeddings=None, is_image=None):
        assert torch.isfinite(x).all(), f"x has NaN/Inf: {x}"

        # quantize
        x, codebook_loss = self.quantize(x, is_image=is_image)

        if torch.isnan(codebook_loss['loss']).any():
            if not is_initialized() or get_rank() == 0:
                print("embedding:", self.quantize.embedding.weight.norm().item())
                print("x:", x.norm().item())
                import pdb; pdb.set_trace()
            if is_initialized():
                barrier()

        return x, codebook_loss
