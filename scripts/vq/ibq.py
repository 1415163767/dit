import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch import einsum
from einops import rearrange

import torch.distributed as dist
from torch.distributed import get_rank, barrier, is_initialized


class new_IBQ(nn.Module):
    def __init__(self, n_e, e_dim, is_train=True, norm=True):
        super().__init__()
        self.nb_code = n_e
        self.code_dim = e_dim
        self.norm = norm
        self.commit_loss_weight = 0.25
        self.mu = 0.99
        self.reset_codebook()
        self.reset_count = 0
        self.is_train = is_train
        self._global_unique_count = 0

        # cross-GPU, per-step stats (not accumulated)
        self._step_unique_global = 0
        self._step_tokens_global = 0
        
        # last-batch stats
        self._last_unique_count = 0
        self._last_total_count = 0

        self.is_resume = True
        self.is_train = False


    def reset_codebook(self):
        self.init = False
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim, dtype=torch.float32))
        self.register_buffer("usage", torch.zeros((self.nb_code, 1), dtype=torch.float32))
        self.register_buffer("global_usage_mask", torch.zeros(self.nb_code, dtype=torch.bool))
        self.register_buffer("code_sum", torch.zeros(self.nb_code, self.code_dim, dtype=torch.float32), persistent=True)
        self.register_buffer("code_count", torch.zeros(self.nb_code, dtype=torch.float32), persistent=True)


    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out
    
    
    def init_codebook(self, x):
        rank = dist.get_rank() if dist.is_initialized() else 0

        # only rank0 creates the initial random/tiled codebook
        if rank == 0 and torch.all(self.codebook == 0):
            out = self._tile(x)
            self.codebook.data.copy_(out[: self.nb_code].to(dtype=self.codebook.dtype))

        # ensure everyone has the same codebook
        if dist.is_initialized():
            dist.broadcast(self.codebook, src=0)
            dist.barrier()

        # initialize sums/counters from codebook (float32)
        self.code_sum.data.copy_(self.codebook.clone().to(dtype=torch.float32))
        self.code_count.data.fill_(1.0)
        if self.is_train:
            self.init = True

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        device = x.device
        eps = 1e-6
        reset_period = 20

        # code_onehot: shape [nb_code, N_tokens_this_rank]
        N = code_idx.numel()
        code_onehot = torch.zeros(self.nb_code, N, device=device, dtype=torch.float32)
        code_onehot.scatter_(0, code_idx.view(1, N), 1.0)
        
        # per-rank sums/counts
        code_sum_local = torch.matmul(code_onehot, x.float())  # [nb_code, code_dim]
        code_count_local = code_onehot.sum(dim=-1)             # [nb_code]

        # Distributed synchronization: all_reduce to get global batch statistics
        if dist.is_initialized():
            dist.all_reduce(code_sum_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(code_count_local, op=dist.ReduceOp.SUM)
        
        # EMA update with synchronized global batch stats
        self.code_sum.data = self.mu * self.code_sum + (1.0 - self.mu) * code_sum_local
        self.code_count.data = self.mu * self.code_count + (1.0 - self.mu) * code_count_local
        current_usage = (code_count_local.view(self.nb_code, 1) >= 1.0).to(dtype=torch.float32)
        
        if self.reset_count >= reset_period:
            self.reset_count = 0
            final_usage_mask = (self.usage + current_usage >= 1.0).to(dtype=torch.float32)
            self.usage.zero_()
        else:
            self.reset_count += 1
            self.usage.data = (self.usage + current_usage >= 1.0).to(dtype=torch.float32)
            final_usage_mask = torch.ones_like(self.usage, device=device)
        
        # code_update (safe divide)
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / (self.code_count.view(self.nb_code, 1) + eps)

        # Synchronized random replacement: use same seed across all ranks
        rand_pool = self._tile(x)
        assert torch.isfinite(rand_pool).all(), "rand_pool has NaN"

        # use a synchronized seed to permute the rand_pool rows
        if dist.is_initialized():
            rand_seed = torch.randint(0, 2**31, (1,), device=device, dtype=torch.long)
            dist.broadcast(rand_seed, src=0)
            generator = torch.Generator(device=device)
            generator.manual_seed(int(rand_seed.item()))
            perm = torch.randperm(rand_pool.shape[0], device=device, generator=generator)
            code_rand = rand_pool[perm][: self.nb_code].to(dtype=self.codebook.dtype)
        else:
            code_rand = rand_pool[torch.randperm(rand_pool.shape[0], device=device)][: self.nb_code]

        # broadcast code_rand from rank0 to others and barrier
        if dist.is_initialized():
            dist.broadcast(code_rand, src=0)
        
        # updata codebook
        new_codebook = final_usage_mask * code_update + (1.0 - final_usage_mask) * code_rand
        self.codebook.data.copy_(new_codebook.to(self.codebook.dtype))
        
        if dist.is_initialized():
            dist.broadcast(self.codebook, src=0)

        # compute perplexity safely
        prob = self.code_count / (torch.sum(self.code_count) + eps)
        total_perplexity = torch.exp(-torch.sum(prob * torch.log(prob + eps)))
        current_prob = code_count_local / (torch.sum(code_count_local) + eps)
        current_perplexity = torch.exp(-torch.sum(current_prob * torch.log(current_prob + eps)))

        return total_perplexity, current_perplexity


    def quantize(self, x):
        # [bs * f * j, dim=2048]
        k_w = self.codebook.t().float()
        distance = torch.sum(x.float() ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x.float(), k_w) + torch.sum(k_w ** 2, dim=0, keepdim=True)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)    # indexing: [bs * f * j, 2048]
        return x

    def forward(self, x, info="vq_final"):
        assert x.shape[-1] == self.code_dim

        if self.norm:
            x_norm = F.normalize(x, p=2, dim=-1)
        else:
            x_norm = x

        # Init codebook if not inited
        if not self.init and self.is_train and not self.is_resume:
            self.init_codebook(x_norm)
        elif self.is_resume:
            self.init = True


        # quantize
        code_idx = self.quantize(x_norm)
        
        # Update global usage tracking with distributed sync
        unique_codes = torch.unique(code_idx)
        batch_usage_mask = torch.zeros_like(self.global_usage_mask)
        batch_usage_mask[unique_codes] = True

        # Sync step usage mask across all ranks to obtain step-level unique codes (not accumulated)
        if dist.is_initialized():
            dist.all_reduce(batch_usage_mask, op=dist.ReduceOp.MAX)
            # Compute step-level global tokens across ranks
            _tok = torch.tensor(int(code_idx.numel()), device=x.device, dtype=torch.long)
            dist.all_reduce(_tok, op=dist.ReduceOp.SUM)
            self._step_tokens_global = int(_tok.item())
        else:
            self._step_tokens_global = int(code_idx.numel())

        # At this point, batch_usage_mask already represents the cross-GPU step mask
        self._step_unique_global = int(batch_usage_mask.sum().item())

        # Accumulate into the running global usage mask
        self.global_usage_mask.data = torch.logical_or(self.global_usage_mask, batch_usage_mask)
        
        # Update per-rank batch stats and accumulated global stats
        self._last_unique_count = int(unique_codes.numel())
        self._global_unique_count = int(self.global_usage_mask.sum().item())

        # dequantize
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.is_train:
            perplexity, local_perplexity = self.update_codebook(x_norm, code_idx)
        else:
            perplexity, local_perplexity = -1, -1

        # Loss
        commit_loss = self.commit_loss_weight * F.mse_loss(x_norm, x_d.detach())
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"[InferLog {info}] "
                f"Global_UsedCodes={self._global_unique_count} "
                f"Global_Perplexity={perplexity.item() if self.is_train else -1:.4f} "
                f"Batch_Samples={x_d.shape[0]} "
                f"Batch_UniqueCodes={code_idx.unique().numel()} "
                f"CommitLoss={commit_loss.item():.6f} "
                f"Batch_Perplexity={local_perplexity.item() if self.is_train else -1:.4f}"
            )

        # Passthrough
        x_d = x_norm + (x_d - x_norm).detach()

        if self.is_train:
            return x_d, code_idx, dict(loss=commit_loss)
        else:
            return x_d, code_idx, dict(loss=commit_loss)



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
    
    def forward(self, z):
        # import pdb; pdb.set_trace()
        # if not self.init:
        #     with torch.no_grad():
        #         # repeat_num = self.n_e // z.shape[0] + 1
        #         # init_weight = z.repeat(repeat_num, 1)[:self.n_e]
        #         # noise = torch.randn_like(init_weight)
        #         # self.embedding.weight.data.copy_(init_weight + noise)
        #         self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        #     self.init = True

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

        if dist.get_rank() == 0 and self.training:
            num_codes = min_encoding_indices[:, 0].unique().numel() if min_encoding_indices.ndim > 1 else min_encoding_indices.unique().numel()
            print(f"[IBQ] Sequence length={z.shape[0]:4d} | Unique codes={num_codes:4d} | "f"Commit loss={commit_loss.item():.6f}")

        return z_q, min_encoding_indices, dict(loss=commit_loss)

    # def forward(self, z, temp=None, rescale_logits=False, return_logits=False, **kwargs):
    #     assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
    #     assert not rescale_logits, "Only for interface compatible with Gumbel"
    #     assert not return_logits, "Only for interface compatible with Gumbel"

    #     # L2 归一化
    #     if self.l2_norm:
    #         z = F.normalize(z, p=2, dim=-1)
    #         embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
    #     else:
    #         embedding = self.embedding.weight

    #     # 计算距离 d
    #     d = torch.sum(z ** 2, dim=1, keepdim=True) + \
    #         torch.sum(embedding**2, dim=1) - 2 * torch.matmul(z, embedding.t())

    #     if self.training:
    #         # 避免 softmax 数值爆炸
    #         logits = -d / max(self.quantization_temp, 1e-6)
    #         soft_one_hot = F.softmax(logits, dim=1)
    #         min_encoding_indices = soft_one_hot.argmax(dim=1, keepdim=True)

    #         # Hard + straight-through
    #         hard_one_hot = torch.zeros_like(logits).scatter_(1, min_encoding_indices, 1.0)
    #         one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot

    #         # quantized vectors
    #         z_q = torch.matmul(one_hot, self.embedding.weight)
    #         z_q_2 = torch.matmul(hard_one_hot, self.embedding.weight)

    #         # commit + codebook loss
    #         commit_loss = (
    #             torch.mean((z_q - z) ** 2) +
    #             torch.mean((z_q_2.detach() - z) ** 2) +
    #             self.beta * torch.mean((z_q_2 - z.detach()) ** 2)
    #         )
    #         commit_loss = self.codebook_loss_weight * commit_loss
    #     else:
    #         # 推理用
    #         min_encoding_indices = torch.argmin(d, dim=1)
    #         z_q = embedding[min_encoding_indices].view(z.shape)
    #         commit_loss = None

    #     # skip_quantization_prob 可选
    #     if self.training and self.skip_quantization_prob > 0.0:
    #         mask = torch.rand_like(z_q[:, 0:1]) <= self.skip_quantization_prob
    #         z_q = torch.where(mask.expand_as(z_q), z, z_q)

    #     print(min_encoding_indices[:10, 0])

    #     return z_q, min_encoding_indices, dict(loss=commit_loss)

    # def decode(self, indices):
    #     z_q = self.embedding(indices)
    #     return z_q


class ResidualBlock(nn.Module):
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding='same')
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.activate = nn.GELU()
        self.conv2 = nn.Conv3d(channels, channels, 3, padding='same')
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    
    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x + res


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


class Process_in(nn.Module):
    def __init__(self, codebook_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(codebook_dim, codebook_dim),
            nn.GELU(),
            nn.Linear(codebook_dim, codebook_dim),
        )
        self.ln_q = Qwen2RMSNorm(codebook_dim, eps=1e-6)

    def forward(self, x):
        x = self.ln_q(x)
        x = self.mlp(x)
        return x


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
        transformer_in_layers=0,
        transformer_out_layers=2,
        use_transformer=False,
        config=None,
        has_mlp_pre=False
    ):
        super().__init__()
        # self.quantize = IBQ(codebook_size, codebook_dim)
        self.quantize = new_IBQ(codebook_size, codebook_dim, norm=False)
        if use_transformer:
            assert config is not None, "Config must be provided for transformer-based quantization"
            from qwen3_vl import Qwen3_VLVisionBlock
            config.hidden_size = z_channels
            self.transformer_in = nn.ModuleList(
                [Qwen3_VLVisionBlock(config, config._attn_implementation) for _ in range(transformer_in_layers)]
            )
            if has_mlp_pre:
                self.process_in = Process_in(codebook_dim)
            self.transformer_out = nn.ModuleList(
                [Qwen3_VLVisionBlock(config, config._attn_implementation) for _ in range(transformer_out_layers)]
            )

        self.use_transformer = use_transformer
        self.config = config
        self.has_mlp_pre = has_mlp_pre
    
    
    def forward(self, x, cu_seqlens=None, position_embeddings=None, info="vq_final"):
        if self.use_transformer and len(self.transformer_in) > 0:
            for blk in self.transformer_in:
                x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        
        if self.has_mlp_pre:
            x = self.process_in(x)
        
        # print("x min:", x.min().item(), "x max:", x.max().item())
        assert torch.isfinite(x).all(), f"x has NaN/Inf: {x}"

        # quantize
        x, code_idx, codebook_loss = self.quantize(x, info=info)

        if self.use_transformer and len(self.transformer_out) > 0:
            for blk in self.transformer_out:
                x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        if torch.isnan(codebook_loss['loss']).any():
            if not is_initialized() or get_rank() == 0:
                print("embedding:", self.quantize.embedding.weight.norm().item())
                print("x:", x.norm().item())
                import pdb; pdb.set_trace()
            if is_initialized():
                barrier()  # 其他 rank 在这里等 rank=0 调试完

        
        return x, code_idx, codebook_loss
