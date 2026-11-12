"""Per-boundary, model-conditioned cross-attention residual over a ridge control."""
import torch
from torch import nn


class CacheAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        sl, sh = config['source_layers'], config['source_heads']
        tl, th = config['target_layers'], config['target_heads']
        dim, width = config['head_dim'], config['width']
        self.register_buffer('source_mean', torch.zeros(2, sl * sh, dim))
        self.register_buffer('source_std', torch.ones(2, sl * sh, dim))
        self.register_buffer('target_mean', torch.zeros(2, tl * th, dim))
        self.register_buffer('target_std', torch.ones(2, tl * th, dim))
        self.register_buffer('ridge', torch.zeros(2, tl, sh * dim + 1, th * dim))
        self.register_buffer('layer_map', torch.linspace(0, sl - 1, tl).round().long())
        self.model_id = nn.Embedding(2, width)
        self.layer_id = nn.Embedding(max(sl, tl), width)
        self.head_id = nn.Embedding(max(sh, th), width)
        self.role_id = nn.Embedding(2, width)
        self.depth = nn.Linear(1, width, bias=False)
        self.source_projection = nn.Linear(dim, width)
        self.query_projection = nn.Linear(dim, width)
        self.source_norm = nn.LayerNorm(width)
        self.query_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, config['attention_heads'], batch_first=True, dropout=0)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.output = nn.Linear(width, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def metadata(self, layers, heads, model, device):
        layer = torch.arange(layers, device=device).repeat_interleave(heads)
        head = torch.arange(heads, device=device).repeat(layers)
        depth = layer.float().unsqueeze(-1) / max(1, layers - 1)
        common = self.layer_id(layer) + self.head_id(head) + self.depth(depth) + self.model_id.weight[model]
        return common[None] + self.role_id.weight[:, None]

    def standardize_source(self, source):
        return (source - self.source_mean) / self.source_std

    def standardize_target(self, target):
        return (target - self.target_mean) / self.target_std

    def affine(self, normalized_source):
        c = self.config
        x = normalized_source.reshape(-1, 2, c['source_layers'], c['source_heads'] * c['head_dim'])
        x = x[:, :, self.layer_map]
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        y = torch.einsum('brli,rlio->brlo', x, self.ridge)
        return y.reshape(-1, 2, c['target_layers'] * c['target_heads'], c['head_dim'])

    def forward(self, source, affine_only=False):
        c = self.config
        x = self.standardize_source(source)
        baseline = self.affine(x)
        if affine_only:
            return baseline
        src_meta = self.metadata(c['source_layers'], c['source_heads'], c['source_id'], source.device)
        dst_meta = self.metadata(c['target_layers'], c['target_heads'], c['target_id'], source.device)
        # Batch contains independent causal text positions and K/V roles. No
        # attention edge can cross positions or mix key and value observations.
        src = (self.source_projection(x) + src_meta).flatten(0, 1)
        q = (self.query_projection(baseline) + dst_meta).flatten(0, 1)
        src = self.source_norm(src)
        attended = self.attention(self.query_norm(q), src, src, need_weights=False)[0]
        h = q + attended
        residual = self.output(h + self.ff(h)).reshape_as(baseline)
        return baseline + residual

    def predict(self, source, affine_only=False):
        return self(source, affine_only) * self.target_std + self.target_mean
