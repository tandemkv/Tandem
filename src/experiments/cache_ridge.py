"""Fit a cross-layer ridge map with the attention checkpoint's settings."""
import torch


class ThreeLayerRidge:
    def __init__(self, baseline, layers=None):
        self.baseline = baseline
        c = baseline.config
        self.source_width = c['source_heads'] * c['head_dim']
        self.target_width = c['target_heads'] * c['head_dim']
        self.target_mean = baseline.target_mean
        # Keep three distinct layers, four apart, shifting the window at edges.
        center = baseline.layer_map.clamp(4, c['source_layers'] - 5)
        self.layers = torch.stack((center - 4, center, center + 4), dim=-1) if layers is None else layers

    def design(self, source):
        x = self.baseline.standardize_source(source).unflatten(-2, (
            self.baseline.config['source_layers'], self.baseline.config['source_heads']))
        x = x[:, :, self.layers].flatten(-3)
        return torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)

    @torch.no_grad()
    def fit(self, data, stride, ridge):
        """Use the same train positions, normalization and penalty as one-layer ridge."""
        count = 0
        device = self.layers.device
        dim = self.layers.shape[-1] * self.source_width + 1
        shape = (2, len(self.layers), dim)
        xx = torch.zeros((*shape, dim), dtype=torch.float64, device=device)
        xy = torch.zeros((*shape, self.target_width), dtype=torch.float64, device=device)
        for sample in data.train:
            source, target = data.pair(sample, stride)
            x = self.design(source.to(device)).double()
            y = self.baseline.standardize_target(target.to(device)).reshape(
                -1, 2, len(self.layers), self.target_width).double()
            x = x.permute(1, 2, 3, 0).flatten(0, 1)
            y = y.permute(1, 2, 0, 3).flatten(0, 1)
            xx.flatten(0, 1).baddbmm_(x, x.transpose(-2, -1))
            xy.flatten(0, 1).baddbmm_(x, y)
            count += x.shape[-1]
        if count < 2:
            raise ValueError('Insufficient training positions for ridge fitting')
        xx.div_(count).diagonal(dim1=-2, dim2=-1)[..., :-1].add_(ridge)
        xy.div_(count)
        self.weights = torch.linalg.solve(xx, xy).float()
        if not torch.isfinite(self.weights).all():
            raise ValueError('Nonfinite ridge fit')
        return count

    def predict(self, source, affine_only=False):
        y = torch.einsum('brli,rlio->brlo', self.design(source), self.weights)
        return y.reshape(-1, *self.target_mean.shape) * self.baseline.target_std + self.target_mean
