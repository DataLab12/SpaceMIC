import logging
import math
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_
import timm
from torchinfo import summary
from torch.utils.checkpoint import checkpoint

# Heatmap Generation Function
def generate_gaussian_heatmap(boxes, image_size, sigma=10.0):
    B = len(boxes)
    H, W = image_size
    heatmap = torch.zeros(B, 1, H, W, device=boxes[0].device)
    for b in range(B):
        for box in boxes[b]:
            x_min, y_min, x_max, y_max = box.int()
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=box.device),
                torch.arange(W, device=box.device),
                indexing='ij'
            )
            gaussian = torch.exp(-((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2) / (2 * sigma ** 2))
            heatmap[b, 0] = torch.maximum(heatmap[b, 0], gaussian)
    return heatmap.clamp(0, 1)

# MultiLinearLayer
class MultiLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=7, spline_order=3, scale_base=1.0, scale_spline=1.0, base_activation=nn.SiLU):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_weight = nn.Parameter(scale_base * torch.randn(out_features, in_features))
        self.base_activation = base_activation()
        h = 2.0 / grid_size
        grid = torch.linspace(-1 - h * spline_order, 1 + h * spline_order, grid_size + 2 * spline_order + 1)[None, :, None]
        self.register_buffer("grid", grid)
        self.spline_weight = nn.Parameter(scale_spline * torch.randn(out_features, in_features, grid_size + spline_order) * 0.01)

    def forward(self, x):
        base_output = F.linear(x, self.base_weight)
        base_output = self.base_activation(base_output)
        spline_output = torch.zeros_like(base_output)
        for i in range(self.grid_size + self.spline_order):
            spline_output += F.linear(x, self.spline_weight[:, :, i])
        spline_output = spline_output / (self.grid_size + self.spline_order)
        return base_output + spline_output

# DW_bn_relu
class DepthWiseConvBNReLU(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# Enhanced ConvLinearLayer
class ConvLinearLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.05):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or (in_features // 2)
        low_rank_dim = hidden_features // 2
        full_rank_dim = hidden_features
        concat_dim = low_rank_dim + full_rank_dim
        self.dim = in_features
        self.fc1_low = MultiLinearLayer(in_features, low_rank_dim)
        self.fc2_low = MultiLinearLayer(low_rank_dim, low_rank_dim)
        self.fc1_full = MultiLinearLayer(in_features, full_rank_dim)
        self.fc3 = MultiLinearLayer(concat_dim, out_features)
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),
            nn.ReLU(),
            nn.Linear(concat_dim // 2, concat_dim)
        )
        self.channel_scale = nn.Parameter(torch.ones(concat_dim) * 1.0)
        self.dwconv_1 = DepthWiseConvBNReLU(low_rank_dim)
        self.dwconv_2 = DepthWiseConvBNReLU(full_rank_dim)
        self.dwconv_3 = DepthWiseConvBNReLU(out_features)
        self.drop = nn.Dropout(drop)
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, (2.0 / fan_out) ** 0.5)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, heatmap=None):
        B, N, C = x.shape
        assert N == H * W, f"Expected N={H*W}, got {N}"
        residual = x
        if heatmap is not None:
            heatmap_resized = F.interpolate(heatmap, size=(H, W), mode='bilinear', align_corners=False)
            heatmap_flat = heatmap_resized.view(B, 1, -1).transpose(1, 2)
            x = x * (heatmap_flat + 1e-6)
        x_low = self.fc1_low(x.reshape(B * N, C))
        x_low = x_low.reshape(B, N, -1).contiguous()
        x_low = self.dwconv_1(x_low, H, W)
        x_low = self.fc2_low(x_low.reshape(B * N, -1))
        x_low = x_low.reshape(B, N, -1).contiguous()
        x_full = self.fc1_full(x.reshape(B * N, C))
        x_full = x_full.reshape(B, N, -1).contiguous()
        x_full = self.dwconv_2(x_full, H, W)
        x = torch.cat([x_low, x_full], dim=-1)
        x = x * self.channel_scale
        x = self.fusion(x.reshape(B * N, -1)).reshape(B, N, -1)
        x = self.fc3(x.reshape(B * N, -1))
        x = x.reshape(B, N, -1).contiguous()
        x = self.dwconv_3(x, H, W)
        x = x + self.res_scale * residual[:, :, :x.shape[-1]]
        return x

# AttentionConvBlock
class AttentionConvBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = nn.Identity()  # DropPath removed as drop_path=0
        self.norm1 = norm_layer(dim)
        self.qk_reduce = nn.Linear(dim, 32)
        self.qk = nn.Linear(32, 32 * 2)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(32, dim)
        self.self_attn_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.norm2 = norm_layer(dim)
        self.layer = ConvLinearLayer(in_features=dim, hidden_features=dim, drop=drop)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.heatmap_mod = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, (2.0 / fan_out) ** 0.5)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, heatmap=None):
        B, N, C = x.shape
        assert N == H * W, f"Expected N={H*W}, got {N}"
        heatmap_resized = F.interpolate(heatmap, size=(H, W), mode='bilinear', align_corners=False) if heatmap is not None else None
        heatmap_flat = heatmap_resized.view(B, 1, -1).transpose(1, 2) if heatmap is not None else None

        if heatmap_resized is not None:
            heatmap_mod = self.heatmap_mod(heatmap_resized).view(B, C, H * W).transpose(1, 2)
            x = x * heatmap_mod

        x_attn = self.norm1(x)
        x_red = self.qk_reduce(x_attn)
        qk = self.qk(x_red).reshape(B, N, 2, 32).permute(2, 0, 1, 3)
        q, k = qk[0], qk[1]
        attn = (q @ k.transpose(-2, -1)) * (32 ** -0.5)
        if heatmap_flat is not None:
            attn = attn + torch.log(heatmap_flat + 1e-6)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_attn = (attn @ q).transpose(1, 2).reshape(B, N, 32)
        x_attn = self.proj(x_attn)
        x = x + self.drop_path(self.self_attn_scale * x_attn)

        x = x + self.drop_path(self.scale * self.layer(self.norm2(x), H, W, heatmap=heatmap_resized))

        if heatmap_resized is not None:
            heatmap_mod = self.heatmap_mod(heatmap_resized).view(B, C, H * W).transpose(1, 2)
            x = x * heatmap_mod

        return x

# Custom Lightweight CNN Feature Extractor 
class SimpleConvFeatureExtractor(nn.Module):
    def __init__(self, in_chans=3, feature_dims=[96, 192, 384, 768], strides=[4, 8, 16, 32]):
        super().__init__()
        self.feature_dims = feature_dims
        self.strides = strides
        self.stages = nn.ModuleList()
        prev_channels = in_chans

        # Stage 0: stride=4 (512 -> 128)
        self.stages.append(nn.Sequential(
            nn.Conv2d(prev_channels, feature_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dims[0]),
            nn.ReLU(),
            nn.Conv2d(feature_dims[0], feature_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dims[0]),
            nn.ReLU()
        ))
        prev_channels = feature_dims[0]

        # Stage 1: stride=8 (128 -> 64)
        self.stages.append(nn.Sequential(
            nn.Conv2d(prev_channels, feature_dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dims[1]),
            nn.ReLU(),
            nn.Conv2d(feature_dims[1], feature_dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dims[1]),
            nn.ReLU()
        ))
        prev_channels = feature_dims[1]

        # Stage 2: stride=16 (64 -> 32)
        self.stages.append(nn.Sequential(
            nn.Conv2d(prev_channels, feature_dims[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dims[2]),
            nn.ReLU(),
            nn.Conv2d(feature_dims[2], feature_dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dims[2]),
            nn.ReLU()
        ))
        prev_channels = feature_dims[2]

        # Stage 3: stride=32 (32 -> 16)
        self.stages.append(nn.Sequential(
            nn.Conv2d(prev_channels, feature_dims[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dims[3]),
            nn.ReLU(),
            nn.Conv2d(feature_dims[3], feature_dims[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dims[3]),
            nn.ReLU()
        ))

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

# PH_FPNEncoder
class PH_FPNEncoder(nn.Module):
    def __init__(self, in_chans=3):
        super().__init__()
        self.backbone = SimpleConvFeatureExtractor(in_chans=in_chans, feature_dims=[96, 192, 384, 768], strides=[4, 8, 16, 32])
        self.feature_dims = [96, 192, 384, 768]
        self.feature_strides = [4, 8, 16, 32]
        self.stage_indices = [0, 1, 2, 3]
       
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 256, 64, 64)) for _ in self.feature_strides
        ])
        self.patch_embeds = nn.ModuleList([
            nn.Conv2d(dim, min(max(dim, 256), 384), kernel_size=1)
            for dim in self.feature_dims
        ])
        self.feat_projs = nn.ModuleList([
            nn.Conv2d(min(max(dim, 256), 384), 256, kernel_size=1)
            if min(max(dim, 256), 384) != 256 else nn.Identity()
            for dim in self.feature_dims
        ])
        self.atten_conv_blocks = nn.ModuleList([
            AttentionConvBlock(dim=256, drop=0.05, drop_path=0.05)
            for _ in range(len(self.feature_dims))
        ])
      
        self.proj_fpn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ) for _ in self.feature_dims[:-1]
        ])
        self.fpn_top_down = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ) for _ in range(len(self.feature_dims))
        ])
        self.fpn_bottom_up = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ) for _ in self.feature_dims[:-1]
        ])
        self.prompt_modulation = nn.ModuleList([
            nn.Linear(256, 256) for _ in range(len(self.feature_dims))
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, (2.0 / fan_out) ** 0.5)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=.02)

    def forward(self, x, sparse_emb=None, boxes=None):
        B, C, H, W = x.shape
        assert H == W, f"Expected square input, got {H}x{W}"
        assert H % 32 == 0, f"Input size {H} must be divisible by 32"
        spatial_sizes = [H // s for s in self.feature_strides]
        
        heatmap = generate_gaussian_heatmap(boxes, (H, W)) if boxes is not None else None
        
        features = self.backbone(x)
        selected_features = features
        if len(selected_features) != len(self.feature_dims):
            raise ValueError(f"Expected {len(self.feature_dims)} features, got {len(selected_features)}")
        
        
        atten_conv_features = []
        pos_enc = []
        for i, (feat, embed, feat_proj, pos, atten_conv_block) in enumerate(zip(
            selected_features, self.patch_embeds, self.feat_projs, self.pos_embeds, self.atten_conv_blocks
        )):
            B, C, H_f, W_f = feat.shape
            assert C == self.feature_dims[i], f"Feature dim mismatch at stage {i}: expected {self.feature_dims[i]}, got {C}"
            assert H_f == spatial_sizes[i], f"Feature height mismatch at stage {i}: expected {spatial_sizes[i]}, got {H_f}"
            feat = embed(feat)
            feat = feat_proj(feat)
            C_new = 256
            feat = feat.flatten(2).transpose(1, 2)
            pos = F.interpolate(pos, size=(H_f, W_f), mode='bilinear', align_corners=False)
            pos = pos.expand(B, -1, -1, -1)
            pos_flat = pos.flatten(2).transpose(1, 2)
            feat = feat + pos_flat
            feat = checkpoint(atten_conv_block, feat, H_f, W_f, heatmap, use_reentrant=False)
            feat = feat.transpose(1, 2).view(B, C_new, H_f, W_f)
            # feat = inter_conv(feat)
            atten_conv_features.append(feat)
            pos = pos_flat.transpose(1, 2).view(B, 256, H_f, W_f)
            pos_enc.append(pos)
        
        if sparse_emb is not None:
            assert sparse_emb.shape[-1] == 256, f"Expected sparse_emb dim 256, got {sparse_emb.shape[-1]}"
            for i, (feat, mod) in enumerate(zip(atten_conv_features, self.prompt_modulation)):
                B, C, H_f, W_f = feat.shape
                mod_emb = mod(sparse_emb).view(B, C, 1, 1)
                atten_conv_features[i] = feat * mod_emb
        fpn_features = []
        top_down = self.fpn_top_down[-1](atten_conv_features[-1])
        fpn_features.append(top_down)
        for i in range(len(atten_conv_features)-2, -1, -1):
            top_down = F.interpolate(top_down, size=atten_conv_features[i].shape[-2:], mode='bilinear', align_corners=False)
            top_down = self.fpn_top_down[i](atten_conv_features[i] + top_down)
            fpn_features.append(top_down)
        fpn_features = fpn_features[::-1]
        fused_features = [fpn_features[0]]
        for i in range(len(fpn_features)-1):
            bottom_up = self.fpn_bottom_up[i](fused_features[-1])
            bottom_up = F.interpolate(bottom_up, size=fpn_features[i+1].shape[-2:], mode='bilinear', align_corners=False)
            fused = fpn_features[i+1] + bottom_up
            fused_features.append(fused)
        fpn_feats = [proj(feat) for feat, proj in zip(fused_features[:-1], self.proj_fpn)]
        vision_features = fused_features[-1]
        return {
            "backbone_fpn": fpn_feats,
            "vision_features": vision_features,
            "vision_pos_enc": pos_enc,
            "atten_conv_features": fused_features
        }

# PromptEncoder
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("positional_encoding", torch.zeros(1, embed_dim, 64, 64))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        self.box_embed = nn.Linear(4, embed_dim)
        self.label_embed = nn.Linear(1, embed_dim)
        self.not_a_box_embed = nn.Parameter(torch.zeros(embed_dim))
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*2, dropout=0.05, activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dense_embed = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        self.transformer = self.transformer.to(dtype=torch.float32)
        self.transformer_encoder = self.transformer_encoder.to(dtype=torch.float32)

    def forward(self, boxes, labels, image_size):
        B = len(boxes)
        assert image_size % 32 == 0, f"Image size {image_size} must be divisible by 32"
        sparse_embs = []
        for i in range(B):
            box = boxes[i]
            label = labels[i].unsqueeze(-1)
            N_i = box.shape[0]
            if N_i == 0 or (N_i == 1 and label[0] == 0):
                emb = self.not_a_box_embed.unsqueeze(0)
            else:
                box_normalized = box / (image_size - 1)
                box_emb = self.box_embed(box_normalized)
                label_emb = self.label_embed(label)
                emb = box_emb + label_emb
                # Disable autocast and ensure float32
                with torch.cuda.amp.autocast(enabled=False):
                    emb = emb.to(dtype=torch.float32)
                    emb = self.transformer_encoder(emb.unsqueeze(0))
                emb = self.pool(emb.transpose(1, 2)).squeeze(-1)
            sparse_embs.append(emb)

        sparse_emb = torch.cat(sparse_embs, dim=0)
        dense_emb = sparse_emb.view(B, self.embed_dim, 1, 1)
        target_size = image_size // 32
        pos_enc = F.interpolate(self.positional_encoding, size=(target_size, target_size), mode='bilinear', align_corners=False)
        dense_emb = F.interpolate(dense_emb, size=(target_size, target_size), mode='bilinear', align_corners=False)
        dense_emb = dense_emb + pos_enc
        dense_emb = self.dense_embed(dense_emb)
        return sparse_emb, dense_emb

# Decoder
class Decoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim // 2),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),
            nn.Sequential(
                nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim // 4),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),
            nn.Sequential(
                nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim // 4),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ),
            nn.Sequential(
                nn.Conv2d(embed_dim // 4, embed_dim // 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim // 8),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
        ])
        self.fpn_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1),
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1),
            nn.Conv2d(embed_dim, embed_dim // 8, kernel_size=1)
        ])
        self.final_conv = nn.Conv2d(embed_dim // 8, 1, kernel_size=1)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, vision_features, sparse_emb, dense_emb, atten_conv_features, image_size, heatmap=None):
        B, C, H, W = vision_features.shape
        assert C == 256, f"Expected vision_features channels 256, got {C}"
        assert dense_emb.shape == (B, 256, image_size//32, image_size//32), f"Expected dense_emb shape (B, 256, {image_size//32}, {image_size//32}), got {dense_emb.shape}"
        assert sparse_emb.shape == (B, 256), f"Expected sparse_emb shape (B, 256), got {sparse_emb.shape}"
        sparse_emb = sparse_emb.view(B, C, 1, 1).expand(-1, -1, H, W)
        x = vision_features + dense_emb + sparse_emb
        num_upsample = int(math.log2(image_size // (image_size // 32)))
        for i, (layer, fpn_proj) in enumerate(zip(self.upsample_layers[:num_upsample], self.fpn_projs[:num_upsample])):
            x = layer(x)
            if i < len(atten_conv_features):
                target_size = x.shape[-2:]
                fpn = fpn_proj(F.interpolate(atten_conv_features[-(i+1)], size=target_size, mode='bilinear', align_corners=False))
                x = x + fpn
        mask_logits = self.final_conv(x)
        
        if heatmap is not None:
            mask_logits_sigmoid = mask_logits.sigmoid()
            grad_x = torch.abs(mask_logits_sigmoid[:, :, :, 1:] - mask_logits_sigmoid[:, :, :, :-1])
            grad_y = torch.abs(mask_logits_sigmoid[:, :, 1:, :] - mask_logits_sigmoid[:, :, :-1, :])
            grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='constant', value=0)
            grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='constant', value=0)
            uncertainty = (grad_x + grad_y).clamp(0, 1)
            heatmap_resized = F.interpolate(heatmap, size=(mask_logits.shape[-2], mask_logits.shape[-1]), 
                                          mode='bilinear', align_corners=False)
            refine_input = torch.cat([mask_logits, uncertainty * heatmap_resized], dim=1)
            refine_offset = self.refine_conv(refine_input)
            mask_logits = mask_logits + refine_offset
        
        return mask_logits

# SegmentationModel
class SegmentationModel(nn.Module):
    def __init__(self, in_chans=3):
        super().__init__()
        self.encoder = PH_FPNEncoder(in_chans=in_chans)
        self.prompt_encoder = PromptEncoder(embed_dim=256)
        self.decoder = Decoder(embed_dim=256)
        self.to(dtype=torch.float32)

    def forward(self, image, boxes, labels):
        B, C, H, W = image.shape
        assert H == W, f"Expected square input, got {H}x{W}"
        assert H % 32 == 0, f"Input size {H} must be divisible by 32"
        sparse_emb, dense_emb = self.prompt_encoder(boxes, labels, image_size=H)
        heatmap = generate_gaussian_heatmap(boxes, (H, W))
        encoder_output = self.encoder(image, sparse_emb=sparse_emb, boxes=boxes)
        mask_logits = self.decoder(
            encoder_output["vision_features"],
            sparse_emb,
            dense_emb,
            encoder_output["atten_conv_features"],
            image_size=H,
            heatmap=heatmap
        )
        mask_logits = F.interpolate(mask_logits, size=(H, W), mode='bilinear', align_corners=False)
        return mask_logits
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegmentationModel(in_chans=3).to(device)

# Architecture summary
# Dummy inputs
image = torch.randn(1, 3, 512, 512).to(device)
boxes = [torch.tensor([[25, 25, 50, 50]], dtype=torch.float32).to(device)]
labels = [torch.tensor([1], dtype=torch.float32).to(device)]

try:
    with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
        summary(model, input_data=(image, boxes, labels), device=device)
except RuntimeError as e:
    print(f"CUDA error: {e}. Falling back to CPU for summary.")
    device = torch.device("cpu")
    model = model.to(device)
    image = image.to(device)
    boxes = [box.to(device) for box in boxes]
    labels = [label.to(device) for label in labels]
    with torch.amp.autocast(device_type='cpu'):
        summary(model, input_data=(image, boxes, labels), device=device)