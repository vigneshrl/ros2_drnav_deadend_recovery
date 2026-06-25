import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np


class PointNetEncoder(nn.Module):
    def __init__(self, input_channels=3, output_channels=512, num_points=4096):
        super(PointNetEncoder, self).__init__()
        self.num_points = num_points
        
        self.input_transform_net = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        self.input_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_channels * input_channels)
        )
        
        self.feature_extraction = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256, output_channels, 1),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )
        
        self.global_features = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x):
        if x.size(2) > self.num_points:
            idx = torch.randperm(x.size(2), device=x.device)[:self.num_points]
            x = x[:, :, idx]
        elif x.size(2) < self.num_points:
            idx = torch.randint(0, x.size(2), (self.num_points - x.size(2),), device=x.device)
            padding = x[:, :, idx]
            x = torch.cat([x, padding], dim=2)
            
        batch_size = x.size(0)
        
        trans_feat = self.input_transform_net(x)
        trans_feat = trans_feat.view(batch_size, -1)
        trans_mat = self.input_fc(trans_feat).view(batch_size, 3, 3)
        
        x = torch.bmm(x.transpose(1, 2), trans_mat).transpose(1, 2)
        
        point_features = self.feature_extraction(x)
        
        global_features = self.global_features(point_features)
        global_features = global_features.view(batch_size, -1)
        
        return global_features, point_features


class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512, freeze_early=True):
        super(ImageEncoder, self).__init__()
        eff = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        self.backbone = eff.features  # outputs [B, 1280, 7, 7] for 224x224 input

        if freeze_early:
            for i, block in enumerate(self.backbone):
                if i < 5:
                    for param in block.parameters():
                        param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Conv2d(1280, output_dim, kernel_size=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        features = self.backbone(x)  # [B, 1280, H, W]
        features = self.projection(features)  # [B, output_dim, H, W]
        global_features = F.adaptive_avg_pool2d(features, 1).view(x.size(0), -1)
        return global_features, features


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(context)
        
        return output, attn_weights


class CrossModalFusion(nn.Module):
    """Fuses image spatial tokens with LiDAR point tokens via cross-attention."""
    def __init__(self, embed_dim=512, num_heads=4, dropout=0.2):
        super(CrossModalFusion, self).__init__()
        
        self.img_to_lidar_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.lidar_to_img_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff_img = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.ff_lidar = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        
    def forward(self, img_tokens, lidar_tokens):
        # img_tokens: [B, S_img, D], lidar_tokens: [B, S_lidar, D]
        attn_out, _ = self.img_to_lidar_attn(lidar_tokens, img_tokens, img_tokens)
        lidar_tokens = self.norm1(lidar_tokens + attn_out)

        attn_out, _ = self.lidar_to_img_attn(img_tokens, lidar_tokens, lidar_tokens)
        img_tokens = self.norm2(img_tokens + attn_out)

        img_tokens = self.norm3(img_tokens + self.ff_img(img_tokens))
        lidar_tokens = self.norm4(lidar_tokens + self.ff_lidar(lidar_tokens))

        return img_tokens, lidar_tokens


class DeadEndDetectionModel(nn.Module):
    def __init__(self, img_embed_dim=512, lidar_embed_dim=512, fusion_dim=512, num_heads=4):
        super(DeadEndDetectionModel, self).__init__()
        
        # Shared encoders across all 3 views
        self.img_encoder = ImageEncoder(output_dim=img_embed_dim, freeze_early=True)
        self.lidar_encoder = PointNetEncoder(output_channels=lidar_embed_dim)

        self.img_projection = nn.Linear(img_embed_dim, fusion_dim)
        self.lidar_projection = nn.Linear(lidar_embed_dim, fusion_dim)
        
        # Per-view fusion (separate so each view can learn its own cross-modal patterns)
        self.front_fusion = CrossModalFusion(fusion_dim, num_heads, dropout=0.2)
        self.right_fusion = CrossModalFusion(fusion_dim, num_heads, dropout=0.2)
        self.left_fusion = CrossModalFusion(fusion_dim, num_heads, dropout=0.2)

        # Number of spatial tokens from image encoder (7x7 for EfficientNet-B0 with 224 input)
        self.num_lidar_tokens = 16

        self.lidar_tokenizer = nn.Sequential(
            nn.Linear(lidar_embed_dim, fusion_dim * self.num_lidar_tokens),
            nn.GELU(),
        )

        self.integration = nn.Sequential(
            nn.Linear(fusion_dim * 6, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.path_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3),
        )

        self.dead_end_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
        nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        self.direction_regressor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 9),
        )

        # --- Single-view head (for single-camera inference) ---
        # Takes only 1 view's fused features (img_pooled + lidar_pooled = fusion_dim * 2)
        self.integration_single = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.path_classifier_single = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        self.dead_end_classifier_single = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        self.direction_regressor_single = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def _encode_view(self, img, lidar, fusion_module):
        """Encode a single view (shared backbone) and fuse modalities."""
        batch_size = img.size(0)

        img_global, img_spatial = self.img_encoder(img)  # [B, D], [B, D, H, W]
        lidar_global, _ = self.lidar_encoder(lidar)  # [B, D]

        # Create spatial tokens from image features: flatten HxW -> sequence
        B, C, H, W = img_spatial.shape
        img_tokens = img_spatial.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        img_tokens = self.img_projection(img_tokens)  # [B, H*W, fusion_dim]

        # Create pseudo-tokens from LiDAR global features
        lidar_tokens = self.lidar_tokenizer(lidar_global)  # [B, fusion_dim * num_tokens]
        lidar_tokens = lidar_tokens.view(B, self.num_lidar_tokens, -1)  # [B, num_tokens, fusion_dim]

        # Cross-modal fusion on spatial tokens
        img_fused, lidar_fused = fusion_module(img_tokens, lidar_tokens)

        # Pool fused tokens to single vector per modality
        img_pooled = img_fused.mean(dim=1)  # [B, fusion_dim]
        lidar_pooled = lidar_fused.mean(dim=1)  # [B, fusion_dim]

        return img_pooled, lidar_pooled
        
    def forward(self, front_img, right_img, left_img, front_lidar, right_lidar, left_lidar):
        front_img_f, front_lidar_f = self._encode_view(front_img, front_lidar, self.front_fusion)
        right_img_f, right_lidar_f = self._encode_view(right_img, right_lidar, self.right_fusion)
        left_img_f, left_lidar_f = self._encode_view(left_img, left_lidar, self.left_fusion)

        all_fused = torch.cat([
            front_img_f, front_lidar_f,
            right_img_f, right_lidar_f,
            left_img_f, left_lidar_f,
        ], dim=1)
        
        integrated_feats = self.integration(all_fused)

        path_logits = self.path_classifier(integrated_feats)
        path_probs = torch.sigmoid(path_logits)

        dead_end_logits = self.dead_end_classifier(integrated_feats)

        batch_size = front_img.size(0)
        direction_vectors = self.direction_regressor(integrated_feats).view(batch_size, 3, 3)
        
        return {
            'path_status': path_probs,
            'is_dead_end': torch.sigmoid(dead_end_logits),
            'path_logits': path_logits,
            'dead_end_logits': dead_end_logits,  
            'direction_vectors': direction_vectors,
        }

    def forward_single_view(self, img, lidar):
        """Single-camera forward pass. Uses front_fusion + single-view heads."""
        img_pooled, lidar_pooled = self._encode_view(img, lidar, self.front_fusion)

        fused = torch.cat([img_pooled, lidar_pooled], dim=1)  # [B, fusion_dim * 2]
        integrated = self.integration_single(fused)  # [B, fusion_dim]

        path_logits = self.path_classifier_single(integrated)  # [B, 1]
        path_probs = torch.sigmoid(path_logits)

        dead_end_logits = self.dead_end_classifier_single(integrated)  # [B, 1]

        batch_size = img.size(0)
        direction_vectors = self.direction_regressor_single(integrated).view(batch_size, 1, 3)

        return {
            'path_status': path_probs,
            'is_dead_end': torch.sigmoid(dead_end_logits),
            'path_logits': path_logits,
            'dead_end_logits': dead_end_logits,
            'direction_vectors': direction_vectors,
        }
