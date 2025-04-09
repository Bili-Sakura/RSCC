import torch
from torch import Tensor, nn
import torch.nn as nn
import numpy as np

import math
from typing import Any, Optional, Tuple, Type
from torch.nn.init import trunc_normal_
from torch.nn import functional as F


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

class SinePositionalEncoding(nn.Module):
    def __init__(self,
                 num_feats: int,
                 temperature: int = 10000,
                 normalize: bool = False,
                 scale: float = 2 * math.pi,
                 eps: float = 1e-6,
                 offset: float = 0.) -> None:
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: Tensor, input: Optional[Tensor] = None) -> Tensor:
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
            input (Tensor, optional): Input image/feature Tensor.
                Shape [bs, c, h, w]

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        assert not (mask is None and input is None)

        if mask is not None:
            B, H, W = mask.size()
            device = mask.device
            # For convenience of exporting to ONNX,
            # it's required to convert
            # `masks` from bool to int.
            mask = mask.to(torch.int)
            not_mask = 1 - mask  # logical_not
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
        else:
            # single image or batch image with no padding
            B, _, H, W = input.shape
            device = input.device
            x_embed = torch.arange(
                1, W + 1, dtype=torch.float32, device=device)
            x_embed = x_embed.view(1, 1, -1).repeat(B, H, 1)
            y_embed = torch.arange(
                1, H + 1, dtype=torch.float32, device=device)
            y_embed = y_embed.view(1, -1, 1).repeat(B, 1, W)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class ChangeFeatureExtractor(nn.Module):
    def __init__(self, in_features):
        super(ChangeFeatureExtractor, self).__init__()
        self.linear_mapping = nn.Linear(2*in_features, in_features)
        self.proj = nn.Linear(3*in_features, in_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, F1, F2):
        # Apply the element-wise sigmoid operation
        sigma_F1 = self.sigmoid(self.linear_mapping(torch.cat((F1, F2), dim=-1)))
        sigma_F2 = self.sigmoid(self.linear_mapping(torch.cat((F2, F1), dim=-1)))
        # Compute F1' and F2'
        F1_cc_prime2 = F1 * sigma_F1
        F2_cc_prime2 = F2 * sigma_F2
        # Compute ΔFcc
        delta_Fcc = torch.cat((F1_cc_prime2, F2_cc_prime2, F1_cc_prime2 - F2_cc_prime2), dim=-1)
        delta_Fcc = self.proj(delta_Fcc)
        # Output ΔFcc
        return delta_Fcc
    
class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        qkv_upsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // qkv_upsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int = 2304,
        activation: Type[nn.Module] = nn.GELU,
        attention_upsample_rate: int = 1,
        skip_first_layer_add: bool = False,
    ) -> None:
        super().__init__()
        num_heads = embedding_dim // 128

        self.change_self_attn = Attention(embedding_dim, num_heads, qkv_upsample_rate=attention_upsample_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_change_to_imageABC = Attention(
            embedding_dim, num_heads, qkv_upsample_rate=attention_upsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_imageAB_to_change = Attention(
            embedding_dim, num_heads, qkv_upsample_rate=attention_upsample_rate
        )

        self.skip_first_layer_add = skip_first_layer_add

        # self.pe_emb = PositionEmbeddingRandom(embedding_dim // 2)
        self.image_a_flag_embed = nn.Embedding(1, embedding_dim)
        self.image_b_flag_embed = nn.Embedding(1, embedding_dim)
        self.change_flag_embed = nn.Embedding(1, embedding_dim)

    def forward(
        self, 
        queries: Tensor,
        change_emb: Tensor, 
        imageA_emb: Tensor, 
        imageB_emb: Tensor,
        pe: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        bs, n_tokens, embedding_dim = queries.shape
        pe = pe.expand(bs, n_tokens, embedding_dim)
        image_a_flag_embed = self.image_a_flag_embed.weight.expand(bs, n_tokens, embedding_dim)
        image_b_flag_embed = self.image_b_flag_embed.weight.expand(bs, n_tokens, embedding_dim)
        change_flag_embed = self.change_flag_embed.weight.expand(bs, n_tokens, embedding_dim)

        # >>>>>>>>>>>>>>>>>>>>>> change self attention <<<<<<<<<<<<<<<<<<<<<<
        if self.skip_first_layer_add:
            attn_out = self.change_self_attn(q=queries+pe, k=queries+pe, v=queries)
        else:
            queries = queries + change_emb
            attn_out = self.change_self_attn(q=queries+pe, k=queries+pe, v=queries)
        queries = queries + attn_out
        queries = self.norm1(queries)

        # >>>>>>>>>>>>>>>>>>>>>> change as query, imageA||imageB||change as key/value <<<<<<<<<<<<<<<<<<<<<<
        change_queries_with_pe_flag = queries + pe + change_flag_embed
        keys_imageA_imageB_change_emb_with_pe_flag = torch.cat((imageA_emb+pe+image_a_flag_embed, imageB_emb+pe+image_b_flag_embed, queries+pe+change_flag_embed), dim=1)  # shape: [bs, n_tokens*3, emb_dim]
        values_imageA_imageB_change_emb = torch.cat((imageA_emb, imageB_emb, queries), dim=1)  # shape: [bs, n_tokens*3, emb_dim]

        attn_out = self.cross_attn_change_to_imageABC(q=change_queries_with_pe_flag, k=keys_imageA_imageB_change_emb_with_pe_flag, v=values_imageA_imageB_change_emb)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # >>>>>>>>>>>>>>>>>>>>>> change MLP block <<<<<<<<<<<<<<<<<<<<<<
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        change_emb = queries

        # >>>>>>>>>>>>>>>>>>>>>> imageA or imageB as query, imageA or imageB||change as key/value <<<<<<<<<<<<<<<<<<<<<<
        imageA_queries_with_pe_flag = imageA_emb + pe + image_a_flag_embed
        imageB_queries_with_pe_flag = imageB_emb + pe + image_b_flag_embed
        keys_imageA_change_emb_with_pe_flag = torch.cat((imageA_emb+pe+image_a_flag_embed, queries+pe+change_flag_embed), dim=1)  # shape: [bs, n_tokens*2, emb_dim]
        keys_imageB_change_emb_with_pe_flag = torch.cat((imageB_emb+pe+image_b_flag_embed, queries+pe+change_flag_embed), dim=1)  # shape: [bs, n_tokens*2, emb_dim]
        values_imageA_change_emb = torch.cat((imageA_emb, queries), dim=1)  # shape: [bs, n_tokens*2, emb_dim]
        values_imageB_change_emb = torch.cat((imageB_emb, queries), dim=1)  # shape: [bs, n_tokens*2, emb_dim]

        q = torch.cat((imageA_queries_with_pe_flag, imageB_queries_with_pe_flag), dim=0)
        k = torch.cat((keys_imageA_change_emb_with_pe_flag, keys_imageB_change_emb_with_pe_flag), dim=0)
        v = torch.cat((values_imageA_change_emb, values_imageB_change_emb), dim=0)

        attn_out = self.cross_attn_imageAB_to_change(q=q, k=k, v=v)
        q = q + attn_out
        image_A_emb_refined, image_B_emb_refined = torch.split(q, q.shape[0]//2, dim=0)

        return image_A_emb_refined, image_B_emb_refined, change_emb

class ChangeAwareTokenReduce(nn.Module):
    def __init__(
            self,
            raw_grid = 27,
            scale_factor = 3,
            embedding_dim = 1152,
    ):
        super(ChangeAwareTokenReduce, self).__init__()

        self.scale_factor = scale_factor
        self.raw_grid = raw_grid
        self.grid_size = raw_grid // scale_factor
        self.num_queries = self.grid_size ** 2

        self.feature_map_level_gate = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

        self.change_for_key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.cross_attn = nn.MultiheadAttention(embedding_dim, embedding_dim // 128)

    # def divide_feature(self, x, kernel_size, token_num, N, c):
    #     # From TokenPacker https://github.com/CircleRadon/TokenPacker/blob/main/llava/model/multimodal_projector/builder.py
    #     h = w = int(token_num**0.5)
    #     reshape_x = x.reshape(h, w, N, c).reshape(h//kernel_size, kernel_size, w, N, c)
    #     reshape_x = reshape_x.permute(0,2,1,3,4)
    #     reshape_x = reshape_x.reshape(h//kernel_size, w//kernel_size, kernel_size, kernel_size, N, c)
    #     reshape_x = reshape_x.permute(0,1,3,2,4,5).reshape(h//kernel_size, w//kernel_size, kernel_size*kernel_size, N, c)
    #     reshape_x = reshape_x.permute(2,0,1,3,4).reshape(kernel_size*kernel_size, -1, c)
    #     return reshape_x
    
    def multi_level_feature_combine(self, A_feats, B_feats, diff_feats):
        fused_A_feat_sum = 0
        fused_B_feat_sum = 0
        fused_change_feat_sum = 0
        total_weight_sum = 0
        # 逐层处理多尺度特征
        for A_feat, B_feat, diff_feat in zip(A_feats, B_feats, diff_feats):
            # 假设特征维度为 [bs, n_tokens, emb_dim]
            bs, n_tokens, emb_dim = A_feat.shape
            # 拼接 A, B, 和差异特征
            gate_input = torch.cat([A_feat, B_feat, diff_feat], dim=-1)  # [bs, n_tokens, emb_dim*3]
            # 在 token 维度上进行平均池化，汇总信息
            gate_input_pooled = gate_input.mean(dim=1)  # [bs, emb_dim*3]
            # 通过门控网络计算全局权重
            gate_value = self.feature_map_level_gate(gate_input_pooled)  # [bs, 1]
            # print(gate_value)
            # 将差异特征和 A/B 特征分别根据全局权重加权
            weighted_A_feat = A_feat * gate_value.unsqueeze(1)  # [bs, n_tokens, emb_dim]
            weighted_B_feat = B_feat * gate_value.unsqueeze(1)
            weighted_change_feat = diff_feat * gate_value.unsqueeze(1)
            # 累加加权后的特征
            fused_A_feat_sum += weighted_A_feat
            fused_B_feat_sum += weighted_B_feat
            fused_change_feat_sum += weighted_change_feat
            total_weight_sum += gate_value.unsqueeze(1)  # 累加权重总和
        # 归一化权重后的特征
        fused_A_feat = fused_A_feat_sum / (total_weight_sum + 1e-6)  # 避免除零
        fused_B_feat = fused_B_feat_sum / (total_weight_sum + 1e-6)
        fused_change_feat = fused_change_feat_sum / (total_weight_sum + 1e-6)
        return fused_A_feat, fused_B_feat, fused_change_feat

    def forward(self, imageA_final_stage_emb, imageB_final_stage_emb, imageA_emb_list, imageB_emb_list, change_emb_list):
        imageA_emb_combined, imageB_emb_combined, change_emb_combined = self.multi_level_feature_combine(imageA_emb_list, imageB_emb_list, change_emb_list)
        if self.scale_factor == 1:
            return imageA_emb_combined + imageA_final_stage_emb, imageB_emb_combined + imageB_final_stage_emb, change_emb_combined

        # I feel very sory, you can not understand this part of code
        # If you understand this part, It is very welcome to continue research.

        # # >>>>>>>>>>>>>>>>>>>>>>>>>> Image A <<<<<<<<<<<<<<<<<<<<<<<<<<<
        # bs, token_nums, emb_dim = imageA_final_stage_emb.shape
        # query = F.interpolate(imageA_final_stage_emb.reshape(bs, self.raw_grid,self.raw_grid, -1).permute(0,3,1,2), size=(self.grid_size, self.grid_size), mode='bilinear').permute(0,2,3,1)  # [bs, grid_size x grid_size, emb_dim]
        # query = query.reshape(query.shape[0], -1, query.shape[-1])  # [bs, n_tokens, emb_dim]
        # reshape_query = self.divide_feature(query, 1, self.num_queries, bs, emb_dim)  # [1, num_queries x bs, emb_dim]

        # if self.version == "v1":
        #     key = imageB_emb_combined + self.change_for_key_proj(change_emb_combined)
        # elif self.version == "v2":
        #     key = imageB_emb_combined + change_emb_combined
        # value = imageA_emb_combined
        # reshape_key = self.divide_feature(key, self.scale_factor, token_nums, bs, emb_dim)  # [scale_facto ** 2, num_queries x bs, emb_dim]
        # reshape_value = self.divide_feature(value, self.scale_factor, token_nums, bs, emb_dim)  # [scale_facto ** 2, num_queries x bs, emb_dim]

        # imageA_out = self.cross_attn(reshape_query, reshape_key, reshape_value)[0]  # [1, num_queries x bs, emb_dim]
        # imageA_out = imageA_out.reshape(self.num_queries, bs, -1).permute(1, 0, 2)  # -> [num_queries, bs, emb_dim] -> [bs, num_queries, emb_dim]

        # # >>>>>>>>>>>>>>>>>>>>>>>>>> Image B <<<<<<<<<<<<<<<<<<<<<<<<<<<
        # bs, token_nums, emb_dim = imageB_final_stage_emb.shape
        # query = F.interpolate(imageB_final_stage_emb.reshape(bs, self.raw_grid,self.raw_grid, -1).permute(0,3,1,2), size=(self.grid_size, self.grid_size), mode='bilinear').permute(0,2,3,1)  # [bs, grid_size x grid_size, emb_dim]
        # query = query.reshape(query.shape[0], -1, query.shape[-1])  # [bs, n_tokens, emb_dim]
        # reshape_query = self.divide_feature(query, 1, self.num_queries, bs, emb_dim)  # [1, num_queries x bs, emb_dim]

        # if self.version == "v1":
        #     key = imageB_emb_combined + self.change_for_key_proj(change_emb_combined)
        # elif self.version == "v2":
        #     key = imageB_emb_combined + change_emb_combined
        # value = imageB_emb_combined
        # reshape_key = self.divide_feature(key, self.scale_factor, token_nums, bs, emb_dim)  # [scale_facto ** 2, num_queries x bs, emb_dim]
        # reshape_value = self.divide_feature(value, self.scale_factor, token_nums, bs, emb_dim)  # [scale_facto ** 2, num_queries x bs, emb_dim]

        # imageB_out = self.cross_attn(reshape_query, reshape_key, reshape_value)[0]  # [1, num_queries x bs, emb_dim]
        # imageB_out = imageB_out.reshape(self.num_queries, bs, -1).permute(1, 0, 2)  # -> [num_queries, bs, emb_dim] -> [bs, num_queries, emb_dim]

        # return imageA_out, imageB_out, change_emb_combined

class CC_expert(nn.Module):
    def __init__(
        self,
        model_args,
        **kwargs
    ) -> None:
        super().__init__()

        model_args = getattr(model_args, "cc_expert_args", {})
        downsample_ratio = model_args.get("downsample_ratio", 1)
        per_expert_depth = model_args.get("per_expert_depth", 2)
        embedding_dim = model_args.get("embedding_dim", 1152)
        mlp_dim = model_args.get("mlp_dim", 1152*4)
        activation = model_args.get("activation", nn.GELU)
        attention_upsample_rate = model_args.get("attention_upsample_rate", 1)

        self.downsample_ratio = downsample_ratio
        self.per_expert_depth = per_expert_depth
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim

        self.layers = nn.ModuleList()

        self.expert_change_feature_extractor_model = ChangeFeatureExtractor(embedding_dim)
        for i in range(per_expert_depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_upsample_rate=attention_upsample_rate,
                    skip_first_layer_add=(i == 0),
                )
            )
        
        self.position_emb_model = SinePositionalEncoding(num_feats=embedding_dim//2, normalize=True, offset=-0.5)

        # I am very sorry. I gave a name like ChangeAwareTokenReduce. Because I was originally planning to continue exploring the application of tokenreduce, but this did not work. Due to time constraints, I did not change the name of the class.
        self.ChangeAwareTokenReduce = ChangeAwareTokenReduce(raw_grid=27, scale_factor=downsample_ratio, embedding_dim=embedding_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, imageA_emb, imageB_emb):

        imageA_emb_final_stage = imageA_emb[-1]
        imageB_emb_final_stage = imageB_emb[-1]

        # >>>>>>>>>>>>>>>>>>>>>>>>>> 处理多尺度特征差异提取与精炼 <<<<<<<<<<<<<<<<<<<<<<<<<<<
        bs_per_level = imageA_emb[0].shape[0]  # 每个尺度的 batch size
        # [[bs, n_tokens, emb], [bs, n_tokens, emb] .... ] 多尺度特征, 按bs拼一起
        imageA_multilevel_concat_emb = torch.cat(imageA_emb, dim=0)
        imageB_multilevel_concat_emb = torch.cat(imageB_emb, dim=0)

        for layer_index, layer in enumerate(self.layers):
            device = imageA_multilevel_concat_emb.device  # 获取 imageA_emb 所在的设备
            bs, n_tokens, emb = imageA_multilevel_concat_emb.shape
            mask_for_pe = torch.zeros(bs, int(n_tokens**0.5), int(n_tokens**0.5), dtype=torch.int, device=device)
            pe = self.position_emb_model(mask_for_pe).flatten(2).permute(0, 2, 1).to(device).to(imageA_multilevel_concat_emb.dtype)  # shape: [bs, n_tokens, emb]
            
            change_multilevel_concat_emb = self.expert_change_feature_extractor_model(imageA_multilevel_concat_emb, imageB_multilevel_concat_emb)
            imageA_multilevel_concat_emb, imageB_multilevel_concat_emb, queries = layer(
                queries=queries if layer_index != 0 else change_multilevel_concat_emb,
                change_emb=change_multilevel_concat_emb,
                imageA_emb=imageA_multilevel_concat_emb,
                imageB_emb=imageB_multilevel_concat_emb,
                pe=pe,
            )
        imageA_emb_after_expert = torch.split(imageA_multilevel_concat_emb, bs_per_level, dim=0)  # [[bs, n_tokens, emb], ...]
        imageB_emb_after_expert = torch.split(imageB_multilevel_concat_emb, bs_per_level, dim=0)  # [[bs, n_tokens, emb], ...]
        change_emb_after_expert = torch.split(queries, bs_per_level, dim=0)  # [[bs, n_tokens, emb], ...]

        # >>>>>>>>>>>>>>>>>>>>>>>>>> 做token reduce <<<<<<<<<<<<<<<<<<<<<<<<<<<
        imageA_emb_after_tokenreduce, imageB_emb_after_tokenreduce, change_emb_combined = self.ChangeAwareTokenReduce(imageA_emb_final_stage, imageB_emb_final_stage, imageA_emb_after_expert, imageB_emb_after_expert, change_emb_after_expert)

        return imageA_emb_after_tokenreduce, imageB_emb_after_tokenreduce, change_emb_combined

if __name__ == "__main__":
    """
    Since you have scrolled here. If you think this project is useful, even giving it a star is the greatest encouragement for me.
    Thanks~
    """
    pass