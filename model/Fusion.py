import torch
import torch.nn as nn

class Compensation_Fusion(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, depth=3):
        super(Compensation_Fusion, self).__init__()
        self.depth = depth
        
        # 定义多层融合模块
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout),
                'conv_adjust': nn.Sequential(
                    nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),  # 将通道数从 2*embed_dim 调整回 embed_dim
                    nn.ReLU(),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),  # 进一步细化特征
                    nn.LayerNorm([embed_dim, 64, 64])  # 假设特征图大小为 64x64
                ),
                'layer_norm': nn.LayerNorm(embed_dim)
            })
            for _ in range(depth)
        ])
        
        # 可学习的权重 β
        self.beta = nn.Parameter(torch.tensor(0.5))  # 初始值为 0.5

    def forward(self, q, q_k, v, attn_mask=None, key_padding_mask=None):
        # q_k: 原始特征 (batch_size, embed_dim, h, w)
        # v: 差异化特征 (batch_size, embed_dim, h, w)
        batch_size, embed_dim, h, w = q_k.shape
        
        # 将特征转换为 (seq_len, batch_size, embed_dim) 形式
        q_k = q_k.view(batch_size, embed_dim, -1).permute(2, 0, 1)  # (h*w, batch_size, embed_dim)
        v = v.view(batch_size, embed_dim, -1).permute(2, 0, 1)      # (h*w, batch_size, embed_dim)
        
        # 多层融合操作
        for layer in self.layers:
            # 多头注意力机制
            attn_output, _ = layer['attention'](q_k, v, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            q_k = q_k + attn_output  # 残差连接
            q_k = layer['layer_norm'](q_k)  # LayerNorm
            
            # 将特征转换回 (batch_size, embed_dim, h, w)
            q_k_reshaped = q_k.permute(1, 2, 0).view(batch_size, embed_dim, h, w)
            v_reshaped = v.permute(1, 2, 0).view(batch_size, embed_dim, h, w)
            
            # 拼接原始特征和差异化特征
            fused_feature = torch.cat([q_k_reshaped, v_reshaped], dim=1)  # (batch_size, 2*embed_dim, h, w)
            
            # 用卷积调整形状
            fused_feature = layer['conv_adjust'](fused_feature)  # (batch_size, embed_dim, h, w)
            
            # 更新 q_k 和 v 用于下一层
            q_k = fused_feature.view(batch_size, embed_dim, -1).permute(2, 0, 1)
            v = v_reshaped.view(batch_size, embed_dim, -1).permute(2, 0, 1)
        
        # 将最终特征转换回 (batch_size, embed_dim, h, w)
        output = q_k.permute(1, 2, 0).view(batch_size, embed_dim, h, w)
        return output