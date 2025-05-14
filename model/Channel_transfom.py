import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DualPathChannelAttention(nn.Module):
    def __init__(self, channel=3, expand_ratio=3):
        super().__init__()
        self.channel = channel
        
        # 通道压缩的中间维度
        self.inter_dim = channel * expand_ratio
        
        self.mlp = nn.Sequential(
            nn.Linear(channel, self.inter_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.inter_dim, channel)
        )
        
        # 最后的通道缩放参数
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, xpl, ppl):
        """
        输入:
            xpl: [B, C, H, W]
            ppl: [B, C, H, W]
        输出:
            weighted_ppl: [B, C, H, W]
        """
        B, C, H, W = xpl.size()

        # 展平空间维度
        xpl_flat = xpl.view(B, self.channel, -1)  # [B, channel, H*W]
        ppl_flat = ppl.view(B, -1,self.channel)  # [B, H*W, channel]

        # 计算通道分配矩阵
        allocation_matrix = torch.matmul(
            xpl_flat,  # [B, H*W, inter_dim]
            ppl_flat   # [B, inter_dim, H*W]
        )  # 结果维度 [B, C, C]

        allocation_matrix = self.mlp(allocation_matrix)/torch.sqrt(H*W)  # [B, C, C]

        # 对分配矩阵做归一化
        # allocation_matrix = F.softmax(allocation_matrix, dim=-1)

        # 应用权重分配
        ppl_reshaped = ppl.view(B, -1, C)  # [B,H*W,C]
        weighted_features = torch.matmul(
            ppl_reshaped,        # [B, H*W, C]
            allocation_matrix    # [B, C, C]
        )  # 结果维度 [B, H*W, C]

        # 恢复空间维度
        weighted_ppl = weighted_features.view(B, C, H, W)
        
        # 残差连接（可选）
        return weighted_ppl #self.gamma * weighted_ppl + ppl
