import torch
import torch.nn as nn
import math
import torch.nn.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

activation = nn.PReLU
norm       = nn.LayerNorm

class LinearAttention(nn.Module):

    def __init__(self, in_heads, out_heads):
        super().__init__()
        in_features = 7
        
        in_sz = in_features * in_heads
        out_sz = in_features * out_heads
        
        self.key = nn.Linear(in_sz, out_sz)
        self.query = nn.Linear(in_sz, out_sz)
        self.value = nn.Linear(in_sz, out_sz)
        
        self.heads = out_heads
        self.in_features = in_features
        
    def split_heads(self, x):
        N, L, D = x.shape
        x = x.view(N, L, self.heads, -1).contiguous()
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, x):
        N, C, L, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous() # N x L x C x D
        x = x.view(N, L, -1).contiguous() # N x L x C*D
        
        
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        k = self.split_heads(k)
        q = self.split_heads(q)
        v = self.split_heads(v)

        scores = torch.einsum("bnqd,bnkd->bnqk", q, k)
        scores = scores / math.sqrt(scores.shape[-1])
        
        weights = F.softmax(scores.float(), dim=-1).type_as(scores) 
        weights = F.dropout(weights, p=0.5, training=self.training)
        attention = torch.matmul(weights, v)
        
        return attention

class MixedDilationConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        dil1 = out_channels // 2
        dil2 = out_channels - dil1
        self.conv = nn.Conv2d(in_channels, dil1, kernel_size=3, padding=1, dilation=1)
        self.conv1 = nn.Conv2d(in_channels, dil2, kernel_size=3, padding=2, dilation=2)

    def forward(self, x):
        o = self.conv(x)
        o1 = self.conv1(x)
        out = torch.cat((o, o1), dim=1)
        return out
    

    
class AttnBlock(nn.Module):
    def __init__(self, in_dv, in_channels, out_dv, conv_channels):
        super().__init__()
        self.attn = LinearAttention(in_dv, out_dv)
        self.conv = MixedDilationConv(in_channels, conv_channels)
        self.context = nn.Conv2d(out_dv+conv_channels, out_dv+conv_channels, kernel_size=1)
    def forward(self, x):
        o = self.attn(x)        
        o1 = self.conv(x)
        
        fo = torch.cat((o, o1), dim=1)
        fo = self.context(fo)
        
        return fo


class HybridAttentionModel(nn.Module):

    def __init__(self):
        super().__init__()
        neurons = 128
        drop = 0.5
        self.net = nn.Sequential(
            AttnBlock(2, 2, 16, 32),
            norm((48,147,7)),
            activation(),
            nn.Dropout(drop),
            AttnBlock(48, 48, 16, 32),
            norm((48,147,7)), 
            activation(),
            nn.Dropout(drop),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(48 * 1029, neurons * 8),
            nn.BatchNorm1d(neurons * 8),
            activation(),
            nn.Dropout(0.6),
            nn.Linear(neurons * 8, 2),
        )

    def forward(self, x):
        N, C, _ = x.shape
        x = x.view(N, C, 147, -1)
        o = self.net(x)
        o = self.classifier(o.view(N, -1))
        return o

