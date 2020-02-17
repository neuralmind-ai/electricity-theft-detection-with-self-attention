import torch
import torch.nn as nn


class GoogleAttention(nn.Module):

    def __init__(self, in_channels, dk, dv, n_heads=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2 * dk + dv, 1)
        self.projection = nn.Conv2d(dv, dv, 1)
        self.dk = dk
        self.dv = dv
        self.n_heads = n_heads

    def reshape_for_matmul(self, x):
        N, Nh, C, H, W = x.shape
        x = x.view(N, Nh, C, H*W)
        return x.transpose(-2, -1)
        return x.view(-1, d, hw).permute(0, 2, 1)
    
    def split_heads(self, x):
        N, C, H, W =  x.shape
        x = x.view(N, self.n_heads, C // self.n_heads, H, W)
        return x
        

    def forward(self, x):
        N, _, H, W = x.shape
        
        ## part 1
        kqv = self.conv(x)
        k, q, v = torch.split(kqv, (self.dk, self.dk, self.dv), dim=1)
        q = q * (self.dk**(-0.5)) # scaled dot−product    
    
        ## part 2        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        ## part 3
        logits = torch.matmul(self.reshape_for_matmul(q),
                              self.reshape_for_matmul(k).transpose(-2, -1))
        
        weights = nn.Softmax(dim=-1)(logits)
        attn = torch.matmul(weights, self.reshape_for_matmul(v))
        
        ## part 4 - combine heads
        attn = attn.transpose(-2, -1)
        attn = torch.split(attn, [1] * self.n_heads, dim=1)
        attn = torch.cat(attn, dim=2).squeeze().view(N, -1, H, W)
        
        ## part 5 - proejction
        attn = self.projection(attn)
        return attn
    
class GoogleAugmentedAttention(nn.Module):

    def __init__(self, in_channels, out_channels, dk, dv, kernel_size=3, padding=1):
        super().__init__()
        self.convolutional = nn.Conv2d(in_channels, out_channels-dv, kernel_size, padding=padding)
        self.attention = GoogleAttention(in_channels, dk, dv)

    def forward(self, x):
        conv = self.convolutional(x)
        attn = self.attention(x)
        out = torch.cat((conv, attn), dim=1)
        return out

    
class GoogleFullModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self.block(2, 32),
            self.block(32,32),
            self.block(32,32),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 147 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512,2)
        )

    def block(self, in_channels, out_channels, dv=16, dropout=0.4):
        return nn.Sequential(
          GoogleAugmentedAttention(in_channels, out_channels, dk=16, dv=dv, kernel_size=3, padding=1),  # n x out_channels x m x out_features
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        N, C, _ = x.shape
        x = x.view(N, C, -1, 7)
        o = self.net(x)
        o = o.view(x.shape[0], -1)
        o = self.classifier(o)
        return o
