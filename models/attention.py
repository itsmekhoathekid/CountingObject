import torch
import torch.nn as nn
import math

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class ChannelGatedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.channel_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid() # Output từ 0 đến 1
        )

    def forward(self, x, y):
        B, Nx, C = x.shape
        Ny = y.shape[1]
        
        q = self.wq(x).reshape(B, Nx, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(y).reshape(B, Ny, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(y).reshape(B, Ny, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        gate = self.channel_gate(x_attn) 
        x_refined = x_attn * gate
        
        return x_refined

class ConvCrossAttentionBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, resolution = 1., 
            init_values=1e-4): 
        
        super().__init__()
        
        self.norm0 = norm_layer(dim)
        self.conv0 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.act0 = act_layer()
        self.selfattn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True) # Dùng luôn class chuẩn cho gọn
        self.drop_path0 = nn.Identity() # Simplified for demo

        self.norm1 = norm_layer(dim)
        self.attn = ChannelGatedCrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.drop_path1 = nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True) # LayerScale cho MLP
        self.drop_path2 = nn.Identity()

        self.resolution = resolution
        if resolution == 1.:
            self.interpolate_layer = nn.Identity()
        elif resolution < 1.:
            self.interpolate_layer = nn.AvgPool2d(kernel_size=int(1/resolution), stride=int(1/resolution))
        else:
            self.interpolate_layer = nn.Upsample(scale_factor=resolution, mode='bilinear', align_corners=False)

    def seq_to_2d(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        return x.transpose(1, 2).reshape(n, c, h, w)
    
    def _2d_to_seq(self, x):
        n, c, h, w = x.shape
        return x.reshape(n, c, h*w).transpose(1, 2)

    def forward(self, x, y, cls_token):

        if self.resolution != 1:
            x_2d = self.seq_to_2d(x)
            x_2d = x_2d + self.act0(self.conv0(x_2d))
            x_2d = self.interpolate_layer(x_2d)
            x = self._2d_to_seq(x_2d)

        x = x + self.selfattn(self.norm0(x), self.norm0(x), self.norm0(x))[0]

        text_interaction = self.attn(self.norm1(cls_token), y)
        
        x = x + self.drop_path1(self.gamma_1 * text_interaction)
        x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        
        return x