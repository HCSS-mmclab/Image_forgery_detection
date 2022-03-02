import torch, torch.nn as nn, torch.nn.functional as F
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import torch.fft as fft
import cv2
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class PatchEmbedding(nn.Module): # 이건 conv로 하는거
    def __init__(self, in_channels: int = 1, patch_size: int = 16, emb_size: int = 256, img_size: int = 256):
        self.patch_size = patch_size
        super().__init__()

        # self.projection = nn.Sequential(
        #     # using a conv layer instead of a linear one -> performance gains
        #     nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
        #     Rearrange('b e (h) (w) -> b (h w) e'),
        # )
        ### 이게 patch로 linear하는거
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.Linear(256, dim),
        # )

        ### 이게 patch로 conv하는거
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(1,emb_size,patch_size,patch_size), # rfft : 1,256,16,8
            Rearrange('b e (h) (w) -> b (h w) e'), # 1,128,256
        )

        ### 이게 flatten linear하는거
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b (c) (h) w -> b w (c h)'),
        #     nn.Linear(img_size, emb_size))


        ## rfft patch로 했을 때(linear, conv)
        self.pos_embedding = nn.Parameter(torch.randn(1,img_size//2,emb_size)) # 128,256

        ## rfft flatten 세로
        # self.pos_embedding = nn.Parameter(torch.randn(1,img_size,dim)) # 256, 256

        ## fft patch로 했을 때
        # self.pos_embedding = nn.Parameter(torch.randn((img_size // patch_size) ** 2, emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape

        x = self.to_patch_embedding(x) # b,256,129*1
        x += self.pos_embedding
        return x

class Rfft_mymodel(nn.Module):
    def __init__(self, enet_type=None, out_dim=4, img_size=256, patch_size=16, n_classes=1000, dim=256, depth=12, heads=12, mlp_dim=256, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(Rfft_mymodel,self).__init__()
        ##################################################################################################################################
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        ##################################################################################################################################

        #flatten embedding
        # patch_height = 1
        # patch_width = img_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) # patch 개수
        patch_dim = channels * patch_height * patch_width # 1 * 16 * 16,  patch하나에 들어있는 값?
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        ## 이게 flatten으로 하는거
        self.to_patch_embedding = PatchEmbedding(in_channels=channels,patch_size=patch_size,emb_size=dim,img_size=img_size)

        ## transformer 들어가기 전
        self.dropout = nn.Dropout(emb_dropout) # transformer 들어가기 전

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.avgpoold2d = nn.AdaptiveAvgPool2d((16,16))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes)
        )
        self.fc1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(n_classes, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, out_dim))

        # self.preconv1 = nn.Sequential(
        #     nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
        #     nn.LayerNorm(img_size),
        #     nn.GELU(),
        # )
        # self.preconv2 = nn.Sequential(
        #     nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1),
        #     # nn.LayerNorm(img_size)
        # )
        #     self.const_weight = nn.Parameter(torch.randn(size=[1, 1, 5, 5]), requires_grad=True)
        #
        # def normalized_F(self):
        #     central_pixel = (self.const_weight.data[:, 0, 2, 2])
        #     for i in range(1):
        #         sumed = self.const_weight.data[i].sum() - central_pixel[i]
        #         self.const_weight.data[i] /= sumed
        #         self.const_weight.data[i, 0, 2, 2] = -1.0
    def forward(self, img):
        # self.normalized_F()
        # img = F.conv2d(img, self.const_weight,padding=2)
        # x = img + self.preconv2(self.preconv1(img))
        img = torch.abs(fft.rfft2(img, dim=(2, 3), norm='ortho')) # b,1,256,129
        img = img[:,:,:,:-1] # b,1,256,128

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        ##########여기 밑은 고정하고 하자..!!################
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = self.fc1(x)
        return x

from pytorch_pretrained_vit import ViT

class Vit(nn.Module):
    def __init__(self,enet_type=None,out_dim=4,pretrained=True):
        super(Vit, self).__init__()
        self.model = ViT('B_16_imagenet1k', pretrained=False, in_channels=1, image_size=256)
        # for param in self.model.parameters():
        #     param.requires_grad = True
        # config = dict(hidden_size=512,num_heads=8,num_layers=6)
        # self.model = self.model.from_config(config)
        self.fc1 = nn.Sequential(nn.LeakyReLU(),
                                 nn.Linear(1000, 512),
                                 nn.LeakyReLU(),
                                 nn.Linear(512, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, out_dim))
    def forward(self,x):
        # x = torch.abs(fft.fft2(x, dim=2))
        x = self.model(x)
        x = self.fc1(x)
        return x