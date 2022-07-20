# modified from https://github.com/zihangJiang/TokenLabeling
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'LV_ViT_Tiny': _cfg(),
    'LV_ViT': _cfg(),
    'LV_ViT_Medium': _cfg(crop_pct=1.0),
    'LV_ViT_Large': _cfg(crop_pct=1.0),
}

class Mlp(nn.Module):
    '''
    MLP with support to use group linear operator
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    '''
    Multi-head self-attention
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with some modification to support different num_heads and head_dim.
    '''
    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim=head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, self.head_dim* self.num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim* self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # B,heads,N,C/heads 

        q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

        
class Block(nn.Module):
    '''
    Pre-layernorm transformer block
    '''
    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, group=1, skip_lam=1.):
        super().__init__()
        self.dim = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.skip_lam = skip_lam

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=self.mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))/self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x)))/self.skip_lam
        return x


class PatchEmbedNaive(nn.Module):
    """ 
    Image to Patch Embedding
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x


class PatchEmbed4CG(nn.Module):
    """
    Image to Patch Embedding with 4 layer convolution
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_chans, embed_dim//8, kernel_size=3, stride=2, padding=1, bias=False)  # 112x112
        self.bn1 = nn.BatchNorm2d(embed_dim//8)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(embed_dim//8, embed_dim//4, kernel_size=3, stride=2, padding=1, bias=False)  # 112x112
        self.bn2 = nn.BatchNorm2d(embed_dim//4)
        self.conv3 = nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(embed_dim//2)
        self.conv4 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gelu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x


class PatchEmbed6CG(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_chans, embed_dim//8, kernel_size=3, stride=2, padding=1, bias=False)  # 112x112
        self.bn1 = nn.BatchNorm2d(embed_dim//8)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(embed_dim//8, embed_dim//8, kernel_size=3, stride=1, padding=1, bias=False)  # 112x112
        self.bn2 = nn.BatchNorm2d(embed_dim//8)
        self.conv3 = nn.Conv2d(embed_dim//8, embed_dim//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(embed_dim//8)
        self.conv4 = nn.Conv2d(embed_dim//8, embed_dim//4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(embed_dim//4)
        self.conv5 = nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(embed_dim//2)
        self.conv6 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gelu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.gelu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.gelu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        return x


class ReverseTSM(nn.Module):
    def __init__(self, dim, keeped_patches, recovered_patches, ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mlp1 = Mlp(keeped_patches, int(recovered_patches*ratio), recovered_patches)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(dim, int(dim*ratio), dim)
    def forward(self, x):
        B, N, C = x.shape
        x = self.norm1(x)
        x = self.mlp1(x.transpose(2, 1))
        x = x.transpose(2, 1)
        x = x + self.mlp2(self.norm2(x))
        return x


class TokenSlimmingModule(nn.Module):
    def __init__(self, dim, keeped_patches, ratio=0.5):
        super().__init__()
        hidden_dim = int(dim*ratio)
        self.weight = nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, keeped_patches))
        self.scale = nn.Parameter(torch.ones(1, 1, 1))
        
    def forward(self, x):
        weight = self.weight(x)
        weight = F.softmax(weight * self.scale, dim=1).transpose(2, 1)
        x = torch.bmm(weight, x)
        return x


def get_dpr(drop_path_rate, depth, drop_path_decay='linear'):
    if drop_path_decay=='linear':
        # linear dpr decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    elif drop_path_decay=='fix':
        # use fixed dpr
        dpr= [drop_path_rate]*depth
    else:
        # use predefined drop_path_rate list
        assert len(drop_path_rate)==depth
        dpr=drop_path_rate
    return dpr


class LVViT(nn.Module):
    """ Vision Transformer with tricks
    Arguements:
        p_emb: different conv based position embedding (default: 4 layer conv)
        skip_lam: residual scalar for skip connection (default: 1.0)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=384, depth=16,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., drop_path_decay='linear', norm_layer=nn.LayerNorm, p_emb='4cg', head_dim=None,
                 skip_lam=1.0, stage_blocks=[1,1,1,13], keeping_ratio=0.5, distillation_type=None, 
                 conv_distillation=False, with_cp=False):
        super().__init__()
        assert sum(stage_blocks) == depth
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.stage_blocks = stage_blocks
        self.num_stages = len(stage_blocks)
        if distillation_type == 'none':
            distillation_type = None
        self.distillation_type = distillation_type
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.output_dim = embed_dim if num_classes==0 else num_classes
        self.conv_distillation = conv_distillation
        self.with_cp = with_cp

        if p_emb=='4c':
            patch_embed_fn = PatchEmbed4CG
        elif p_emb=='6c':
            patch_embed_fn = PatchEmbed6CG
        else:
            patch_embed_fn = PatchEmbedNaive
        self.patch_embed = patch_embed_fn(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr=get_dpr(drop_path_rate, depth, drop_path_decay)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Aux head for token-labeling. It is inherited from the original LV-ViT and we do not modify it.
        self.aux_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.aux_head.weight.requires_grad = False
        self.aux_head.bias.requires_grad = False
        
        self.keeped_patches = [int(num_patches*((keeping_ratio)**i)) for i in range(self.num_stages)]
        self.layer_patches = []
        for i in range(self.num_stages):
            self.layer_patches += self.stage_blocks[i]*[self.keeped_patches[i]]
        self.layer_patches += self.stage_blocks[-1]*[self.keeped_patches[-1]]

        if self.distillation_type is not None:
            self.slim_index = [sum(stage_blocks[:i]) for i in range(1, self.num_stages)]
            self.rtsm = nn.ModuleList()
            self.tsm = nn.ModuleList()
            for i in range(1, self.num_stages):
                self.tsm.append(TokenSlimmingModule(self.embed_dim, self.keeped_patches[i]))
            dropped_token = 0
            for i in range(self.depth):
                dropped_token = max(dropped_token, self.num_patches - self.layer_patches[i+1])
                if dropped_token > 0:
                    self.rtsm.append(ReverseTSM(self.embed_dim, self.layer_patches[i+1], self.num_patches))
                else:
                    self.rtsm.append(nn.Identity())

        if self.conv_distillation:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
            self.dist_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.dist_token, std=.02)
            trunc_normal_(self.dist_pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distillation_type is not None and self.conv_distillation:
            dist_token = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
            pos_embed = torch.cat((self.pos_embed[:,0,:].unsqueeze(1), self.dist_pos_embed, self.pos_embed[:,1:,]), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
            pos_embed = self.pos_embed
        x = x + pos_embed
        x = self.pos_drop(x)

        layer_tokens = []
        for i, blk in enumerate(self.blocks):
            if self.with_cp and self.training:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            if self.distillation_type is not None:
                if self.conv_distillation:
                    cls_t, dist_t, tokens = x[:,0:1], x[:,1:2], x[:,2:]
                else:
                    cls_t, tokens = x[:,0:1], x[:,1:]
                if (i+1) in self.slim_index:
                    tsm = self.tsm[self.slim_index.index(i+1)]
                    tokens = tsm(tokens)
                if self.conv_distillation:
                    x = torch.cat((cls_t, dist_t, tokens), dim=1)
                else:
                    x = torch.cat((cls_t, tokens), dim=1)
                if self.training:
                    tokens = self.rtsm[i](tokens)
            else:
                tokens = x[:,1:]
            layer_tokens.append(tokens)

        x = self.norm(x)
        x_cls = self.head(x[:,0])
        if self.training:
            if self.conv_distillation:
                x_dist = self.dist_head(x[:,1])
                return x_cls, x_dist, layer_tokens
            else:
                return x_cls, x_cls, layer_tokens
        else:
            if self.conv_distillation:
                return (x_cls+x_dist)/2, layer_tokens
            else:
                return x_cls, layer_tokens
                

@register_model
def sit_lvvit_tiny(pretrained=True, **kwargs):
    model = LVViT(
        patch_size=16, embed_dim=320, depth=14, num_heads=5,
        mlp_ratio=3., p_emb='4c', skip_lam=2., 
        stage_blocks=[1, 1, 1, 11], **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def sit_lvvit_xsmall(pretrained=True, **kwargs):
    model = LVViT(
        patch_size=16, embed_dim=384, depth=16, num_heads=6,
        mlp_ratio=3., p_emb='4c', skip_lam=2., 
        stage_blocks=[1, 1, 1, 13], **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def sit_lvvit_small(pretrained=True, **kwargs):
    model = LVViT(
        patch_size=16, embed_dim=384, depth=16, num_heads=6,
        mlp_ratio=3., p_emb='4c', skip_lam=2., 
        stage_blocks=[9, 3, 2, 2], **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def sit_lvvit_medium(pretrained=True, **kwargs):
    model = LVViT(
        patch_size=16, embed_dim=512, depth=20, num_heads=8,
        mlp_ratio=3., p_emb='4c', skip_lam=2., 
        stage_blocks=[10, 4, 3, 3], **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def sit_lvvit_large(pretrained=True, **kwargs):
    model = LVViT(
        img_size=288, patch_size=16, embed_dim=768, depth=24, 
        num_heads=12, mlp_ratio=3., p_emb='6c', skip_lam=3., 
        stage_blocks=[10, 4, 3, 7], with_cp=True, **kwargs)
    model.default_cfg = _cfg()
    return model