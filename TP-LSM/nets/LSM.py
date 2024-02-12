from timm.models.layers import trunc_normal_
import math
import torch.nn as nn


class Expan_Compre_Conv(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super(Expan_Compre_Conv, self).__init__()
        self.channel_ratio = 2
        # 创建一个深度可分离一维卷积层
        self.DC = nn.Conv2d(in_channels=self.channel_ratio, out_channels=self.channel_ratio, kernel_size=3, dilation=2, stride=1, padding=2, groups=self.channel_ratio, bias=False)
        # 创建一个一维卷积层，用于将通道数从32降到1
        self.DC2 = nn.Conv2d(in_channels=self.channel_ratio, out_channels=1, kernel_size=1, bias=False)
        # self.DC2 = nn.Linear(32,1,bias=False)
    def forward(self, x):
        x = x.unsqueeze(1)
        # 将通道数复制32倍，得到[N, 32, C, D]，这是为了与第一个卷积层的深度可分离卷积匹配
        x = x.repeat(1, self.channel_ratio, 1, 1)
        # 经过第一个卷积层
        x = self.DC(x)
        # 经过第二个卷积层，将通道数从32降到1
        x = self.DC2(x)
        # 在第一个维度上去除多余的维度，得到[N, C, D]
        x = x.squeeze(1)

        return x


class Short_TDLearn_Block(nn.Module):

    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        self.linear_E = nn.Linear(in_features, hidden_features)
        self.EC_Conv = Expan_Compre_Conv(in_features, hidden_features, act_layer=nn.GELU, drop=0.)
        self.act = nn.GELU()
        self.linear_C = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(0.)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.linear_E(x)
        x = x.transpose(1, 2)

        x = self.EC_Conv(x)

        x = x.transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear_C(x)
        x = self.drop(x)

        return x




class MEAttention(nn.Module):
    def __init__(self, dim, S, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads

        self.coef = 4
        self.trans_dims = nn.Linear(dim, dim * self.coef)    

        self.num_heads = self.num_heads * self.coef
        self.k = S // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.attn_drop = nn.Dropout(attn_drop)        
        self.proj = nn.Linear(dim * self.coef, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        x = self.trans_dims(x) # B, N, C 
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        
        attn = self.linear_0(x)


        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        attn = self.attn_drop(attn)
        x = self.linear_1(attn).permute(0,2,1,3).reshape(B, N, -1)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Long_TDLearn_Block(nn.Module):
    def __init__(self, dim, s=64):
        super().__init__()
        self.mea = MEAttention(dim, s)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.mea(x)
        return x


class LSModule(nn.Module):
    

    def __init__(self, dim, s, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.Long_TDLearn_Block = Long_TDLearn_Block(dim, s)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.Short_TDLearn_Block = Short_TDLearn_Block(in_features=dim, 
                                                       hidden_features=mlp_hidden_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.Long_TDLearn_Block(self.norm1(x))
        x = x + self.Short_TDLearn_Block(self.norm2(x))
        return x


class TD_Merging_Block(nn.Module):
   
        def __init__(self, kernel_size=3, stride=1, in_chans=1024, embed_dim=256):
            super().__init__()
            self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=kernel_size, stride=1,
                                padding=(kernel_size// 2))
            self.proj2 = nn.MaxPool1d(kernel_size=3, stride=stride, padding=1)
            self.norm = nn.LayerNorm(embed_dim)
    

    
            self.apply(self._init_weights)

        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

        def forward(self, x):

            x = self.proj(x)
            x = self.proj2(x)
            x = x.transpose(1, 2)
            x = self.norm(x)
            return x


class TPLSM_Encoder(nn.Module):
    def __init__(self, in_feat_dim=1024, embed_dims=[864,576,384,256],
                 s_size=[128,128,64,64], mlp_ratio=8, num_block=3):
        super().__init__()


        # Stage 1
        self.TD_Merging_Block1 = TD_Merging_Block(
            kernel_size=3, stride=1, in_chans=in_feat_dim, embed_dim=embed_dims[0])
        self.block1 = nn.ModuleList([LSModule(
            dim=embed_dims[0], s=s_size[0], mlp_ratio=mlp_ratio)
            for i in range(num_block)])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        # Stage 2
        self.TD_Merging_Block2 = TD_Merging_Block(
            kernel_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.block2 = nn.ModuleList([LSModule(
            dim=embed_dims[1], s=s_size[1], mlp_ratio=mlp_ratio)
            for i in range(num_block)])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        # Stage 3
        self.TD_Merging_Block3 = TD_Merging_Block(
            kernel_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.block3 = nn.ModuleList([LSModule(
            dim=embed_dims[2], s=s_size[2], mlp_ratio=mlp_ratio)
            for i in range(num_block)])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        # Stage 4
        self.TD_Merging_Block4 = TD_Merging_Block(
            kernel_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.block4 = nn.ModuleList([LSModule(
            dim=embed_dims[3], s=s_size[3], mlp_ratio=mlp_ratio)
            for i in range(num_block)])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def freeze_init_emb(self):
    #     self.TD_Merging_Block1.requires_grad = False

    def forward(self, x):
        outs = []


        # stage 1
        x = self.TD_Merging_Block1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x)
        x = self.norm1(x)
        x = x.permute(0, 2, 1).contiguous()
        outs.append(x)

        # stage 2
        x = self.TD_Merging_Block2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x)
        x = self.norm2(x)
        x = x.permute(0, 2, 1).contiguous()
        outs.append(x)

        # stage 3
        x = self.TD_Merging_Block3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x)
        x = self.norm3(x)
        x = x.permute(0, 2, 1).contiguous()
        outs.append(x)

        # stage 4
        x = self.TD_Merging_Block4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x)
        x = self.norm4(x)
        x = x.permute(0, 2, 1).contiguous()
        outs.append(x)

        return outs