from utils import *
from modules import *


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=False) # infer a one-channel attention map
    def forward(self, ftr):
        # ftr: [B, C, H, W]
        ftr_avg = torch.mean(ftr, dim=1, keepdim=True) # [B, 1, H, W], average
        ftr_max, _ = torch.max(ftr, dim=1, keepdim=True) # [B, 1, H, W], max
        ftr_cat = torch.cat([ftr_avg, ftr_max], dim=1) # [B, 2, H, W]
        att_map = F.sigmoid(self.conv(ftr_cat)) # [B, 1, H, W]
        return att_map
    
    
class CPA(nn.Module):
    # Cascaded Pyramid Attention
    def __init__(self, in_channels):
        super(CPA, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.conv_1 = nn.Conv2d(in_channels, in_channels//4, kernel_size=1)
        self.conv_2 = nn.Conv2d(in_channels, in_channels//4, kernel_size=1)
        self.SA0 = SpatialAttention()
        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()
    def forward(self, ftr):
        # ftr: [B, C, H, W]
        d0 = self.conv_0(ftr) # [B, C/2, H, W]
        d1 = self.conv_1(DS2(ftr)) # [B, C/4, H/2, W/2]
        d2 = self.conv_2(DS4(ftr)) # [B, C/4, H/4, W/4]
        # level-2
        a2 = self.SA2(d2) #  [B, 1, H/4, W/4]
        d2 = a2*d2 + d2 # [B, C/4, H/4, W/4]
        # level-1
        d1 = torch.cat([d1, US2(d2)], dim=1) # [B, C/2, H/2, W/2]
        a1 = self.SA1(d1) # [B, 1, H/2, W/2]
        d1 = a1*d1 + d1 # [B, C/2, H/2, W/2]
        # level-0
        d0 = torch.cat([d0, US2(d1)], dim=1) # [B, C, H, W]
        a0 = self.SA0(d0) # [B, 1, H, W]
        return a0, d0


class ChannelRecalibration(nn.Module):
    def __init__(self, in_channels):
        super(ChannelRecalibration, self).__init__()
        inter_channels = in_channels // 4 # channel squeezing
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_fc = nn.Sequential(nn.Linear(in_channels, inter_channels, bias=False),
                           nn.ReLU(inplace=True),
                           nn.Linear(inter_channels, in_channels, bias=False))
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_fc = nn.Sequential(nn.Linear(in_channels, inter_channels, bias=False),
                           nn.ReLU(inplace=True),
                           nn.Linear(inter_channels, in_channels, bias=False))
    def forward(self, ftr):
        # ftr: [B, C, H, W]
        ftr_avg = self.avg_fc(self.avg_pool(ftr).squeeze(-1).squeeze(-1)) # [B, C]
        ftr_max = self.max_fc(self.max_pool(ftr).squeeze(-1).squeeze(-1)) # [B, C]
        weights = F.sigmoid(ftr_avg + ftr_max).unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]
        out = weights * ftr
        return out
    
    
class GFA(nn.Module):
    # Global Feature Aggregation
    def __init__(self, in_channels, squeeze_ratio=4):
        super(GFA, self).__init__()
        inter_channels = in_channels // squeeze_ratio # reduce computation load
        self.conv_q = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.delta = nn.Parameter(torch.Tensor([0.1])) # initiate as 0.1
        self.cr = ChannelRecalibration(in_channels)
    def forward(self, ftr):
        B, C, H, W = ftr.size()
        P = H * W
        ftr_q = self.conv_q(ftr).view(B, -1, P).permute(0, 2, 1) # [B, P, C']
        ftr_k = self.conv_k(ftr).view(B, -1, P) # [B, C', P]
        ftr_v = self.conv_v(ftr).view(B, -1, P) # [B, C, P]
        weights = F.softmax(torch.bmm(ftr_q, ftr_k), dim=1) # column-wise softmax, [B, P, P]
        G = torch.bmm(ftr_v, weights).view(B, C, H, W)
        out = self.delta * G + ftr
        out_cr = self.cr(out)
        return out_cr
    
    
class GCA(nn.Module):
    # Global Context-aware Attention
    def __init__(self, in_channels, use_pyramid):
        super(GCA, self).__init__()
        assert isinstance(use_pyramid, bool)
        self.use_pyramid = use_pyramid
        self.gfa = GFA(in_channels)
        if self.use_pyramid:
            self.cpa = CPA(in_channels)
        else:
            self.sau = SpatialAttention()
    def forward(self, ftr):
        ftr_global = self.gfa(ftr)
        if self.use_pyramid:
            att, ftr_refined = self.cpa(ftr_global)
            return att, ftr_refined
        else:
            att = self.sau(ftr_global)
            return att, ftr_global
        
        
class AttentionFusion(nn.Module):
    def __init__(self, num_att_maps):
        super(AttentionFusion, self).__init__()
        dim = 256
        self.conv_1 = ConvBlock(num_att_maps, dim, 3, False, 'ReLU')
        self.conv_2 = ConvBlock(dim, dim, 3, False, 'ReLU')
        self.conv_3 = ConvBlock(dim, 1, 3, False, 'Sigmoid')
    def forward(self, concat_att_maps):
        fusion_att_maps = self.conv_3(self.conv_2(self.conv_1(concat_att_maps)))
        return fusion_att_maps  
    
    