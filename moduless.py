from utils import *


class ConvBlock(nn.Module):
    def __init__(self, ic, oc, ks, use_bn, nl):
        # ic: input channels
        # oc: output channels
        # ks: kernel size
        # use_bn: True or False
        # nl: type of non-linearity, 'Non' or 'ReLU' or 'Sigmoid'
        super(ConvBlock, self).__init__()
        assert ks in [1, 3, 5, 7]
        assert isinstance(use_bn, bool)
        assert nl in ['Non', 'ReLU', 'Sigmoid']
        self.use_bn = use_bn
        self.nl = nl
        if ks == 1:
            self.conv = nn.Conv2d(ic, oc, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(ic, oc, kernel_size=ks, padding=(ks-1)//2, bias=False)
        if self.use_bn == True:
            self.bn = nn.BatchNorm2d(oc)
        if self.nl == 'ReLU':
            self.ac = nn.ReLU(inplace=True)
        if self.nl == 'Sigmoid':
            self.ac = nn.Sigmoid()
    def forward(self, x):
        y = self.conv(x)
        if self.use_bn == True:
            y = self.bn(y)
        if self.nl != 'Non':
            y = self.ac(y)
        return y
    
    
class DecodingUnit(nn.Module):
    def __init__(self, deep_channels, shallow_channels, dec_channels, inter_ks, is_upsample):
        super(DecodingUnit, self).__init__()
        assert isinstance(is_upsample, bool)
        self.is_upsample = is_upsample
        self.unit_conv = ConvBlock(deep_channels+shallow_channels, dec_channels, 1, True, 'ReLU')
        conv_1 = ConvBlock(dec_channels, dec_channels//4, 3, True, 'ReLU')
        conv_2 = ConvBlock(dec_channels//4, dec_channels//4, inter_ks, True, 'ReLU')
        conv_3 = ConvBlock(dec_channels//4, dec_channels, 3, True, 'ReLU')
        self.bottle_neck = nn.Sequential(conv_1, conv_2, conv_3)
    def forward(self, deep_ftr, shallow_ftr):
        if self.is_upsample:
            deep_ftr = US2(deep_ftr)
        concat_ftr = torch.cat((deep_ftr, shallow_ftr), dim=1)
        inter_ftr = self.unit_conv(concat_ftr)
        dec_ftr = self.bottle_neck(inter_ftr)
        return dec_ftr
 
    
class SalHead(nn.Module):
    def __init__(self, in_channels, inter_ks):
        super(SalHead, self).__init__()
        self.conv_1 = ConvBlock(in_channels, in_channels//2, inter_ks, False, 'ReLU')
        self.conv_2 = ConvBlock(in_channels//2, in_channels//2, 3, False, 'ReLU')
        self.conv_3 = ConvBlock(in_channels//2, in_channels//8, 3, False, 'ReLU')
        self.conv_4 = ConvBlock(in_channels//8, 1, 1, False, 'Non')
    def forward(self, dec_ftr):
        dec_ftr_ups = US2(dec_ftr)
        outputs = self.conv_4(self.conv_3(self.conv_2(self.conv_1(dec_ftr_ups))))
        
        return outputs

class RA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RA, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 3, padding=1),
        )
        self.channel = out_channel

    def forward(self, x, y):
        a = torch.sigmoid(-y)
        x = self.convert(x)
        x = a.expand(-1, self.channel, -1, -1).mul(x)   
        y = y + self.convs(x)

        return y
    
class MSCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSCM, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.ReLU(True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=2, dilation=2), nn.ReLU(True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=4, dilation=4), nn.ReLU(True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 7, padding=3, dilation=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=6, dilation=6), nn.ReLU(True),
        )
        self.score = nn.Conv2d(out_channel*4, 1, 3, padding=1)

    def forward(self, x):
        x = self.convert(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x = torch.cat((x1, x2, x3, x4), 1)  #n*c*h*w   n代表batch数目
        x = self.score(x)   #B*1*h*w

        return x

class RA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RA, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 3, padding=1),
        )
        self.channel = out_channel

    def forward(self, x, y):
        a = torch.sigmoid(-y)
        x = self.convert(x)
        x = a.expand(-1, self.channel, -1, -1).mul(x)   
        y = y + self.convs(x)

        return y

class RAS(nn.Module):
    def __init__(self, channel=64):
        super(RAS, self).__init__()
        #self.DAFnet = DAFNet()
        self.mscm = MSCM(768, channel)
        self.ra1 = RA(256, channel)
        self.ra2 = RA(384, channel)
        self.ra3 = RA(512, channel)
        #self.ra4 = RA(512, channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        #self.initialize_weights()

    def forward(self, D1, D2, D3, sm4):
           
        #x5=7x7x2048   
        #x4=14x14x1024  D4=14x14x768
        #x3=28x28x512   D3=28X28X512
        #x2=56x56x256   D2=56X56X384
        #x1=112x112x64  D1=112X112X256
        '''     
        #x_size = x.size()[2:]  #取图片大小
        #x1_size = x1.size()[2:]
        #x2_size = x2.size()[2:]
        #x3_size = x3.size()[2:]
        #x4_size = x4.size()[2:]
        '''
        D1_size = D1.size()[2:]   #112*112
        D2_size = D2.size()[2:]   #56*56
        D3_size = D3.size()[2:]   #28*28
        #D4_size = D4.size()[2:]   #14*14
        

        y4 = sm4
        #y4 = self.mscm(D4)  #B*1*14*14
        #score5 = F.interpolate(y5, x_size, mode='bilinear', align_corners=True)

        D4_3 = F.interpolate(y4, D3_size, mode='bilinear', align_corners=True)
        y3 = self.ra3(D3, D4_3) #b*1*28*28
        #score4 = F.interpolate(y4, x_size, mode='bilinear', align_corners=True)

        D3_2 = F.interpolate(y3, D2_size, mode='bilinear', align_corners=True)
        y2 = self.ra2(D2, D3_2) #b*1*56*56
        #score3 = F.interpolate(y3, x_size, mode='bilinear', align_corners=True)

        #y3_2 = F.interpolate(y3, x2_size, mode='bilinear', align_corners=True)
        #y2 = self.ra2(x2, y3_2)
        #score2 = F.interpolate(y2, x_size, mode='bilinear', align_corners=True)

        D2_1 = F.interpolate(y2, D1_size, mode='bilinear', align_corners=True)
        y1 = self.ra1(D1, D2_1)  #b*1*112*112
        score = US2(y1) #b*1*224*224
        #score1 = F.interpolate(y1, x_size, mode='bilinear', align_corners=True)

        #concat_ftr = torch.cat((deep_ftr, shallow_ftr), dim=1)
        
        return score  #, score2, score3, score4, score5

    #def initialize_weights(self):
        #res50 = models.resnet50(pretrained=True)
        #self.resnet.load_state_dict(res50.state_dict(), False)
       
class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3,
                              padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x  
    