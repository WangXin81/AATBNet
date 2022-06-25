from moduless import *
from attention import *




    
class Encoder(nn.Module):
    def __init__(self, init_path):
        super(Encoder, self).__init__()
        ch = [64, 128, 256, 512, 512]
        dr = [2, 4, 8, 16, 16] 
        bb = torch.load(os.path.join(init_path, 'bb.pth'))
        self.C1, self.C2, self.C3, self.C4, self.C5 = bb.C1, bb.C2, bb.C3, bb.C4, bb.C5
        self.GCA2 = GCA(ch[1], True)
        self.GCA3 = GCA(ch[2], True)
        self.GCA4 = GCA(ch[3], False)
        self.GCA5 = GCA(ch[4], False)
        self.AF3 = AttentionFusion(2)
        self.AF4 = AttentionFusion(3)
        self.AF5 = AttentionFusion(4)
    def forward(self, Im): # [3, 224, 224]
        # stage-1
        F1 = self.C1(Im) # [64, 112, 112]
        # stage-2
        F2 = self.C2(F1) # [128, 56, 56]
        A2, F2 = self.GCA2(F2) # [1, 56, 56] & [128, 56, 56]
        F2 = RC(F2, A2) # [128, 56, 56]
        # stage-3
        F3 = self.C3(F2) # [256, 28, 28]
        A3, F3 = self.GCA3(F3) # [1, 28, 28] & [256, 28, 28]
        A3 = self.AF3(torch.cat([A3, DS2(A2)], dim=1)) # [1, 28, 28]
        F3 = RC(F3, A3) # [256, 28, 28]
        # stage-4
        F4 = self.C4(F3) # [512, 14, 14]
        A4, F4 = self.GCA4(F4) # [1, 14, 14] & [512, 14, 14]
        A4 = self.AF4(torch.cat([A4, DS2(A3), DS4(A2)], dim=1)) # [1, 14, 14]
        F4 = RC(F4, A4) # [512, 14, 14]
        # stage-5
        F5 = self.C5(F4) # [512, 14, 14]
        A5, F5 = self.GCA5(F5) # [1, 14, 14] & [512, 14, 14]
        A5 = self.AF5(torch.cat([A5, A4, DS2(A3), DS4(A2)], dim=1)) # [1, 14, 14]
        F5 = RC(F5, A5) # [512, 14, 14]
        # F1: [B, 64, 112, 112]
        # F2: [B, 128, 56, 56]
        # F3: [B, 256, 28, 28]
        # F4: [B, 512, 14, 14]
        # F5: [B, 512, 14, 14]
        return F1, F2, F3, F4, F5
'''
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        ch = [64, 128, 256, 512, 512]
        dr = [2, 4, 8, 16, 16]
        dec_ch = [256, 384, 512, 768]
        self.du_5 = DecodingUnit(ch[4], ch[3], dec_ch[3], 3, False)
        self.du_4 = DecodingUnit(dec_ch[3], ch[2], dec_ch[2], 5, True)
        self.du_3 = DecodingUnit(dec_ch[2], ch[1], dec_ch[1], 5, True)
        self.du_2 = DecodingUnit(dec_ch[1], ch[0], dec_ch[0], 7, True)
       
        self.ra1 = RA(64, channel)
        self.ra2 = RA(256, channel)
        self.ra3 = RA(512, channel)
        self.ra4 = RA(1024, channel)
        
    def forward(self, F1, F2, F3, F4, F5):                  
          
        #x5=7x7x2048   F5=14X14X512
        #x4=14x14x1024  F4=14x14x512
        #x3=28x28x512   F3=28X28X256
        #x2=56x56x256   F2=56X56X128
        #x1=112x112x64  F1=112X112X64
       
        x_size = x.size()[2:]  #取原图片大小  224*224
        x1_size = x1.size()[2:]
        x2_size = x2.size()[2:]
        x3_size = x3.size()[2:]
        x4_size = x4.size()[2:]

        #y5 = self.mscm(x5)
        #score5 = F.interpolate(y5, x_size, mode='bilinear', align_corners=True)

        y5_4 = F.interpolate(y5, x4_size, mode='bilinear', align_corners=True)
        y4 = self.ra4(x4, y5_4)   #y4=14*14*1
        #score4 = F.interpolate(y4, x_size, mode='bilinear', align_corners=True)

        y4_3 = F.interpolate(y4, x3_size, mode='bilinear', align_corners=True)
        y3 = self.ra3(x3, y4_3)  #y3=28*28*1
        #score3 = F.interpolate(y3, x_size, mode='bilinear', align_corners=True)

        y3_2 = F.interpolate(y3, x2_size, mode='bilinear', align_corners=True)
        y2 = self.ra2(x2, y3_2)  #y2=56*56*1
        #score2 = F.interpolate(y2, x_size, mode='bilinear', align_corners=True)

        y2_1 = F.interpolate(y2, x1_size, mode='bilinear', align_corners=True)
        y1 = self.ra1(x1, y2_1)  #y1=112*112*1
        #score1 = F.interpolate(y1, x_size, mode='bilinear', align_corners=True)

        #score1,2,3,4,5=224*224
        
        
        D4 = self.du_5(F5, F4)   
        D3 = self.du_4(D4, F3)
        D2 = self.du_3(D3, F2)
        D1 = self.du_2(D2, F1)
        # D1: [256, 112, 112]
        # D2: [384, 56, 56]
        # D3: [512, 28, 28]
        # D4: [768, 14, 14]
        return D1, D2, D3, D4
     
'''                            
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        ch = [64, 128, 256, 512, 512]
        dr = [2, 4, 8, 16, 16]
        dec_ch = [256, 384, 512, 768]
        self.du_5 = DecodingUnit(ch[4], ch[3], dec_ch[3], 3, False)
        self.du_4 = DecodingUnit(dec_ch[3], ch[2], dec_ch[2], 5, True)
        self.du_3 = DecodingUnit(dec_ch[2], ch[1], dec_ch[1], 5, True)
        self.du_2 = DecodingUnit(dec_ch[1], ch[0], dec_ch[0], 7, True)
    def forward(self, F1, F2, F3, F4, F5):                  
        D4 = self.du_5(F5, F4)            
        D3 = self.du_4(D4, F3)
        D2 = self.du_3(D3, F2)
        D1 = self.du_2(D2, F1)
        # D1: [256, 112, 112]
        # D2: [384, 56, 56]
        # D3: [512, 28, 28]
        # D4: [768, 14, 14]
        return D1, D2, D3, D4


    

class DAFNet(nn.Module):
    def __init__(self, return_loss, init_path):
        super(DAFNet, self).__init__()   
        ch = [64, 128, 256, 512, 512]
        dr = [2, 4, 8, 16, 16]
        dec_ch = [256, 384, 512, 768]
        self.return_loss = return_loss 
        self.encoder = Encoder(init_path)
        initiate(self.encoder, os.path.join(init_path, 'ei.pth'))
        self.decoder = Decoder()
        initiate(self.decoder, os.path.join(init_path, 'di.pth'))
        self.ras = RAS() #增加
        mh = SalHead(dec_ch[0], 7)
        eh = SalHead(dec_ch[0], 7)
        #mh1 = SalHead(dec_ch[0], 7)
        mh_4 = SalHead(dec_ch[3], 7)
        
        self.duc1 = DUC(768, 512*4)
        self.duc2 = DUC(512, 384*4)
        self.duc3 = DUC(384, 256*4)
        self.duc4 = DUC(256, 1*4)
        #self.duc5 = DUC(64, 64*2)

        #self.conv_tra = ConvBlock(3, 1, 1, False, 'Sigmoid')

        self.head = nn.ModuleList([mh,eh,mh_4]) #nn.ModuleList([mh, eh ,mh_4])   #modulelist 
        self.bce = nn.BCELoss()
        
        
       
        
    def forward(self, image, label, edge):
        F1, F2, F3, F4, F5 = self.encoder(image)
        D1, D2, D3, D4 = self.decoder(F1, F2, F3, F4, F5)
      
        sm = self.head[0](D1)   #
        se = self.head[1](D1)
        sm4 = self.head[2](D4)
        score = self.ras(D1,D2,D3,sm4)

        dfm1 = D3 + self.duc1(D4)   #12*512*28*28
        #out16 = self.out1(dfm1)

        dfm2 = D2 + self.duc2(dfm1)  #12*384*56*56
        #out8 = self.out2(dfm2)

        dfm3 = D1 + self.duc3(dfm2)  #12*256*112*112
        
        sm1 = self.duc4(dfm3)  #12*1*224*224
        
        #pp= torch.sigmoid(sm + sm1)   # 12*1*224*224
        pp = score+sm
        pp1 = se +sm1
        #pp = self.conv_tra(torch.cat((score,sm1, sm), 1))
        #pp = self.duc4(pp)
        #out4 = self.out3(dfm3_t)

        
        if self.return_loss:
            losses_list = self.compute_loss(label,edge,pp,pp1 )
            #losses_list = self.compute_loss(sm, se, label, edge, score )
            return pp,losses_list
            #return sm, se, losses_list
        else:
            features = []
            features.extend([F1,F2,F3,F4,F5,D4,D3,D2,sm,se,sm4,score,dfm1,dfm2,dfm3,sm1,pp,pp1,torch.sigmoid(pp)])
            return torch.sigmoid(pp),features  

            #return  torch.sigmoid(pp),se   #,se  #sm se
    '''
    def bceloss(self, pred, gt):
        bce  = F.binary_cross_entropy_with_logits(pred, gt, reduction='mean')

        pred  = torch.sigmoid(pred)
        inter = (pred*gt).sum(dim=(2,3))
        union = (pred+gt).sum(dim=(2,3))
        iou  = 1-(inter+1)/(union-inter+1)

        return (bce+iou).mean()
    '''
    def compute_loss(self,label,edge,pp,pp1 ):
        mask_loss = self.bce(torch.sigmoid(pp), label)
        edge_loss = self.bce(torch.sigmoid(pp1), edge)
        #re_loss = self.bce(pp1, label)
        total_loss = 0.8*mask_loss+0.2*edge_loss
        return [total_loss, mask_loss, edge_loss ]
        
 
        
        
