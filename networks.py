import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import unetConv3, Combiner, _make_dense, SingleLayer, Transition

class unetEncoder(nn.Module):
    def __init__(self):
        super(unetEncoder,self).__init__()
        seedChannels = 20
        self.inputLayer = nn.Conv3d(1,seedChannels,3,stride=2,padding=1)
        self.conv1 = unetConv3(seedChannels,seedChannels*2,True) ; seedChannels*=2
        self.trans1 = Transition(seedChannels,seedChannels)        
        self.conv2 = unetConv3(seedChannels,seedChannels*2,True) ; seedChannels*=2
        self.trans2 = Transition(seedChannels,seedChannels)
        self.conv3 = unetConv3(seedChannels,seedChannels*2,True) ; seedChannels*=2
        self.trans3 = Transition(seedChannels,seedChannels)
        self.conv4 = unetConv3(seedChannels,seedChannels*2,True) ; seedChannels=seedChannels*2 #seedChannels*=2
        self.trans4 = Transition(seedChannels,seedChannels)        
        self.center = unetConv3(seedChannels,seedChannels,True) ;

    def forward(self,x):
        x = self.inputLayer(x)
        c1 = self.trans1(self.conv1(x))
        c2 = self.trans2(self.conv2(c1))
        c3 = self.trans3(self.conv3(c2))
        c4 = self.trans4(self.conv4(c3))
        c5 = self.center(c4)
        return c5,c4,c3,c2,c1

class unetDecoder(nn.Module):
    def __init__(self,nClasses):
        super(unetDecoder,self).__init__()
        self.combiner = Combiner()
        seedChannels = 320#128
        self.conv4 = unetConv3(seedChannels*2,160,True) ; seedChannels = 160 #seedChannels//4,True)  ; seedChannels//=4
        self.upTrans4 = nn.ConvTranspose3d(seedChannels,seedChannels,2,2)        
        self.conv3 = unetConv3(seedChannels*2,seedChannels//2,True)           ; seedChannels//=2
        self.upTrans3 = nn.ConvTranspose3d(seedChannels,seedChannels,2,2)  
        self.conv2 = unetConv3(seedChannels*2,seedChannels//2,True)         ; seedChannels//=2
        self.upTrans2 = nn.ConvTranspose3d(seedChannels,seedChannels,2,2) 
        self.conv1 = unetConv3(seedChannels*2,seedChannels//2,True)         ; seedChannels//=2
        self.upTrans1 = nn.ConvTranspose3d(seedChannels,seedChannels,4,4)    # 4 times upsampling
        self.final = nn.Conv3d(seedChannels,nClasses,kernel_size=1)

    def forward(self,c5,c4,c3,c2,c1):
        # pdb.set_trace()
        x = self.conv4(self.combiner(c4,c5))
        x = self.conv3(self.combiner(c3,self.upTrans4(x)))
        x = self.conv2(self.combiner(c2,self.upTrans3(x)))
        x = self.conv1(self.combiner(c1,self.upTrans2(x)))
        x = F.softmax(self.final(self.upTrans1(x)),1)
        return x        



class dense_unet_encoder(nn.Module):
    def __init__(self):
        super(dense_unet_encoder,self).__init__() 
        growthRate = 16 ; nFeat = 1 ; nLayers = [2,2,4,8]
        isBottleneck = False
        self.inputLayer = nn.Conv3d(nFeat,4,3,stride=2,padding=1) ; nFeat = 4#(3,3,nSlices) 
        self.conv1 = _make_dense(nFeat,growthRate//4,nLayers[0],isBottleneck) ; nFeat += 4*nLayers[0]
        # Transition layer applies 1x1 conv, brings to given output channels (16) and applies stride 2 conv to downsample
        self.trans1 = Transition(nFeat,16) ; nFeat = 16                      
        self.conv2 = _make_dense(nFeat,growthRate,nLayers[1],isBottleneck) ; nFeat += growthRate*nLayers[1]
        self.trans2 = Transition(nFeat,16) ; nFeat = 16
        self.conv3 = _make_dense(nFeat,growthRate,nLayers[2],isBottleneck) ; nFeat += growthRate*nLayers[2]
        self.trans3 = Transition(nFeat,16) ; nFeat = 16
        self.center = _make_dense(nFeat,growthRate,nLayers[3],isBottleneck) ; nFeat += growthRate*nLayers[3]

    def forward(self,x):
        x = self.inputLayer(x)
        c1_out = self.trans1(self.conv1(x))
        c2_out = self.trans2(self.conv2(c1_out))
        c3_out = self.trans3(self.conv3(c2_out))
        x = self.center(c3_out)
        return x,c1_out,c2_out,c3_out

class dense_unet_decoder(nn.Module):
    def __init__(self,nClasses):
        super(dense_unet_decoder,self).__init__() 
        self.combiner = Combiner()
        self.conv4 = unetConv3(144+16,128,True)         # +16 due to added channels from encoder, always 16 due to transition layer
        self.upTrans2 = nn.ConvTranspose3d(128,64,2,2) 
        self.conv5 = unetConv3(64+16,32,True)
        self.upTrans3 = nn.ConvTranspose3d(32,16,2,2)
        self.conv6 = unetConv3(16+16,4,True)
        self.upTrans4 = nn.ConvTranspose3d(4,4,2,2)
        self.final = nn.Conv3d(4,nClasses,kernel_size=1)

    def forward(self,x,c1_out,c2_out,c3_out):
        x = self.conv4(self.combiner(c3_out,x))
        x = self.conv5(self.combiner(c2_out,self.upTrans2(x)))
        x = self.conv6(self.combiner(c1_out,self.upTrans3(x)))
        x = self.upTrans4(x)
        x = self.upTrans4(x)                # extra upsampling to compensate for input layer's stride 2
        out = F.softmax(self.final(x),1)
        return out

class dense_unet_autoencoder(nn.Module):
    def __init__(self):
        super(dense_unet_autoencoder,self).__init__() 
        self.conv1 = unetConv3(144,128,True)
        self.upTrans1 = nn.ConvTranspose3d(128,64,2,2) 
        self.conv2 = unetConv3(64,32,True)
        self.upTrans2 = nn.ConvTranspose3d(32,16,2,2)
        self.conv3 = unetConv3(16,16,True)
        self.upTrans3 = nn.ConvTranspose3d(16,16,4,4)
        self.recons = nn.Conv3d(16,1,kernel_size=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(self.upTrans1(x))
        x = self.conv3(self.upTrans2(x))
        x = self.upTrans3(x)
        x = self.recons(x)
        return x

# self.adaptiveAP = nn.AdaptiveAvgPool3d((1,1,1))
# self.final = nn.Conv3d(nFeat,2,kernel_size=1)
# self.adaptiveAP = nn.AdaptiveAvgPool3d((8,16,16))
# self.final = nn.Linear(nFeat*8*16*16,1) 
# self.finalConvFeat = nFeat
# x = self.adaptiveAP(x)
## x = x.view(-1,self.finalConvFeat*8*16*16)
# out = F.softmax(self.final(x),1)        
# out = out.view(-1,2)

# class SimpleNetEncoder(nn.Module):
#     def __init__(self):
#         super(SimpleNetEncoder,self).__init__()

#         self.conv1 = nn.Conv3d(1,32,kernel_size=3,stride=2)
#         self.conv2 = nn.Conv3d(32,32,kernel_size=3,stride=2)

#         # self.conv3 = nn.Conv3d(32,64,kernel_size=3,stride=2)
#         # self.conv4 = nn.Conv3d(64,64,kernel_size=3,stride=1)
#         # self.conv5 = nn.Conv3d(64,64,kernel_size=3,stride=2)

#         # self.conv5 = nn.Conv3d(64,128,kernel_size=3)
#         # self.conv6 = nn.Conv3d(128,128,kernel_size=3,stride=1)
#         # self.conv7 = nn.Conv3d(128,128,kernel_size=3,stride=2)

#         self.adaptiveAP = nn.AdaptiveAvgPool3d((1,1,1)) # (8,16,16)) 
#         self.conv8 = nn.Conv3d(32,2,kernel_size=1)
#         # self.final = nn.Linear(8*16*16,1)

#     def forward(self,x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         # x = F.relu(self.conv3(x))
#         # x = F.relu(self.conv4(x))
#         # x = F.relu(self.conv5(x))
#         # x = F.relu(self.conv6(x))

#         # x = F.relu(self.conv8(x))
#         x = self.adaptiveAP(x)
#         x = F.softmax(self.conv8(x),1)
#         out = x.view(-1,2)
#         # x = x.view(-1,8*16*16)
#         # out = torch.sigmoid(self.final(x))
#         return out

# class unetEncoder(nn.Module):
#     def __init__(self):
#         super(unetEncoder,self).__init__()
#         seedChannels = 4
#         self.inputLayer = nn.Conv3d(1,seedChannels,3,stride=2,padding=1)
#         self.conv1 = unetConv3(seedChannels,seedChannels*2,True) ; seedChannels*=2
#         self.trans1 = Transition(seedChannels,seedChannels)        
#         self.conv2 = unetConv3(seedChannels,seedChannels*2,True) ; seedChannels*=2
#         self.trans2 = Transition(seedChannels,seedChannels)
#         self.conv3 = unetConv3(seedChannels,seedChannels*2,True) ; seedChannels*=2
#         self.trans3 = Transition(seedChannels,seedChannels)        
#         self.center = unetConv3(seedChannels,seedChannels*2,True) ;

#     def forward(self,x):
#         x = self.inputLayer(x)
#         c1 = self.trans1(self.conv1(x))
#         x = self.conv2(c1)
#         c2 = self.conv3(self.trans2(x))
#         c3 = self.center(self.trans3(c2))
#         return c3,c2,c1

# class unetDecoder(nn.Module):
#     def __init__(self,nClasses):
#         super(unetDecoder,self).__init__()
#         self.combiner = Combiner()
#         seedChannels = 64#128
#         self.conv3 = unetConv3(seedChannels,seedChannels//2,True)         ; seedChannels//=2
#         self.upTrans3 = nn.ConvTranspose3d(seedChannels,seedChannels,2,2)  
#         self.conv2 = unetConv3(seedChannels*2,seedChannels//4,True)         ; seedChannels//=4
#         self.upTrans2 = nn.ConvTranspose3d(seedChannels,seedChannels,2,2) 
#         self.conv1 = unetConv3(seedChannels*2,seedChannels//2,True)         ; seedChannels//=2
#         self.upTrans1 = nn.ConvTranspose3d(seedChannels,seedChannels,4,4)   # 4 times upsampling
#         self.final = nn.Conv3d(seedChannels,nClasses,kernel_size=1)

#     def forward(self,c3,c2,c1):
#         x = self.conv3(c3)
#         x = self.conv2(self.combiner(c2,self.upTrans3(x)))
#         x = self.conv1(self.combiner(c1,self.upTrans2(x)))
#         x = F.softmax(self.final(self.upTrans1(x)),1)
#         return x        