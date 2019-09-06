import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import unetConv3, Combiner, _make_dense, SingleLayer, Transition

class SimpleNetEncoder(nn.Module):
    def __init__(self):
        super(SimpleNetEncoder,self).__init__()

        self.conv1 = nn.Conv3d(1,32,kernel_size=3,stride=2)
        self.conv2 = nn.Conv3d(32,32,kernel_size=3,stride=2)

        # self.conv3 = nn.Conv3d(32,64,kernel_size=3,stride=2)
        # self.conv4 = nn.Conv3d(64,64,kernel_size=3,stride=1)
        # self.conv5 = nn.Conv3d(64,64,kernel_size=3,stride=2)

        # self.conv5 = nn.Conv3d(64,128,kernel_size=3)
        # self.conv6 = nn.Conv3d(128,128,kernel_size=3,stride=1)
        # self.conv7 = nn.Conv3d(128,128,kernel_size=3,stride=2)

        self.adaptiveAP = nn.AdaptiveAvgPool3d((1,1,1)) # (8,16,16)) 
        self.conv8 = nn.Conv3d(32,2,kernel_size=1)
        # self.final = nn.Linear(8*16*16,1)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))

        # x = F.relu(self.conv8(x))
        x = self.adaptiveAP(x)
        x = F.softmax(self.conv8(x),1)
        out = x.view(-1,2)
        # x = x.view(-1,8*16*16)
        # out = torch.sigmoid(self.final(x))
        return out

class dense_unet_encoder(nn.Module):
    def __init__(self):
        super(dense_unet_encoder,self).__init__() 
        growthRate = 16 ; nFeat = 1 ; nLayers = 4
        isBottleneck = False
        self.inputLayer = nn.Conv3d(nFeat,16,3,stride=2,padding=1) ; nFeat = 16#(3,3,nSlices) 
        self.conv1 = _make_dense(nFeat,8,2,isBottleneck) ; nFeat += 8*2#nLayers
        self.trans1 = Transition(nFeat,16) ; nFeat = 16
        self.conv2 = _make_dense(nFeat,16,4,isBottleneck) ; nFeat += growthRate*nLayers
        self.trans2 = Transition(nFeat,16) ; nFeat = 16
        self.conv3 = _make_dense(nFeat,16,4,isBottleneck) ; nFeat += growthRate*nLayers
        self.trans3 = Transition(nFeat,16) ; nFeat = 16
        self.center = _make_dense(nFeat,16,8,isBottleneck) ; nFeat += growthRate*nLayers*2
        # self.adaptiveAP = nn.AdaptiveAvgPool3d((1,1,1))
        # self.final = nn.Conv3d(nFeat,2,kernel_size=1)
        # self.adaptiveAP = nn.AdaptiveAvgPool3d((8,16,16))
        # self.final = nn.Linear(nFeat*8*16*16,1) 
        # self.finalConvFeat = nFeat

    def forward(self,x):
        x = self.inputLayer(x)
        c1_out = self.trans1(self.conv1(x))
        c2_out = self.trans2(self.conv2(c1_out))
        c3_out = self.trans3(self.conv3(c2_out))
        x = self.center(c3_out)
        # x = self.adaptiveAP(x)
        ## x = x.view(-1,self.finalConvFeat*8*16*16)
        # out = F.softmax(self.final(x),1)        
        # out = out.view(-1,2)
        return x,c1_out,c2_out,c3_out

class dense_unet_decoder(nn.Module):
    def __init__(self):
        super(dense_unet_decoder,self).__init__() 
        self.combiner = Combiner()
        self.conv4 = unetConv3(144+16,128,True)
        self.upTrans2 = nn.ConvTranspose3d(128,64,2,2) 
        self.conv5 = unetConv3(64+16,32,True)
        self.upTrans3 = nn.ConvTranspose3d(32,16,2,2)
        self.conv6 = unetConv3(16+16,16,True)
        self.upTrans4 = nn.ConvTranspose3d(16,16,2,2)
        self.final = nn.Conv3d(16,2,kernel_size=1)

    def forward(self,x,c1_out,c2_out,c3_out):
        x = self.conv4(self.combiner(c3_out,x))
        x = self.conv5(self.combiner(c2_out,self.upTrans2(x)))
        x = self.conv6(self.combiner(c1_out,self.upTrans3(x)))
        x = self.upTrans4(x)
        x = self.upTrans4(x)
        out = F.softmax(self.final(x),1)
        return out
