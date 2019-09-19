import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import torch.nn.modules.normalization as norms

import numpy as np
import pdb

class unetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv3, self).__init__()

        if is_batchnorm:
            if in_size<64:
                nGrps = 4
            else:
                nGrps = 8#16 
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, 3, 1, 1),
                                       norms.GroupNorm(nGrps,out_size),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, 3, 1, 1),
                                       norms.GroupNorm(nGrps,out_size),
                                       nn.ReLU(),)
#            self.conv3 = nn.Sequential(nn.Conv3d(out_size, out_size, 3, 1, 1),
#                                       norms.GroupNorm(nGrps,out_size),
#                                       nn.ReLU(),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, 3, 1, 2),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, 3, 1, 2),
                                       nn.ReLU(),)
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
#        outputs = self.conv3(outputs)
        return outputs

class Combiner(nn.Module):
    '''
    Combines outputs of two layers by padding (if needed) and concatenation.
    '''
    def __init__(self):
        super(Combiner,self).__init__()

    def getPadding(self,offset):
        if offset%2==0:
            padding = 2*[np.sign(offset)*(np.abs(offset) // 2)]
        else:
            padding = [np.sign(offset)*(np.abs(offset) // 2),np.sign(offset)*( (np.abs(offset) // 2) + 1)]
        return padding
    
    def forward(self,input1,input2):
        '''
        input1 - from decoder ; input2 - from encoder.
        '''
        offset1 = input2.size()[2] - input1.size()[2]
        padding1 = self.getPadding(offset1)
        offset2 = input2.size()[3] - input1.size()[3]
        padding2 = self.getPadding(offset2)
        padding = padding2+padding1
        output1 = F.pad(input1, padding)
        return torch.cat([output1, input2], 1)

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        if nChannels<64:
            nGrps = 4
        else:
            nGrps = 8#16 
        self.bn1 = norms.GroupNorm(nGrps,nChannels)
#        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv3d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)
        self.dnConv1 = nn.Conv3d(nOutChannels,nOutChannels,2,stride=2)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dnConv1(out)
#        out = F.avg_pool2d(out, 2)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
#        self.bn1 = nn.BatchNorm2d(nChannels)
        if nChannels<64:
            nGrps = 4
        else:
            nGrps = 8#16 
        self.bn1 = norms.GroupNorm(nGrps,nChannels)
        self.conv1 = nn.Conv3d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=True)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=0, bias=True)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = F.pad(out,(1,1,1,1),mode='replicate')
        out = torch.cat((x, out), 1)
        return out

def _make_dense(nChannels, growthRate, nDenseBlocks, bottleneck):
    layers = []
    for i in range(int(nDenseBlocks)):
        if bottleneck:
            layers.append(Bottleneck(nChannels, growthRate))
        else:
            layers.append(SingleLayer(nChannels, growthRate))
        nChannels += growthRate
    return nn.Sequential(*layers)

def toCategorical(batch_size,yArr,nClasses,dims):
        if dims==3:
            y_OH = torch.FloatTensor(batch_size,nClasses,yArr.shape[2],yArr.shape[3],yArr.shape[4])
        elif dims==1:
            y_OH = torch.FloatTensor(batch_size,nClasses)
        y_OH.zero_()
        y_OH.scatter_(1,yArr,1)
        return y_OH

def contrastiveLoss(a,b,y,m,reduction='sum',gpuID=0):
    '''
    Contrastive loss LC=∑N,n=1 (y)d^2+ (1−y) max(0,m−d)^2 where d is L2 distance b/w a and b
    a, b = final embeddings of the pair
    y = label
    m = margin ; If distance is greater than margin, loss becomes 0 for negative pair
    '''
    d = torch.norm(a-b)
    loss = y*(d**2) + (1-y)*torch.max(torch.Tensor([0]).cuda(gpuID), (m-d) )**2
    if reduction=='sum':
        loss = torch.sum(loss)
    return loss


class myBCELoss(nn.Module):
    '''
    Weighted Binary cross entropy loss. Tested.
    '''
    def __init__(self,weight):
        super(myBCELoss,self).__init__()
        self.weight = weight

    def forward(self,inputs,target):
        normVal = 1e-24
#        target = target[:,1,:,:,:]
#        inputs = inputs[:,1,:,:,:]
        weights = 1 + (self.weight-1)*target # to make pos wt as self.weight and others as 1
        loss = -((weights*target)*inputs.clamp(min=normVal).log()+(1-target)*(1-inputs).clamp(min=normVal).log()).mean()
        return loss 

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def integralDice(pred,gt,k):
    '''
    Dice coefficient for multiclass hard thresholded prediction consisting of integers instead of binary
    values. k = integer for class for which Dice is being calculated.
    '''
    return torch.sum(pred[gt==k]==k)*2.0 / (torch.sum(pred[pred==k]==k) + torch.sum(gt[gt==k]==k)).float()

def compare_models(model_1, model_2):
    '''
    Copied from https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/6 
    '''
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):    # if names are same but weights are different
                print('Mismatch found at', key_item_1[0])
            else:
                print(key_item_1[0]+' layer found in model 1 but layer '+key_item_2[0]+' found in model 2 at this place.') 
#                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

def getClassWts(nBatches,trainGenObj):
    wtList = []
    ct = 0
    for i in range(2,3):
        weight = 0
        for j in range(nBatches):
            vol,labels,case,_ = trainGenObj.__next__()
            if 2 in torch.unique(labels):
                wt = torch.sum(labels[0,0,:,:,:]==i).float() / (labels.shape[2]*labels.shape[3]*labels.shape[4])
                weight+=(1/wt)
                ct+=1
            # print(case+': '+str(1/wt)+' Average so far: '+str(weight.float()/j))
        avWt = weight/float(ct)#float(j-1)#285
        wtList.append(avWt)
        # timeTaken = time.time() - initTime
        print(wtList)
    return wtList