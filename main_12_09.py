import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pdb
import sys
from tqdm import trange

import numpy as np
import SimpleITK as sitk

from dataHandlers import dataHandler
from networks import dense_unet_encoder as encoder #SimpleNetEncoder as encoder
from networks import dense_unet_decoder as decoder, dense_unet_autoencoder as autoencoder 
from utils import myBCELoss, toCategorical, dice_coeff, integralDice, compare_models, contrastiveLoss


def globalAcc(predList,labelList):
    '''
    Compute accuracy over all samples using list of predictions and labels.
    '''
    predList = torch.cat(predList)
    labelList = torch.cat(labelList)
    acc = torch.sum(predList==labelList).float()/( predList.shape[0] )
    return acc    

def saveVolume(vol,fileName):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(fileName)
    if isinstance(vol,torch.Tensor):
        if vol.requires_grad:
            vol = vol.detach()
        if vol.is_cuda:
            vol = vol.cpu()
        vol = vol.numpy()
        if vol.dtype=='int64':
            vol = vol.astype('uint8')
    writer.Execute(sitk.GetImageFromArray(vol.swapaxes(0,2)))

def predictAndGetLoss(model,X,y,batchSize,taskType):
    '''
    Generate predictions and calculate loss as well as skeleton info for metric,
    as per the given task - binary classification, siamese n/w classification or segmentation
    '''
    predListBatch = [] ; labelListBatch = []
    if taskType=='classifyDirect' or taskType=='classifySiamese':
        dims = 1
    elif taskType=='segment':
        dims = 3
    yOH = toCategorical(batchSize,y.cpu(),2,dims).cuda()
    if taskType=='classifyDirect':
        out = model.forward(X)
        loss = 0
        for i in range(2):
            loss += myBCELoss(1).forward(out[:,i],yOH[:,i])            
        pred = torch.argmax(out,1)
        predListBatch.append(pred.reshape(pred.shape[0]).cpu())
        labelListBatch.append(y.reshape(y.shape[0]).cpu())
        dataForMetric = (predListBatch,labelListBatch)
        return loss, dataForMetric
    elif taskType=='classifySiamese':
        out1,_,_,_ = model.forward(X[0].unsqueeze(0))
        out2,_,_,_ = model.forward(X[1].unsqueeze(0))
        if y[0]==y[1]:
            pairLabel = 1
        else:
            pairLabel = 0 
        loss = contrastiveLoss(out1,out2,pairLabel,1)
        return loss
    elif taskType=='segment':
        out,recons = model.forward(X)
        pred = torch.argmax(out,1)
        loss = 0
        diceCoeff = 0
        for i in range(2):
            lossDice = ( 1-dice_coeff(out[:,i,:,:,:].float(),yOH[:,i,:,:,:].float()) )
            lossMSE = F.mse_loss(recons,X)
            loss += (0.95*lossDice+0.05*lossMSE)
            if i>0:
                diceCoeff += integralDice(pred.float().detach().cpu(),y[:,0,:,:,:].float().detach(),i)
        return loss,lossDice.item(),lossMSE.item(),diceCoeff

def train(model,genObj,optimizer,scheduler,epoch,batchSize,nBatches,taskType):
    runningLoss = 0.0
    runningLossDice = 0.0
    runningLossMSE = 0.0
    runningDice = 0.0
    predList = []
    labelList = []
    model.train()
    with trange(nBatches,desc='Epoch '+str(epoch+1),ncols=100) as t:
        for m in range(nBatches):
#            pdb.set_trace()
            X,y,_,_ = genObj.__next__()
            optimizer.zero_grad()
            if taskType=='classifyDirect':
                loss, dataForMetric = predictAndGetLoss(model,X,y,batchSize,taskType)
                predList.extend(dataForMetric[0])
                labelList.extend(dataForMetric[1])
            elif taskType=='classifySiamese':
                loss = predictAndGetLoss(model,X,y,batchSize,taskType)
            elif taskType=='segment':
                loss, lossDice, lossMSE, dataForMetric = predictAndGetLoss(model,X,y,batchSize,taskType)
                runningDice += dataForMetric
                runningLossDice += lossDice
                runningLossMSE += lossMSE
            runningLoss += loss.item()
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=runningLoss/(float(m+1)*batchSize))
            t.update()
    if taskType=='classifyDirect':
        acc = globalAcc(predList,labelList)
        print('Epoch num. %d \t Trn. Loss : %.7f ; \t Trn. Acc : %.3f' %(epoch+1,runningLoss/( (m+1)*batchSize), acc.item() ))
    elif taskType=='classifySiamese':
        print('Epoch num. %d \t Trn. Loss : %.7f ; \t ' %(epoch+1,runningLoss/( (m+1)*batchSize) ))
    elif taskType=='segment':
        # print(runningDice)
        batchLoss = runningLoss/( (m+1)*batchSize)
        batchLossDice = runningLossDice / ( (m+1)*batchSize)
        batchLossMSE = runningLossMSE / ( (m+1)*batchSize)
        #scheduler.step(batchLoss)
        dice = runningDice / (m+1)
        print('Epoch num. %d Trn. Loss : %.7f ; \t Trn. DiceLoss : %.7f ; \t Trn. MSELoss : %.7f ; \t Trn. Dice : %.3f' %(epoch+1, batchLoss, batchLossDice, batchLossMSE, dice ))        

def validate(model,genObj,epoch,batchSize,nBatches,taskType,dh):
    runningLoss = 0.0
    runningDice = 0.0
    runningLossDice = 0.0
    runningLossMSE = 0.0
    predList = []
    labelList = []
    model.eval()
    for m in range(nBatches):
        # pdb.set_trace()
        X,y,case,direction = genObj.__next__()    
#        predictAndSave(X,case,model,dh)
#        print(direction+'side of '+case+' has dice coeff '+str(dataForMetric))
        if taskType=='classifyDirect':
            loss, dataForMetric = predictAndGetLoss(model,X,y,batchSize,taskType)
            predList.extend(dataForMetric[0])
            labelList.extend(dataForMetric[1])
        elif taskType=='classifySiamese':
            loss = predictAndGetLoss(model,X,y,batchSize,taskType)
        elif taskType=='segment':
            loss, lossDice, lossMSE, dataForMetric = predictAndGetLoss(model,X,y,batchSize,taskType)
            runningDice += dataForMetric
            runningLossDice += lossDice
            runningLossMSE += lossMSE
        runningLoss += loss.item()
    if taskType=='classifyDirect':
        acc = globalAcc(predList,labelList)
        print('Epoch num. %d \t Val. Loss : %.7f ; \t Val. Acc : %.3f' %(epoch+1,runningLoss/( (m+1)*batchSize), acc.item() ))
        return acc
    elif taskType=='classifySiamese':
        print('Epoch num. %d \t Val. Loss : %.7f ; \t ' %(epoch+1,runningLoss/( (m+1)*batchSize) ))
    elif taskType=='segment':
        dice = runningDice / (m+1)
        batchValLoss = runningLoss/( (m+1)*batchSize)
        batchLossDice = runningLossDice / ( (m+1)*batchSize)
        batchLossMSE = runningLossMSE / ( (m+1)*batchSize)
        print('Epoch num. %d Val. Loss : %.7f ; \t Val. DiceLoss : %.7f ; \t Val. MSELoss : %.7f ; \t Val. Dice : %.3f' %(epoch+1, batchValLoss , batchLossDice, batchLossMSE, dice ))   
        return dice


def predictAndSave(X,case,model,dh):
    out,_ = model.cuda(0).forward(X[0].cuda(0).unsqueeze(0))
    pred1 = torch.argmax(out.cpu(),1)
    leftPred = dh.cropResize(pred1[0],dh.leftSize) 
    out,_ = model.cuda(0).forward(X[1].cuda(0).unsqueeze(0))
    pred2 = torch.argmax(out.cpu(),1)
    rightPred = dh.cropResize(pred2[0],dh.rightSize)
    fullPred = torch.cat([rightPred,leftPred],-1)
    fullPredResized = dh.cropResize(fullPred,())
#    fullPredResized = dh.resizeToSize(fullPred[0],dh.origSize)
    del pred1
    del pred2
    del out
    del X
    saveVolume(fullPredResized,'valPreds_scratch_11Oct/prediction_'+case+'.nii.gz') 
    torch.cuda.empty_cache()

def test(model,genObj,dh,nBatches):
    model.eval()
#    pdb.set_trace()
    for m in range(nBatches):
        X, case, direction = genObj.__next__()
        predictAndSave(X,case,model,dh)
 
class DUN(nn.Module):
    '''
    Dense U-Net for segmentation.
    '''
    def __init__(self,encoder):
        super(DUN,self).__init__()
        self.encoder = encoder
        self.decoder = nn.DataParallel(decoder(2).cuda())
        self.autoEncoderModel = nn.DataParallel(autoencoder().cuda())

    def forward(self,x):
        x,c1_out,c2_out,c3_out = self.encoder(x)
        out = self.decoder(x,c1_out,c2_out,c3_out)
        recons = self.autoEncoderModel(x)
        return out,recons

class DenseClassifier(nn.Module):
    '''
    Classifier for tumor proxy task
    '''
    def __init__(self,encoder):
        super(DenseClassifier,self).__init__()
        self.encoder = encoder
        self.preClassifier = nn.DataParallel(nn.Conv3d(144,1,3,padding=1).cuda())
        self.classifier = nn.DataParallel(nn.Linear(196,2).cuda())

    def forward(self,x):
        feat,_,_,_ = self.encoder(x)
        feat = self.preClassifier(feat)
        feat = feat.view(-1,196)
        pred = F.softmax(self.classifier(feat),1)
        return pred

def main():
    '''
    Control section, contains all class/function calls. Use of separate main function allows isolation of variables.
    '''
    ## Control parameters.  
    ## Valid params : problemType = {'main','proxy'} ; taskType = {'classifyDirect','classifySiamese','segmentKidney','segmentTumor'}
    problemType = 'proxy' 
    taskType = 'classifyDirect'

    ## Constants and training parameters
    batchSize = 2
    nSamples = 207
    valSize = 30
    nTrnBatches = ((nSamples - valSize)*2)//batchSize                              # *2 since volumes are loaded in halves
    nValBatches = (valSize*2)//batchSize

    testBatchSize = 2
    nTestSamples = 90
    nTestBatches = nTestSamples // testBatchSize

    nEpochs = 10
    lr = 1e-3
    weightDecay = 1e-5
    multipleOf = 16                                                                 # volumes should have size multiple of this number
    initEpochNum = int(sys.argv[1])                                                 # Starting epoch number, for display in progress

    ## Paths
    trnPath = '/scratch/abhinavdhere/kits_train/'
    valPath = '/scratch/abhinavdhere/kits_val/'
    currentBestRecord = 'bestVal_tumorProxy.txt'                                    # Stores dice for best performance so far
    testPath = '/scratch/abhinavdhere/kits_test/'
    saveName = 'proxy_tumor_classify.pt' #   loadName = 'proxy_kidney_siamese.pt' #'kidneyOnlySiamese.pt'
#    saveName = #'self_tumor_dense_wAE_revisited.pt'
    # saveName = 'self_tumor_dense_wAE.pt'
#    saveName = 'scratch_tumor_dense_wAE.pt'
#    saveName = 'self_dense_wAE_wAug.pt'
    # saveName =  'proxy_kidney_siamese.pt'#'scratch_tumor_dense.pt' #'selfSiamese.pt' # 'models/segKidney.pt'
    ## Construct appropriate model, optimizer and scheduler. Get data loaders
    if os.path.exists(saveName):
        model = torch.load(saveName).cuda()
    else:
        encoderModel = nn.DataParallel(encoder().cuda())
        if problemType=='main':
            proxyModel = torch.load(loadName)
            pretrained_dict = proxyModel.state_dict()
            encoderModel.load_state_dict(pretrained_dict,strict=False)
            compare_models(proxyModel, encoderModel)                                # Ensure that weights are loaded
            del proxyModel                                                          # Cleanup to save memory
            del pretrained_dict   
            torch.cuda.empty_cache()
            model = DUN(encoderModel)
        elif problemType=='proxy':
            if taskType=='classifyDirect':
                model = DenseClassifier(encoderModel)
            elif taskType=='classifySiamese':
                model = encoderModel
    dh = dataHandler(trnPath,valPath,batchSize,valSize,multipleOf,gpuID)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weightDecay)
    scheduler = ReduceLROnPlateau(optimizer,factor=0.5,patience=3,verbose=True)

    trainDataLoader = dh.giveGenerator('train',problemType,taskType)
    valDataLoader = dh.giveGenerator('val',problemType,taskType)

    ## Learn and evaluate !!
    # valDice = validate(model,valDataLoader,0,batchSize,nValBatches,taskType,dh)
    if not os.path.exists(currentBestRecord):
        if problemType=='main':
            os.system('echo "Best Dice so far: 0.0" > '+currentBestRecord)
        elif problemType=='proxy':
            os.system('echo "Best Acc so far: 0.0" > '+currentBestRecord)
    statusFile = open(currentBestRecord,'r')
    bestValMetric = float(statusFile.readline().strip('\n').split()[-1])
    statusFile.close()
    for epoch in range(initEpochNum, initEpochNum+nEpochs):
        # train(model,trainDataLoader,optimizer,scheduler,epoch,batchSize,nTrnBatches,taskType)
        # torch.save(model,saveName)
        valMetric = validate(model,valDataLoader,epoch,batchSize,nValBatches,taskType,dh)
        if valMetric > bestValMetric:
            statusFile = open(currentBestRecord,'w')
            diff = valMetric - bestValMetric
            bestValMetric = valMetric
            # torch.save(model,saveName.split('.')[0]+'_chkpt_'+str(epoch+1)+'epoch.pt')
            if problemType=='main':
                statusFile.write('Best Dice so far: '+str(bestValMetric.item()))
                print('Model checkpoint saved since Dice has improved by '+str(diff))
            else:
                statusFile.write('Best Acc so far: '+str(bestValMetric.item()))
                print('Model checkpoint saved since Acc has improved by '+str(diff))
            statusFile.close()
 
#    testDh = dataHandler(testPath,'',testBatchSize,valSplit=0,dataShapeMultiple=16,gpuID=gpuID)
#    testDataLoader = testDh.giveGenerator('test',problemType)
#    test(model,testDataLoader,testDh,nTestBatches) 
    # test(model,testDataLoader,0,batchSize,nTestBatches,testDh)

if __name__ == '__main__':
    gpuID = 0
    main()

## ---------------------- Graveyard -------------------------
    ## GradCam  
    # vol,label = valDataLoader.__next__()
    # 
    # gradCamObj = GradCam(model,target_layer_names=['conv8'] ,use_cuda=True)
    # mask = gradCamObj(vol,label.item()) 

    ## IG
    # pdb.set_trace()
    # X, labels = valDataLoader.__next__()
    # baseline = np.random.uniform(low=-71,high=304,size=X.shape)
    # integrated_grad = integrated_gradients(X.detach().cpu().numpy(), model, labels.detach().cpu().numpy(), \
    #     calculate_outputs_and_gradients, baseline=baseline, steps=100, cuda=True)

        # trainSegment(model,trainDataLoader,optimizer,epoch,batchSize,nTrnBatches)
        # validateSegment(model,valDataLoader,batchSize,nValBatches)

# def trainClassify(model,genObj,optimizer,epoch,batchSize,nBatches):
#     runningLoss = 0.0
#     predList = []
#     labelList = []
#     with trange(nBatches,desc='Epoch '+str(epoch+1),ncols=100) as t:
#         for m in range(nBatches):
#             X,y = genObj.__next__()
#             # pdb.set_trace()
#             yOH = toCategorical(batchSize,y.cpu(),2,1).cuda(gpuID)
#             optimizer.zero_grad()
#             out,_,_,_ = model.forward(X)
#             # loss = F.binary_cross_entropy(out,yOH,reduction='sum')
#             loss = 0
#             for i in range(2):
#                 loss += myBCELoss(1).forward(out[:,i],yOH[:,i])
#             loss.backward()
#             optimizer.step()
#             pred = torch.argmax(out,1)
#             predList.append(pred.reshape(pred.shape[0]).cpu())
#             labelList.append(y.reshape(y.shape[0]).cpu())
#             runningLoss += loss.item()
#             t.set_postfix(loss=runningLoss/(float(m+1)*batchSize))
#             t.update()
#         acc = globalAcc(predList,labelList)
#     print('Epoch num. %d \t Trn. Loss : %.7f ; \t Trn. Acc : %.3f' %(epoch+1,runningLoss/( (m+1)*batchSize), acc.item() ))

# def validateClassify(model,genObj,batchSize,nBatches):
#     runningLoss = 0.0
#     predList = []
#     labelList = []
#     for m in range(nBatches):
#         X,y = genObj.__next__()
#         yOH = toCategorical(batchSize,y.cpu(),2,1).cuda(gpuID)
#         model.eval()
#         out,_,_,_ = model.forward(X)
#         loss = 0
#         for i in range(2):
#             loss += myBCELoss(1).forward(out[:,i],yOH[:,i])
#         pred = torch.argmax(out,1)
#         predList.append(pred.reshape(pred.shape[0]).cpu())
#         labelList.append(y.reshape(y.shape[0]).cpu())
#         runningLoss += loss.item()
#     acc = globalAcc(predList,labelList)
#     print('\t Val. Loss : %.7f ; \t Val. Acc : %.3f' %(runningLoss/((m+1)*batchSize), acc.item() ))

# def trainSegment(model,genObj,optimizer,epoch,batchSize,nBatches):
#     runningLoss = 0.0
#     runningDice = 0.0
#     predList = []
#     labelList = []
#     with trange(nBatches,desc='Epoch '+str(epoch+1),ncols=100) as t:
#         for m in range(nBatches):
#             X,y = genObj.__next__()
#             # pdb.set_trace()
#             yOH = toCategorical(batchSize,y.cpu(),2,3).cuda(gpuID)
#             optimizer.zero_grad()
#             out = model.forward(X)
#             pred = torch.argmax(out,1)
#             # loss = F.binary_cross_entropy(out,yOH,reduction='sum')
#             loss = 0
#             diceCoeff = 0
#             for i in range(2):
#                 loss += ( 1-dice_coeff(out[:,i,:,:,:].float(),yOH[:,i,:,:,:].float()) )
#                 if i>0:
#                     diceCoeff += integralDice(pred.float().detach().cpu(),y[:,0,:,:,:].float().detach(),i)
#             loss.backward()
#             optimizer.step()
#             runningLoss += loss.item()
#             runningDice += diceCoeff
#             t.set_postfix(loss=runningLoss/(float(m+1)*batchSize))
#             t.update()
#     dice = runningDice / (m+1)
#     print('Epoch num. %d \t Trn. Loss : %.7f ; \t Trn. Dice : %.3f' %(epoch+1,runningLoss/( (m+1)*batchSize), dice ))

# def validateSegment(model,genObj,batchSize,nBatches):
#     runningLoss = 0.0
#     runningDice = 0.0
#     predList = []
#     labelList = []
#     for m in range(nBatches):
#         X,y = genObj.__next__()
#         yOH = toCategorical(batchSize,y.cpu(),2,3).cuda(gpuID)
#         model.eval()
#         out = model.forward(X)
#         pred = torch.argmax(out,1)
#         loss = 0
#         diceCoeff = 0
#         for i in range(2):
#             loss += ( 1-dice_coeff(out[:,i,:,:,:].float(),yOH[:,i,:,:,:].float()) )
#             if i>0:
#                 diceCoeff += integralDice(pred.float().detach().cpu(),y[:,0,:,:,:].float().detach(),i)
#         runningLoss += loss.item()
#         runningDice += diceCoeff
#     dice = runningDice / (m+1)
#     print('\t Val. Loss : %.7f ; \t Val. Dice : %.3f' %(runningLoss/((m+1)*batchSize), dice ))

