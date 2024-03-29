import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pdb
import sys
from tqdm import trange

import numpy as np
import SimpleITK as sitk

from dataHandlers import dataHandler

from networks import (
    unetEncoder as encoder,
    unetDecoder as decoder
    )

from utils import myBCELoss, toCategorical, dice_coeff, integralDice, compare_models, contrastiveLoss, getClassWts

# from networks import dense_unet_encoder as encoder #SimpleNetEncoder as encoder
# from networks import dense_unet_decoder as decoder
# from gradCam import GradCam
# from integratedGradientPytorch.integrated_gradients import integrated_gradients
# from integratedGradientPytorch.utils import calculate_outputs_and_gradients

def globalAcc(predList,labelList):
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

def predictAndGetLoss(model,X,y,batchSize,classWts,taskType,isVal):
    '''
    Generate predictions and calculate loss as well as skeleton info for metric,
    as per the given task - binary classification, siamese n/w classification or segmentation
    '''
    predListBatch = [] ; labelListBatch = []
    if taskType=='classifyDirect' or taskType=='classifySiamese':
        dims = 1
    elif taskType=='segment':
        dims = 3
    yOH = toCategorical(batchSize,y.cpu(),3,dims).cuda(gpuID)
    if taskType=='classifyDirect':
        out,_,_,_ = model.forward(X)
        loss = 0
        for i in range(2):
            loss += myBCELoss(1).forward(out[:,i],yOH[:,i])            
        pred = torch.argmax(out,1)
        predListBatch.append(pred.reshape(pred.shape[0]).cpu())
        labelListBatch.append(y.reshape(y.shape[0]).cpu())
        dataForMetric = (predList,labelList)
    elif taskType=='classifySiamese':
        out1,_,_,_ = model.forward(X[0].unsqueeze(0))
        out2,_,_,_ = model.forward(X[1].unsqueeze(0))
        if y[0]==y[1]:
            pairLabel = 1
        else:
            pairLabel = 0 
        loss = contrastiveLoss(out1,out2,pairLabel,1)
        dataForMetric = None
    elif taskType=='segment':
        if not isVal:
            out = model.forward(X)
        else:
            out = model.forward(X).detach()
            yOH = yOH.detach()
        pred = torch.argmax(out,1).detach()
        loss = 0
        diceCoeff = 0
        diceClasses = 0
        for i in range(3):
            # pdb.set_trace()
            lossBCE = myBCELoss(classWts[i]).forward(out[:,i],yOH[:,i])  
            # lossDice = ( 1-dice_coeff(out[:,i,:,:,:].float(),yOH[:,i,:,:,:].float()) )
            loss += lossBCE #+ lossDice
        dataForMetric = pred#diceCoeff
    return loss, dataForMetric

def getDiceMetrics(pred,label):
    predKidney = pred.clone()
    predKidney[predKidney==2] = 1
    labelKidney = label.clone()
    labelKidney[labelKidney==2] = 1
    diceCoeffKidney = integralDice(predKidney.float(),labelKidney[0].float(),1)
    diceCoeffTumor = integralDice(pred.float(),label[0].float(),2)
    return diceCoeffKidney,diceCoeffTumor

def train(model,genObj,optimizer,epoch,batchSize,nBatches,classWts,taskType):
    runningLoss = 0.0
    runningDiceKidney = 0.0
    runningDiceTumor = 0.0
    predList = []
    labelList = []
    model.train()
    with trange(nBatches,desc='Epoch '+str(epoch+1),ncols=100) as t:
        for m in range(nBatches):
            X,y,_,_ = genObj.__next__()
            optimizer.zero_grad()
            if taskType=='classifyDirect':
                predList.extend(dataForMetric[0])
                labelList.extend(dataForMetric[1])
            elif taskType=='segment':
                loss, pred = predictAndGetLoss(model,X,y,batchSize,classWts,taskType,False)
                fullPred = torch.cat([pred[1],pred[0]],-1)
                torch.cuda.empty_cache()
                fullLabel = torch.cat([y[1].cpu(),y[0].cpu()],-1)
                # pdb.set_trace()
                diceCoeffKidney,diceCoeffTumor = getDiceMetrics(fullPred.cpu(),fullLabel.detach().cpu())
                runningDiceKidney += diceCoeffKidney
                runningDiceTumor += diceCoeffTumor
                # loss = (loss1 + loss2)
            runningLoss += loss 
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=runningLoss.item()/(float(m+1)*batchSize))
            t.update()
    if taskType=='classifyDirect':
        acc = globalAcc(predList,labelList)
        print('Epoch num. %d \t Trn. Loss : %.7f ; \t Trn. Acc : %.3f' %(epoch+1,runningLoss/( (m+1)*batchSize), acc.item() ))
    elif taskType=='classifySiamese':
        print('Epoch num. %d \t Trn. Loss : %.7f ; \t ' %(epoch+1,runningLoss/( (m+1)*batchSize) ))
    elif taskType=='segment':
        diceKidney = runningDiceKidney / (m+1)
        diceTumor = runningDiceTumor / (m+1)
        print('Epoch num. %d \t Trn. Loss : %.7f ; \t Kidney Trn. Dice : %.3f ; \t Tumor Trn. Dice : %.3f' 
            %(epoch+1,runningLoss/( (m+1)*batchSize), diceKidney, diceTumor ))        

def validate(model,genObj,epoch,scheduler,batchSize,nBatches,classWts,taskType,dh):
    runningLoss = 0.0
    runningDiceKidney = 0.0
    runningDiceTumor = 0.0
    predList = []
    labelList = []
    model.eval()
    for m in range(nBatches):
        X,y,case,direction = genObj.__next__()
        # if case=='case_00022' and direction=='right':
        # pdb.set_trace()
        # predictAndSave(X,case,model,dh)
        # loss, dataForMetric = predictAndGetLoss(model,X,y,batchSize,classWts,taskType,True)
        if taskType=='classifyDirect':
            predList.extend(dataForMetric[0])
            labelList.extend(dataForMetric[1])
        elif taskType=='segment':
                loss, pred = predictAndGetLoss(model,X,y,batchSize,classWts,taskType,True)
                fullPred = torch.cat([pred[1],pred[0]],-1)
                # del pred1 ; del pred2 ; torch.cuda.empty_cache()
                fullLabel = torch.cat([y[1].cpu(),y[0].cpu()],-1)
                diceCoeffKidney,diceCoeffTumor = getDiceMetrics(fullPred.cpu(),fullLabel.detach().cpu())
                runningDiceKidney += diceCoeffKidney
                runningDiceTumor += diceCoeffTumor
        runningLoss += loss.item()
        # print('Case: '+case+' '+direction+' side ')#+str(dataForMetric.item()))
    valLoss = runningLoss/( (m+1)*batchSize)
    scheduler.step(valLoss)
    if taskType=='classifyDirect':
        acc = globalAcc(predList,labelList)
        print('Epoch num. %d \t Val. Loss : %.7f ; \t Val. Acc : %.3f' %(epoch+1, valLoss, acc.item() ))
    elif taskType=='classifySiamese':
        print('Epoch num. %d \t Val. Loss : %.7f ; \t ' %(epoch+1,valLoss ))
    elif taskType=='segment':
        diceKidney = runningDiceKidney / (m+1)
        diceTumor = runningDiceTumor / (m+1)
        print('Epoch num. %d \t Trn. Loss : %.7f ; \t Kidney Trn. Dice : %.3f ; \t Tumor Trn. Dice : %.3f' 
            %(epoch+1,runningLoss/( (m+1)*batchSize), diceKidney, diceTumor ))     

def predictAndSave(X,case,model,dh):
    out = model.cuda(0).forward(X[0].cuda(0).unsqueeze(0))
    pred1 = torch.argmax(out.cpu(),1)
    out = model.cuda(1).forward(X[1].cuda(1).unsqueeze(0))
    pred2 = torch.argmax(out.cpu(),1)
    fullPred = torch.cat([pred2,pred1],-1)
    fullPredResized = dh.cropResize(fullPred[0],())
    del pred1
    del pred2
    del out
    del X
    saveVolume(fullPredResized,'valPreds_scratch/prediction_'+case.split('_')[1]+'.nii.gz') #
    torch.cuda.empty_cache()

def test(model,genObj,dh,nBatches):
    model.eval()
    for m in range(nBatches):
        # pdb.set_trace()
        X, _ , case, _ = genObj.__next__()
        predictAndSave(X,case,model,dh)
 
class DUN(nn.Module):
    def __init__(self,encoder,decoder):
        super(DUN,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x):
        c3,c2,c1 = self.encoder(x)
        out = self.decoder(c3,c2,c1)
        # x,c1_out,c2_out,c3_out = self.encoder(x)
        # out = self.decoder(x,c1_out,c2_out,c3_out)
        return out

def main():
    batchSize = 2
    nSamples = 210
    valSplit = 20
    nTrnBatches = (nSamples - valSplit)//batchSize
    nValBatches = valSplit*2 

    testBatchSize = 2
    nTestSamples = 90
    nTestBatches = nSamples // batchSize

    nEpochs = 10
    lr = 5e-3
    weightDecay = 1e-2
    initEpochNum = int(sys.argv[1])

    problemType = 'main'
    taskType = 'segment'

    # path = '/scratch/abhinavdhere/kits_train/'
    path = '/scratch/abhinavdhere/kits_resampled/Train/'#'/home/abhinav/kits_resampled/Train/'
    # testPath = '/home/abhinav/kits_resampled/Test/'
    loadName = 'kidneyOnlySiamese.pt'
    saveName =  'segKidney_multiClass_unet_bceOnly.pt'  # 'selfSiamese.pt' # 'models/segKidney.pt' 
    # model = torch.load(saveName).cuda(gpuID)
    # saveName = 'segKidneySelf.pt'
    # proxyModel = torch.load(loadName)
    # pretrained_dict = proxyModel.state_dict()
    encoderModel = encoder().cuda()
    # encoderModel.load_state_dict(pretrained_dict,strict=False)
    # compare_models(proxyModel, encoderModel)
    # del proxyModel
    # del pretrained_dict   
    # torch.cuda.empty_cache()
    decoderModel = decoder(3).cuda()
    model = DUN(encoderModel,decoderModel)
    # model = encoder().cuda(gpuID)
    model = nn.DataParallel(model)
    # pdb.set_trace()
    dh = dataHandler(path,batchSize,valSplit,16,gpuID)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weightDecay)
    scheduler = ReduceLROnPlateau(optimizer,factor=0.5,patience=3,verbose=True)

    trainDataLoader = dh.giveGenerator('train',problemType)
    valDataLoader = dh.giveGenerator('val',problemType)

    # wts = getClassWts(nTrnBatches,trainDataLoader)
    wts = [1,1500,3850] # [1,15,38.5]

    # testDh = dataHandler(testPath,testBatchSize,valSplit=0,dataShapeMultiple=16,gpuID=gpuID)
    # testDataLoader = testDh.giveGenerator('test',problemType)

    for epoch in range(initEpochNum, initEpochNum+nEpochs):
        # train(model,trainDataLoader,optimizer,epoch,batchSize,nTrnBatches,wts,taskType)
        # torch.save(model,saveName)
        # if (epoch)%2==0:
            validate(model,valDataLoader,epoch,scheduler,batchSize,nValBatches,wts,taskType,dh)
    # test(model,valDataLoader,dh,nValBatches)
    # test(model,testDataLoader,testDh,nTestBatches)
    ## GradCam  
    # vol,label = valDataLoader.__next__()
    # 
    # gradCamObj = GradCam(model,target_layer_names=['conv8'] ,use_cuda=True)
    # mask = gradCamObj(vol,label.item()) 

    ## IG
    # X, labels = valDataLoader.__next__()
    # baseline = np.random.uniform(low=-71,high=304,size=X.shape)
    # integrated_grad = integrated_gradients(X.detach().cpu().numpy(), model, labels.detach().cpu().numpy(), \
    #     calculate_outputs_and_gradients, baseline=baseline, steps=100, cuda=True)

if __name__ == '__main__':
    gpuID = 0
    main()

## ----------------------
            # if i==1 and len(torch.unique(y))>1:
            #     predKidney = pred.clone()
            #     predKidney[predKidney==2] = 1
            #     labelKidney = y.clone()
            #     labelKidney[labelKidney==2] = 1
            #     diceCoeff += integralDice(predKidney.float().detach().cpu(),labelKidney[:,0,:,:,:].float().detach(),i)
            #     diceClasses+=1
            # elif i==2 and len(torch.unique(y))>2:
            #     diceCoeff += integralDice(pred.float().detach().cpu(),y[:,0,:,:,:].float().detach(),i)
            #     diceClasses+=1
        # if diceClasses>0:
        #     dataForMetric = diceCoeff/diceClasses
        # else:

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
