import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pdb
import sys
from tqdm import trange

import numpy as np
import SimpleITK as sitk

from dataHandlers import dataHandler
from networks import dense_unet_encoder as encoder #SimpleNetEncoder as encoder
from networks import dense_unet_decoder as decoder
from utils import myBCELoss, toCategorical, dice_coeff, integralDice, compare_models, contrastiveLoss

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
    yOH = toCategorical(batchSize,y.cpu(),2,dims).cuda(gpuID)
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
        out = model.forward(X)
        pred = torch.argmax(out,1)
        loss = 0
        diceCoeff = 0
        for i in range(2):
            loss += ( 1-dice_coeff(out[:,i,:,:,:].float(),yOH[:,i,:,:,:].float()) )
            if i>0:
                diceCoeff += integralDice(pred.float().detach().cpu(),y[:,0,:,:,:].float().detach(),i)
        dataForMetric = diceCoeff
    return loss, dataForMetric

def train(model,genObj,optimizer,epoch,batchSize,nBatches,taskType):
    runningLoss = 0.0
    runningDice = 0.0
    predList = []
    labelList = []
    model.train()
    with trange(nBatches,desc='Epoch '+str(epoch+1),ncols=100) as t:
        for m in range(nBatches):
            X,y = genObj.__next__()
            optimizer.zero_grad()
            # loss, dataForMetric = predictAndGetLoss(model,X,y,batchSize,taskType)
            if taskType=='classifyDirect':
                predList.extend(dataForMetric[0])
                labelList.extend(dataForMetric[1])
            elif taskType=='segment':
                runningDice += dataForMetric
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
        print(runningDice)
        dice = runningDice / (m+1)
        print('Epoch num. %d \t Trn. Loss : %.7f ; \t Trn. Dice : %.3f' %(epoch+1,runningLoss/( (m+1)*batchSize), dice ))        

def validate(model,genObj,epoch,batchSize,nBatches,taskType,dh):
    runningLoss = 0.0
    runningDice = 0.0
    predList = []
    labelList = []
    model.eval()
    for m in range(nBatches):
        X,y,case,direction = genObj.__next__()
        # if case=='case_00022' and direction=='right':
        # pdb.set_trace()
        # predictAndSave(X,case,model,dh)
        loss, dataForMetric = predictAndGetLoss(model,X,y,batchSize,taskType)
        if taskType=='classifyDirect':
            predList.extend(dataForMetric[0])
            labelList.extend(dataForMetric[1])
        elif taskType=='segment':
            runningDice += dataForMetric
        runningLoss += loss.item()
        # print('Case: '+case+' '+direction+' side ')#+str(dataForMetric.item()))
    if taskType=='classifyDirect':
        acc = globalAcc(predList,labelList)
        print('Epoch num. %d \t Val. Loss : %.7f ; \t Val. Acc : %.3f' %(epoch+1,runningLoss/( (m+1)*batchSize), acc.item() ))
    elif taskType=='classifySiamese':
        print('Epoch num. %d \t Val. Loss : %.7f ; \t ' %(epoch+1,runningLoss/( (m+1)*batchSize) ))
    elif taskType=='segment':
        dice = runningDice / (m+1)
        print('Epoch num. %d \t Val. Loss : %.7f ; \t Val. Dice : %.3f' %(epoch+1,runningLoss/( (m+1)*batchSize), dice ))   

def predictAndSave(X,case,model,dh):
    out = model.cuda(2).forward(X[0].cuda(2).unsqueeze(0))
    pred1 = torch.argmax(out.cpu(),1)
    out = model.cuda(3).forward(X[1].cuda(3).unsqueeze(0))
    pred2 = torch.argmax(out.cpu(),1)
    fullPred = torch.cat([pred2,pred1],-1)
    fullPredResized = dh.cropResize(fullPred[0],())
    del pred1
    del pred2
    del out
    del X
    saveVolume(fullPredResized,'testPreds_scratch/prediction_'+case.split('_')[1]+'.nii.gz') #
    torch.cuda.empty_cache()

def test(model,genObj,dh,nBatches):
    model.eval()
    for m in range(nBatches):
        X, case = genObj.__next__()
        predictAndSave(X,case,model,dh)
 
class DUN(nn.Module):
    def __init__(self,encoder,decoder):
        super(DUN,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x):
        x,c1_out,c2_out,c3_out = self.encoder(x)
        out = self.decoder(x,c1_out,c2_out,c3_out)
        return out

def main():
    batchSize = 1
    nSamples = 210
    valSplit = 20
    nTrnBatches = (nSamples - valSplit)//batchSize
    nValBatches = valSplit*2 

    testBatchSize = 2
    nTestSamples = 90
    nTestBatches = nSamples // batchSize

    nEpochs = 1#20
    lr = 5e-4
    weightDecay = 1e-2
    initEpochNum = int(sys.argv[1])

    problemType = 'main'
    taskType = 'segment'

    # path = '/scratch/abhinavdhere/kits_train/'
    path = '/home/abhinav/kits_train/'
    testPath = '/home/abhinav/kits_test/'
    loadName = 'kidneyOnlySiamese.pt'
    saveName =   'models/segKidney.pt' # 'selfSiamese.pt' # 
    model = torch.load(saveName).cuda(gpuID)
    # saveName = 'segKidneySelf.pt'
    # proxyModel = torch.load(loadName)
    # pretrained_dict = proxyModel.state_dict()
    # encoderModel = encoder().cuda()
    # encoderModel.load_state_dict(pretrained_dict,strict=False)
    # compare_models(proxyModel, encoderModel)
    # del proxyModel
    # del pretrained_dict   
    # torch.cuda.empty_cache()
    # decoderModel = decoder().cuda()
    # model = DUN(encoderModel,decoderModel)
    # model = encoder().cuda(gpuID)
    # model = nn.DataParallel(model)
    dh = dataHandler(path,batchSize,valSplit,16,gpuID)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weightDecay)

    trainDataLoader = dh.giveGenerator('train',problemType)
    valDataLoader = dh.giveGenerator('val',problemType)

    testDh = dataHandler(testPath,testBatchSize,valSplit=0,dataShapeMultiple=16,gpuID=gpuID)
    testDataLoader = testDh.giveGenerator('test',problemType)

    # for epoch in range(initEpochNum, initEpochNum+nEpochs):
        # train(model,trainDataLoader,optimizer,epoch,batchSize,nTrnBatches,taskType)
        # validate(model,valDataLoader,epoch,batchSize,nValBatches,taskType,dh)
        # torch.save(model,saveName)
    test(model,testDataLoader,testDh,nTestBatches)
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

if __name__ == '__main__':
    gpuID = 3
    main()

## ----------------------
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
