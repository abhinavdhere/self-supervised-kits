import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os, signal   
import pdb
import sys
from tqdm import trange

import numpy as np
import SimpleITK as sitk
import skimage.segmentation as segTools    
from dataHandlers import dataHandler
from networks import dense_unet_encoder as encoder #SimpleNetEncoder as encoder
from networks import dense_unet_decoder as decoder, dense_unet_autoencoder as autoencoder
from focalLoss import FocalLoss 
from utils import ( 
myBCELoss, dice_coeff, contrastiveLoss, tversky_loss,
toCategorical, compare_models, saveVolume, 
integralDice, getHausdorff, globalAcc,
getClassWts
)


def predictAndGetLoss(model,X,y,batchSize,taskType):
    '''
    Generate predictions and calculate loss as well as skeleton info for metric,
    as per the given task - binary classification, siamese n/w classification or segmentation
    '''
    predListBatch = [] ; labelListBatch = []
    if taskType=='classifyDirect' or taskType=='classifySiamese':
        dims = 1
    elif taskType=='segment' or taskType=='segmentKidney':
        dims = 3
    yOH = toCategorical(batchSize,y.cpu(),2,dims).cuda()
    if taskType=='classifyDirect':
        out, recons = model.forward(X)
        lossBCE = 0
        lossMSE = F.mse_loss(recons,X)
        for i in range(2):
            lossBCE += myBCELoss(1).forward(out[:,i],yOH[:,i])
        loss = 0.8*lossBCE + 0.2*lossMSE            
        pred = torch.argmax(out,1)
        predListBatch.append(pred.reshape(pred.shape[0]).cpu())
        labelListBatch.append(y.reshape(y.shape[0]).cpu())
        dataForMetric = (predListBatch,labelListBatch)
        return loss, dataForMetric
    elif taskType=='classifySiamese':
        out1,_,_,_ = model.forward(X[0].unsqueeze(0))
        out2,_,_,_ = model.forward(X[1].unsqueeze(0))
        d = torch.norm(out1-out2)
        # pdb.set_trace()
        if d>=0.75:
            pred = 0
        elif d<0.75:
            pred = 1
        if y[0]==y[1]:
            pairLabel = 1
        else:
            pairLabel = 0 
        loss = contrastiveLoss(out1,out2,pairLabel,1)
        predListBatch.append(torch.Tensor([pred]))
        labelListBatch.append(torch.Tensor([pairLabel]))
        dataForMetric = (predListBatch,labelListBatch)
        return loss, dataForMetric
    elif taskType=='segment' or taskType=='segmentKidney':
        out,recons = model.forward(X)
        pred = torch.argmax(out,1)
        lossSeg = 0
        lossBCE = 0
        lossDice = 0 
        diceCoeff = 0
        # pdb.set_trace()
        for i in range(2):
            lossBCE += myBCELoss(1).forward(out[:,i],yOH[:,i])
            # lossFocal += FocalLoss(alpha=0.95,gamma=2,reduction='mean')(out[:,i,:,:,:],yOH[:,i,:,:,:])
            # lossTversky += tversky_loss(0.5,yOH[:,i,:,:,:],out[:,i,:,:,:])
            lossDice = ( 1-dice_coeff(out[:,i,:,:,:].float(),yOH[:,i,:,:,:].float()) )
            #(0.9*lossDice+0.1*lossMSE)
            if i>0:
                diceCoeff += integralDice(pred.float().detach().cpu(),y[:,0,:,:,:].float().detach(),i)
        lossSeg = 0.4*lossBCE + 0.6*lossDice
        lossMSE = F.mse_loss(recons,X)
        loss = 0.9*lossSeg + 0.1*lossMSE
        return loss,lossBCE.item(),lossDice.item(),lossMSE.item(),diceCoeff

def train(model,genObj,optimizer,scheduler,epoch,batchSize,nBatches,taskType):
    runningLoss = 0.0
    runningLossBCE = 0.0
    runninglossDice = 0.0
    runningLossMSE = 0.0
    runningDice = 0.0
    predList = []
    labelList = []
    model.train()
    with trange(nBatches,desc='Epoch '+str(epoch+1),ncols=100) as t:
        for m in range(nBatches):
            # pdb.set_trace()
            X,y,_,_ = genObj.__next__()
            optimizer.zero_grad()
            if taskType=='classifyDirect':
                loss, dataForMetric = predictAndGetLoss(model,X,y,batchSize,taskType)
                predList.extend(dataForMetric[0])
                labelList.extend(dataForMetric[1])
            elif taskType=='classifySiamese':
                loss = predictAndGetLoss(model,X,y,batchSize,taskType)
            elif taskType=='segment' or taskType=='segmentKidney':
                loss, lossBCE, lossDice, lossMSE, dataForMetric = predictAndGetLoss(model,X,y,batchSize,taskType)
                runningDice += dataForMetric
                runningLossBCE += lossBCE
                runninglossDice += lossDice
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
    elif taskType=='segment' or taskType=='segmentKidney':
        # print(runningDice)
        batchLoss = runningLoss/( (m+1)*batchSize)
        batchLossBCE = runningLossBCE / ( (m+1)*batchSize)
        batchlossDice = runninglossDice / ( (m+1)*batchSize)
        batchLossMSE = runningLossMSE / ( (m+1)*batchSize)
        #scheduler.step(batchLoss)
        dice = runningDice / (m+1)
        print('Epoch num. %d  Trn. Loss : %.7f ;  Trn. BCE : %.7f ;  Trn. DL : %.7f ;  Trn. MSELoss : %.7f ;  Trn. Dice : %.3f' 
            %(epoch+1, batchLoss, batchLossBCE, batchlossDice, batchLossMSE, dice ))        
        # print('Alpha for FL is now '+str(model.alpha.item())+ '. Beta for TL is now '+str(model.beta.item()))

def validate(model,genObj,epoch,batchSize,nBatches,taskType,dh):
    runningLoss = 0.0
    runningLossBCE = 0.0
    runninglossDice = 0.0
    runningDice = 0.0
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
            loss, dataForMetric = predictAndGetLoss(model,X,y,batchSize,taskType)
            predList.extend(dataForMetric[0])
            labelList.extend(dataForMetric[1])
        elif taskType=='segment' or taskType=='segmentKidney':
            loss, lossBCE, lossDice, lossMSE, dataForMetric = predictAndGetLoss(model,X,y,batchSize,taskType)
            runningDice += dataForMetric
            runningLossBCE += lossBCE
            runninglossDice += lossDice
            runningLossMSE += lossMSE
        runningLoss += loss.item()
    if taskType=='classifyDirect':
        acc = globalAcc(predList,labelList)
        print('Epoch num. %d \t Val. Loss : %.7f ; \t Val. Acc : %.3f' %(epoch+1,runningLoss/( (m+1)*batchSize), acc.item() ))
        return acc
    elif taskType=='classifySiamese':
        acc = globalAcc(predList,labelList)
        print('Epoch num. %d \t Val. Loss : %.7f ; \t Val. Acc : %.3f' %(epoch+1,runningLoss/( (m+1)*batchSize), acc.item() ))
    elif taskType=='segment' or taskType=='segmentKidney':
        dice = runningDice / (m+1)
        batchValLoss = runningLoss/( (m+1)*batchSize)
        batchLossBCE = runningLossBCE / ( (m+1)*batchSize)
        batchlossDice = runninglossDice / ( (m+1)*batchSize)
        batchLossMSE = runningLossMSE / ( (m+1)*batchSize)
        print('Epoch num. %d Val. Loss : %.7f ;  Val. BCE : %.7f ; Val. DL : %.7f ;  Val. MSELoss : %.7f ;  Val. Dice : %.3f' 
            %(epoch+1, batchValLoss , batchLossBCE, batchlossDice, batchLossMSE, dice ))   
        # print('Alpha for FL is now '+str(model.alpha.item())+ '. Beta for TL is now '+str(model.beta.item()))
        return dice

# def sig_handler(signum, frame):
#     print ("segfault")
#     print(case)
#     pdb.set_trace()

def getHDMetric(model,genObj,nBatches,dh):
    runningBL = 0.0
    model.eval()
    # signal.signal(signal.SIGSEGV, sig_handler)
    for m in range(nBatches):
        X,y,case,direction = genObj.__next__()
        fullPred = generateFullVolume(X,model,dh,'pred')
        fullLabel = generateFullVolume(y,model,dh,'label')
        # out, recons = model.forward(X)
        boundary1 = segTools.find_boundaries(fullPred.detach().cpu().numpy(),mode='inner')
        boundary2 = segTools.find_boundaries(fullLabel.detach().cpu().numpy(),mode='inner')
        bl1 = np.sum(boundary1)
        bl2 = np.sum(boundary2)
        bl = abs(bl1-bl2)/bl2
        # hd = getHausdorff(fullPred,fullLabel)
        print('Diff in BL for '+case+' is: '+str(bl))   
        runningBL += bl
    return runningBL/nBatches

def generateFullVolume(X,model,dh,dataType):
    if dataType=='pred':
        out,_ = model.cuda(0).forward(X[0].cuda(0).unsqueeze(0))
        pred1 = torch.argmax(out.cpu(),1)
        leftPred = dh.cropResize(pred1[0],dh.leftSize) 
        out,_ = model.cuda(0).forward(X[1].cuda(0).unsqueeze(0))
        pred2 = torch.argmax(out.cpu(),1)
        rightPred = dh.cropResize(pred2[0],dh.rightSize)
        fullPred = torch.cat([rightPred,leftPred],-1)
        fullPredResized = dh.cropResize(fullPred,())
        return fullPredResized
    elif dataType=='label':
        leftLabel = dh.cropResize(X[0],dh.leftSize)
        rightLabel = dh.cropResize(X[1],dh.rightSize)
        fullLabel = torch.cat([rightLabel,leftLabel],-1)
        fullLabelResized = dh.cropResize(fullLabel,())
        return fullLabelResized

def predictAndSave(X,case,model,dh):
    fullPredResized = generateFullVolume(X,model,dh,'pred')    
    saveVolume(fullPredResized,'predictions/testPreds_self_genLoss/prediction_'+case+'.nii.gz') 
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
        self.alpha = torch.nn.Parameter(torch.Tensor([0.5])).cuda()
        self.beta = torch.nn.Parameter(torch.Tensor([0.5])).cuda()
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
    def __init__(self,encoder,autoEncoder):
        super(DenseClassifier,self).__init__()
        self.encoder = encoder
        self.preClassifier = nn.DataParallel(nn.Conv3d(144,1,3,padding=1).cuda())
        self.classifier = nn.DataParallel(nn.Linear(196,2).cuda())
        self.autoEncoder = autoEncoder

    def forward(self,x):
        x,_,_,_ = self.encoder(x)
        feat = self.preClassifier(x)
        feat = feat.view(-1,196)
        pred = F.softmax(self.classifier(feat),1)
        recons = self.autoEncoder(x)
        return pred, recons

def main():
    '''
    Control section, contains all class/function calls. Use of separate main function allows encapsulation of variables.
    '''
    ## Control parameters.  
    ## Valid params : problemType = {'main','proxy'} ; taskType = {'classifyDirect','classifySiamese','segmentKidney','segmentTumor'}
    problemType = 'main'#'proxy' 
    taskType = 'segmentKidney'#'segmentKidney' #'classifyDirect'

    ## Constants and training parameters
    batchSize = 4
    nSamples = 207
    valSize = 30
    nTrnBatches = ((nSamples - valSize)*12)//batchSize                              # *2 since volumes are loaded in halves
    nValBatches = (valSize*2)//batchSize

    testBatchSize = 2
    nTestSamples = 90
    nTestBatches = nTestSamples*2 // testBatchSize

    nEpochs = 10
    lr = 6.25e-5
    weightDecay = 1e-4
    multipleOf = 16                                                                 # volumes should have size multiple of this number
    initEpochNum = int(sys.argv[1])                                                 # Starting epoch number, for display in progress

    ## Paths
    trnPath = '/home/abhinav/kits_train/'
    valPath = '/home/abhinav/kits_val/'
    currentBestRecord = 'bestVal_kidney_revisited_wMoreAug.txt'                      # Stores dice for best performance so far
    testPath = '/home/abhinav/kits_test/'
#    saveName = 'proxy_tumor_classify.pt' #   
    loadName = 'proxy_kidney_siamese.pt' #'kidneyOnlySiamese.pt'
    #saveName = 'scratch_tumor_dense_wAE_revisited.pt' #'self_kidney_wAE_genLoss.pt' 
    # saveName = 'self_tumor_dense_wAE.pt'  
    # saveName = 'scratch_tumor_dense_wAE.pt'
    saveName = 'self_dense_wAE_wMoreAug.pt'
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
            # pdb.set_trace()
            model = DUN(encoderModel)
        elif problemType=='proxy':
            if taskType=='classifyDirect':
                autoEncoderModel = nn.DataParallel(autoencoder().cuda())
                model = DenseClassifier(encoderModel,autoEncoderModel)
            elif taskType=='classifySiamese':
                model = encoderModel
    # model = torch.load(loadName)
    dh = dataHandler(trnPath,valPath,batchSize,valSize,multipleOf,gpuID)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weightDecay)
    scheduler = ReduceLROnPlateau(optimizer,factor=0.5,patience=3,verbose=True)

    trainDataLoader = dh.giveGenerator('train',problemType,taskType)
    valDataLoader = dh.giveGenerator('val',problemType,taskType)
    # for m in range(nTrnBatches):
    #     # pdb.set_trace()
    #     X,y,case,direction = trainDataLoader.__next__()
    #     if len(torch.unique(y))>2:
    #         print('Check case '+case+direction+', the label has '+str(torch.unique(y)))
    #     if m%10==0:
    #         print(str(m)+'volumes processed.')
    # avgHD = getHDMetric(model,trainDataLoader,nTrnBatches,dh)
    # print(avgHD)

    ## Learn and evaluate !!
    #wts = getClassWts(nTrnBatches,trainDataLoader)
    #print(wts)
    # validate(model,valDataLoader,0,batchSize,nValBatches,taskType,dh)
    if not os.path.exists(currentBestRecord):
        if problemType=='main':
            os.system('echo "Best Dice so far: 0.0" > '+currentBestRecord)
        elif problemType=='proxy':
            os.system('echo "Best Acc so far: 0.0" > '+currentBestRecord)
    statusFile = open(currentBestRecord,'r')
    bestValMetric = float(statusFile.readline().strip('\n').split()[-1])
    statusFile.close()
    for epoch in range(initEpochNum, initEpochNum+nEpochs):
        train(model,trainDataLoader,optimizer,scheduler,epoch,batchSize,nTrnBatches,taskType)
        torch.save(model,saveName)
        valMetric = validate(model,valDataLoader,epoch,batchSize,nValBatches,taskType,dh)
        if valMetric > bestValMetric:
            statusFile = open(currentBestRecord,'w')
            diff = valMetric - bestValMetric
            bestValMetric = valMetric
            torch.save(model,saveName.split('.')[0]+'_chkpt_'+str(epoch+1)+'epoch.pt')
            if problemType=='main':
                statusFile.write('Best Dice so far: '+str(bestValMetric.item()))
                print('Model checkpoint saved since Dice has improved by '+str(diff))
            else:
                statusFile.write('Best Acc so far: '+str(bestValMetric.item()))
                print('Model checkpoint saved since Acc has improved by '+str(diff))
            statusFile.close()
 
    # testDh = dataHandler(testPath,'',testBatchSize,valSplit=0,dataShapeMultiple=16,gpuID=gpuID)
    # testDataLoader = testDh.giveGenerator('test',problemType,taskType)
    # test(model,testDataLoader,testDh,nTestBatches) 
    # test(model,testDataLoader,0,batchSize,nTestBatches,testDh)

if __name__ == '__main__':
    gpuID = 0
    main()

## ---------------------- Graveyard -------------------------
     # del pred1
    # del pred2
    # del out
    # del X
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

