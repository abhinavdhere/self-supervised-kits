import os
import pdb

import numpy as np
import SimpleITK as sitk

import torch
from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import rotate
import scipy.ndimage as spIm

# from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop

# from keras.utils import to_categorical

class dataHandler(object):
    '''
    Methods for loading and handling KiTS data
    '''
    def __init__(self,trnPath,valPath,batchSize,valSplit,dataShapeMultiple,gpuID,window=(-79,304)):
        self.trnPath = trnPath
        self.valPath = valPath
        self.batchSize = batchSize  
        if batchSize>2:
            print('Warning: batchSize greater than 2 may cause problems as the sizes of volumes are varying')
        self.dataShapeMultiple = dataShapeMultiple
        self.window = window
        if valSplit>0:
            self.trainFileList = os.listdir(trnPath)
            self.valFileList = os.listdir(valPath)
            #self.trainFileList,self.valFileList = train_test_split(self.fileList,test_size=valSplit,random_state=0)
        elif valSplit==0:
            self.fileList = os.listdir(trnPath)
            self.trainFileList = self.fileList
        self.gpuID = gpuID

    def getBB(self,vol):
        '''
        https://stackoverflow.com/a/31402351
        '''
        axialMin, axialMax = np.where(np.any(vol, axis=(1,2)))[0][[0,-1]]
        coronalMin, coronalMax = np.where(np.any(vol, axis=(0,2)))[0][[0,-1]]
        sagMin, sagMax = np.where(np.any(vol, axis=(0,1)))[0][[0,-1]]
        return axialMin,axialMax,coronalMin,coronalMax,sagMin,sagMax

    def getPerturbation(self,vol):
        '''
        Get random augmentations for given volume. 
        '''
        augFlag = np.random.choice([0,1])
        if augFlag==0:
            vol = vol        
        elif augFlag==1:
            augIdx = np.random.choice(4)
            if augIdx==0:
                transVal = np.random.choice([-10,10])
                tfmMat = np.eye(4,4)
                tfmMat[:3,-1] = transVal
                vol = spIm.affine_transform(vol,tfmMat,vol.shape)
            elif augIdx==1:
                vol = np.flip(vol,1).copy()
            elif augIdx==2:
                vol = np.flip(vol,0).copy()
            elif augIdx==3:
                rotAng = np.random.choice([-20,-10,10,20])
                vol = spIm.rotate(vol,rotAng,reshape=False)
        return vol

    def resizeToNearestMultiple(self,vol,base):
        diffList = []
        for i in range(3):
            nSlices = vol.shape[i]
            reqSlices = base[i] * np.ceil(nSlices/base[i])
            diff = int(reqSlices - nSlices)
            if diff%2==0:
                sizeVec = (diff//2,diff//2)
            elif diff%2!=0:
                sizeVec = ((diff-1)//2,(diff+1)//2)
            diffList.append(sizeVec)
        vol = np.pad( vol,( diffList[0] , diffList[1], diffList[2] ), mode='constant' )
        return vol

    def resizeToSize(self,vol,sizeGiven):
        sizeDiff = [sizeGiven[i] - vol.shape[i] for i in range(len(vol.shape))]
        sizeVecList = []
        for j in range(len(vol.shape)):
            if sizeDiff[j]%2==0:
                sizeVec = (sizeDiff[j]//2,sizeDiff[j]//2)
            elif sizeDiff[j]%2!=0:
                sizeVec = ( (sizeDiff[j]-1)//2,(sizeDiff[j]+1)//2)
            sizeVecList.append(sizeVec)
        vol = np.pad( vol, ( sizeVecList[0], sizeVecList[1], sizeVecList[2]), mode='constant')
        return vol


    def cropResize(self,vol,targetSize):
        if not targetSize:
            targetSize = self.origSize
        size1 = vol.shape
        sizeDiff = [size1[i]-targetSize[i] for i in range(3)]
        initVals = []
        endVals = []
        for j in range(3):
            diff = sizeDiff[j]
            if sizeDiff[j]%2==0 and sizeDiff[j]>0:
                initVals.append(diff//2)
                endVals.append(size1[j]-diff//2)
            elif sizeDiff[j]%2!=0 and sizeDiff[j]>0:
                initVals.append( (diff-1)//2 )
                endVals.append( size1[j] - (diff+1)//2 )
            elif sizeDiff[j]<=0:
                initVals.append(0)
                endVals.append(size1[j])
        resizedVol = vol[initVals[0]:endVals[0],initVals[1]:endVals[1],initVals[2]:endVals[2]]
        return resizedVol

    def clipToMaxSize(self,vol,targetInitSize,targetEndSize):
        diff = [vol.shape[i]-targetEndSize[i] for i in range(3)]
        finalEndVals = []
        for j,diffVal in enumerate(diff):
            if diffVal>0:
                endVal = targetEndSize[j]
            else:
                endVal = vol.shape[j]
            finalEndVals.append(endVal)
        vol = vol[targetInitSize[0]:finalEndVals[0],targetInitSize[1]:finalEndVals[1],targetInitSize[2]:finalEndVals[2]]
        return vol

    def loadVolume(self,caseName,sectionSide,volType):
        if volType=='data':
            vol = sitk.ReadImage(caseName+'/resampled_vol.nii.gz') #'/imaging.nii.gz')
            self.origSize = vol.GetSize()
            vol = sitk.GetArrayFromImage(vol).swapaxes(0,2)
            vol[vol<self.window[0]] = self.window[0]
            vol[vol>self.window[1]] = self.window[1]
            vol = (vol - np.mean(vol))/np.std(vol)
        elif volType=='label':
            vol = sitk.ReadImage(caseName+'/resampled_labels.nii.gz')
            vol = sitk.GetArrayFromImage(vol).swapaxes(0,2)
            # vol[vol==1] = 0
            vol[vol==2] = 1 # tumor to kidney
        vol = self.resizeToNearestMultiple(vol,2)#(16,16,32))#self.dataShapeMultiple)
        if sectionSide=='left':
            vol = vol[:,:,(vol.shape[2]//2):]
        elif sectionSide=='right':
            vol = vol[:,:,:(vol.shape[2]//2)]
            # vol = np.fliplr(vol).copy()
        return vol

    def loadProxyData(self,fileList):
        gpuID = 0
        targetSize = (64,112,112)
        fListLeft = [fName+'-left' for fName in fileList]
        fListRight = [fName+'-right' for fName in fileList]
        fileList = fListLeft + fListRight
        while True:
            fileList = np.random.permutation(fileList)
            # directions = ['left','right']
            volArr = []
            labelArr = []
            count = 0
            for caseName in fileList:
                case, direction = caseName.split('-')
                # directions = np.random.permutation(directions)
                # for direction in directions:
                fullVol = self.loadVolume(case,direction,'data')
                segLabel = self.loadVolume(case,direction,'label')
                axialMin,axialMax,coronalMin,coronalMax,sagMin,sagMax = self.getBB(segLabel)
                volKidneyOnly = fullVol*segLabel
                vol = volKidneyOnly[axialMin:axialMax,coronalMin:coronalMax,sagMin:sagMax]
                if direction=='left':
                    label = 0
                elif direction=='right':
                    label = 1
                vol = self.resizeToSize(vol,targetSize)
                # vol = self.resizeToNearestMultiple(vol,self.dataShapeMultiple)
                vol = torch.Tensor(vol).cuda(self.gpuID)
                label = torch.Tensor([label]).long().cuda(self.gpuID)
                volArr.append(vol)
                labelArr.append(label)
                count+=1
                if count==self.batchSize:
                    yield torch.stack(volArr).unsqueeze(1), torch.stack(labelArr)
                    # yield np.expand_dims(np.stack(volArr),-1), to_categorical(np.stack(labelArr))
                    count = 0 ; volArr = [] ; labelArr = []

    def loadSegData(self,fileList,isTrn,isTest=False):
        while True:
            fileList = np.random.permutation(fileList)
            directions = ['left','right']
            volArr = []
            labelArr = []
            directionList = []
            count = 0
            if isTrn or isTest:
                path = self.trnPath
            elif (not isTrn) and (not isTest):
                path = self.valPath 
            for case in fileList:
                if isTrn:
                    directions = np.random.permutation(directions)
                for direction in directions:
                    vol = self.loadVolume(path+case,direction,'data')
#                    vol = self.clipToMaxSize(vol,[0,48,0],[160,256,160])  
 #                   vol = self.clipToMaxSize(vol,[0,120,0],[176,504,320])   # safe size is 176,384,320 (0:176,120:504,0:320)
                    vol = self.resizeToNearestMultiple(vol,self.dataShapeMultiple)
                    if direction=='left':
                        self.leftSize = vol.shape
                    elif direction=='right':
                        self.rightSize = vol.shape
                    if not isTest:
                        segLabel = self.loadVolume(path+case,direction,'label')
                        if segLabel.dtype=='uint16':
                            segLabel = segLabel.astype('uint8')
 #                       segLabel = self.clipToMaxSize(segLabel,[0,120,0],[176,504,320])
#                        segLabel = self.clipToMaxSize(segLabel,[0,48,0],[160,256,160]) 
                        segLabel = self.resizeToNearestMultiple(segLabel,self.dataShapeMultiple)
                    # if isTrn:
                    #     vol = self.getPerturbation(vol)
                    #     segLabel = self.getPerturbation(segLabel)
                    vol = torch.Tensor(vol).cuda(self.gpuID)
                    volArr.append(vol)
                    if not isTest:
                        label = torch.Tensor(segLabel).long().cuda(self.gpuID)
                        labelArr.append(label)
                    count+=1
                    directionList.append(direction)
#                    print('Loaded '+direction+' side of '+case)
                    if count==self.batchSize:
                        if isTest:
                            yield torch.stack(volArr).unsqueeze(1), case, direction
                        else:
                            yield torch.stack(volArr).unsqueeze(1), torch.stack(labelArr).unsqueeze(1), case, direction
                        # yield np.expand_dims(np.stack(volArr),-1), to_categorical(np.stack(labelArr))
                        count = 0 ; volArr = [] ; labelArr = []

    def giveGenerator(self,genType,task):
        if task=='main' and genType=='train':
            fList = self.trainFileList
            return self.loadSegData(fList,True)
        elif task=='main' and genType=='val':
            fList = self.valFileList
            return self.loadSegData(fList,False)
        elif task=='main' and genType=='test':
            fList = self.trainFileList
            return self.loadSegData(fList,False,True)
        elif task=='proxy' and genType=='train':
            fList = self.trainFileList
            return self.loadProxyData(fList)
        elif task =='proxy' and genType=='val':
            fList = self.valFileList
            return self.loadProxyData(fList)




