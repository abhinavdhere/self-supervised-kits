import os
import pdb

import numpy as np
import SimpleITK as sitk

import torch
from sklearn.model_selection import train_test_split

# from keras.utils import to_categorical

class dataHandler(object):
	'''
	Methods for loading and handling KiTS data
	'''
	def __init__(self,path,batchSize,valSplit,dataShapeMultiple,window=(-79,304)):
		self.path = path
		self.batchSize = batchSize	
		if batchSize>2:
			print('Warning: batchSize greater than 2 may cause problems as the sizes of volumes are varying')
		self.fileList = os.listdir(path)
		self.dataShapeMultiple = dataShapeMultiple
		self.window = window
		self.trainFileList,self.valFileList = train_test_split(self.fileList,test_size=valSplit,random_state=0)

	def getBB(self,vol):
		'''
		https://stackoverflow.com/a/31402351
		'''
		axialMin, axialMax = np.where(np.any(vol, axis=(1,2)))[0][[0,-1]]
		coronalMin, coronalMax = np.where(np.any(vol, axis=(0,2)))[0][[0,-1]]
		sagMin, sagMax = np.where(np.any(vol, axis=(0,1)))[0][[0,-1]]
		return axialMin,axialMax,coronalMin,coronalMax,sagMin,sagMax

	def resizeToNearestMultiple(self,vol,base):
		diffList = []
		for i in range(3):
			nSlices = vol.shape[i]
			reqSlices = base * np.ceil(nSlices/base)
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

	def loadVolume(self,caseName,sectionSide,volType):
		if volType=='data':
			vol = sitk.ReadImage(self.path+caseName+'/resampled_vol.nii.gz')
			vol = sitk.GetArrayFromImage(vol).swapaxes(0,2)
			vol[vol<self.window[0]] = self.window[0]
			vol[vol>self.window[1]] = self.window[1]
			vol = (vol - np.mean(vol))/np.std(vol)
		elif volType=='label':
			vol = sitk.ReadImage(self.path+caseName+'/resampled_labels.nii.gz')
			vol = sitk.GetArrayFromImage(vol).swapaxes(0,2)
			vol[vol==2] = 1
		# vol = self.resizeToNearestMultiple(vol,self.dataShapeMultiple)
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
				vol = torch.Tensor(vol).cuda(gpuID)
				label = torch.Tensor([label]).long().cuda(gpuID)
				volArr.append(vol)
				labelArr.append(label)
				count+=1
				if count==self.batchSize:
					yield torch.stack(volArr).unsqueeze(1), torch.stack(labelArr)
					# yield np.expand_dims(np.stack(volArr),-1), to_categorical(np.stack(labelArr))
					count = 0 ; volArr = [] ; labelArr = []

	def loadSegData(self,fileList):
		while True:
			fileList = np.random.permutation(fileList)
			directions = ['left','right']
			volArr = []
			labelArr = []
			count = 0
			for case in fileList:
				directions = np.random.permutation(directions)
				for direction in directions:
					fullVol = self.loadVolume(case,direction,'data')
					segLabel = self.loadVolume(case,direction,'label')
					if segLabel.dtype=='uint16':
						segLabel = segLabel.astype('uint8')
					vol = self.resizeToNearestMultiple(fullVol,self.dataShapeMultiple)
					segLabel = self.resizeToNearestMultiple(segLabel,self.dataShapeMultiple)
					vol = torch.Tensor(vol).cuda()
					label = torch.Tensor(segLabel).long().cuda()
					volArr.append(vol)
					labelArr.append(label)
					count+=1
					if count==self.batchSize:
						yield torch.stack(volArr).unsqueeze(1), torch.stack(labelArr).unsqueeze(1)
						# yield np.expand_dims(np.stack(volArr),-1), to_categorical(np.stack(labelArr))
						count = 0 ; volArr = [] ; labelArr = []

	def giveGenerator(self,genType,task):
		if task=='main' and genType=='train':
			fList = self.trainFileList
			return self.loadSegData(fList)
		elif task=='main' and genType=='val':
			fList = self.valFileList
			return self.loadSegData(fList)			
		elif task=='proxy' and genType=='train':
			fList = self.trainFileList
			return self.loadProxyData(fList)
		elif task =='proxy' and genType=='val':
			fList = self.valFileList
			return self.loadProxyData(fList)




