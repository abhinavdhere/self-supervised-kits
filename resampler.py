import os
import time

import pdb

import numpy as np
import SimpleITK as sitk

def resample(vol,new_spacing):
    orig_spacing = [3.22,1.62,1.62]#vol.GetSpacing()
    orig_size = np.array(vol.GetSize(),dtype=np.int)

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(vol.GetDirection())
    resample.SetOutputOrigin(vol.GetOrigin())
    resample.SetOutputSpacing(new_spacing)

    newSize = orig_size*(np.array(orig_spacing)/np.array(new_spacing))
    newSize = np.floor(newSize).astype('int')
    resample.SetSize(tuple(newSize.tolist()))
    
    newVol = resample.Execute(vol)
    return newVol

def resampleVolume(path,fName,volType):
    initTime = time.time()
    new_spacing = [3.22,1.62,1.62]
    
    if volType=='Data':
        vol = sitk.ReadImage(path+fName+'/imaging.nii.gz')
    elif volType=='Labels':
        vol = sitk.ReadImage(path+fName+'/segmentation.nii.gz')

    newVol = resample(vol,new_spacing)

    print('Took '+str(time.time()-initTime)+' seconds to resize.')
    print('Resized Case '+fName+' from size '+str(vol.GetSize())+' to '+str(newVol.GetSize()))
    writer = sitk.ImageFileWriter()
    if volType=='Data':
        writer.SetFileName(path+fName+'/resampled_vol.nii.gz')
    elif volType=='Labels':
        writer.SetFileName(path+fName+'/resampled_labels.nii.gz')
    writer.Execute(newVol)

def inverseResample(path,fName,volType):
    pdb.set_trace()
    origVol = sitk.ReadImage(path+fName+'/imaging.nii.gz')
    resampledVol = sitk.ReadImage(path+fName+'/resampled_vol.nii.gz')
    predVol = sitk.ReadImage('/home/abhinav/selfSupervised_kidney/testPreds/prediction_'+fName+'.nii.gz')
    resampledSize = resampledVol.GetSize()
    predVol = sitk.GetArrayFromImage(predVol)
    currentSize = predVol.shape
    resizedVol = cropResize(predVol,currentSize,resampledSize)
    new_spacing = origVol.GetSpacing()
    invResampledVol = resample(sitk.GetImageFromArray(resizedVol.swapaxes(0,2)),new_spacing)
    if not np.equal(np.array(origVol.GetSize()),np.array(invResampledVol.GetSize())).all():
        invResampledVol = cropResize(invResampledVol,invResampledVol.GetSize(),origVol.GetSize())
    writer = sitk.ImageFileWriter()
    writer.SetFileName('/home/abhinav/selfSupervised_kidney/testPredsInv/prediction_'+fName+'.nii.gz')
    if np.equal(np.array(origVol.GetSize()),np.array(invResampledVol.GetSize())).all():
        writer.Execute(invResampledVol)
        print('Resized Case '+fName+' from size '+str(currentSize)+' to '+str(invResampledVol.GetSize()))

# fName = 'case_00023'
path = '/scratch/abhinavdhere/kits_test/'#'/home/abhinav/kits_test/'
fList = os.listdir(path)
for fName in fList:
    inverseResample(path,fName,'Data')
    # resampleVolume(path,fName,'Labels')
