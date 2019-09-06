import os
import time

import numpy as np
import SimpleITK as sitk

def resampleVolume(path,fName,volType):
    initTime = time.time()
    new_spacing = [3.22,1.62,1.62]
    
    if volType=='Data':
        vol = sitk.ReadImage(path+fName+'/imaging.nii.gz')
    elif volType=='Labels':
        vol = sitk.ReadImage(path+fName+'/segmentation.nii.gz')
    orig_spacing = vol.GetSpacing()
    orig_size = np.array(vol.GetSize(),dtype=np.int)

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(vol.GetDirection())
    resample.SetOutputOrigin(vol.GetOrigin())
    resample.SetOutputSpacing(new_spacing)

    newSize = orig_size*(np.array(orig_spacing)/np.array(new_spacing))
    newSize = np.ceil(newSize).astype('int')
    resample.SetSize(tuple(newSize.tolist()))
    
    newVol = resample.Execute(vol)
    print('Took '+str(time.time()-initTime)+' seconds to resize.')
    print('Resized Case '+fName+' from size '+str(orig_size)+' to '+str(newVol.GetSize()))
    writer = sitk.ImageFileWriter()
    if volType=='Data':
        writer.SetFileName(path+fName+'/resampled_vol.nii.gz')
    elif volType=='Labels':
        writer.SetFileName(path+fName+'/resampled_labels.nii.gz')
    writer.Execute(newVol)

# fName = 'case_00023'
path = '/scratch/abhinavdhere/kits/train/'
fList = os.listdir(path)
for fName in fList:
    resampleVolume(path,fName,'Data')
    resampleVolume(path,fName,'Labels')
