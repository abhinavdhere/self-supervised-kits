import os
import time

import pdb

import numpy as np
import SimpleITK as sitk


from collections import OrderedDict
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
from multiprocessing.pool import Pool


'''
Following three functions taken from MIC-DKFZ repositories
https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/resample_augmentations.py
https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py
Both repositories are Apache licensed (available at given URLs above) and the code is used as is.
'''
def resize_segmentation(segmentation, new_shape, order=3, cval=0):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

def resample_patient(data, seg, original_spacing, target_spacing, order_data=3, order_seg=0, force_separate_z=False,
                     cval_data=0, cval_seg=-1, order_z_data=0, order_z_seg=0):
    """
    :param cval_seg:
    :param cval_data:
    :param data:
    :param seg:
    :param original_spacing:
    :param target_spacing:
    :param order_data:
    :param order_seg:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately
    :param order_z_seg: only applies if do_separate_z is True
    :param order_z_data: only applies if do_separate_z is True
    :return:
    """
    assert not ((data is None) and (seg is None))
    if data is not None:
        assert len(data.shape) == 4, "data must be c x y z"
    if seg is not None:
        assert len(seg.shape) == 4, "seg must be c x y z"

    if data is not None:
        shape = np.array(data[0].shape)
    else:
        shape = np.array(seg[0].shape)
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(original_spacing):
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing):
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

    if data is not None:
        data_reshaped = resample_data_or_seg(data, new_shape, False, axis, order_data, do_separate_z, cval=cval_data,
                                             order_z=order_z_data)
    else:
        data_reshaped = None
    if seg is not None:
        seg_reshaped = resample_data_or_seg(seg, new_shape, True, axis, order_seg, do_separate_z, cval=cval_seg,
                                            order_z=order_z_seg)
    else:
        seg_reshaped = None
    return data_reshaped, seg_reshaped

def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, cval=0, order_z=0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param cval:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    data = data.astype(float)
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        if do_separate_z:
            print("separate z")
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, cval=cval, **kwargs))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, cval=cval, **kwargs))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, cval=cval,
                                                       **kwargs))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z, cval=cval,
                                                                   mode='nearest')[None])
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                cval=cval, mode='nearest'))
                            reshaped[reshaped_multihot >= 0.5] = cl
                        reshaped_final_data.append(reshaped[None])
                else:
                    reshaped_final_data.append(reshaped_data[None])
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, cval=cval, **kwargs)[None])
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data

'''
My code starts here
'''
def inverseResample(path,fName):
    '''
    Resample predictions to original voxel spacing and thus, size.
    '''
    origVol = sitk.ReadImage(path+fName+'/imaging.nii.gz')
    predVol = sitk.ReadImage('/home/abhinav/selfSupervised_kidney/testPreds_scratch/prediction_'+fName.split('_')[1]+'.nii.gz')
    currentSize = predVol.GetSize()
    targetSize = origVol.GetSize()
    resampledVol = resample_data_or_seg(data=np.expand_dims(sitk.GetArrayFromImage(predVol).swapaxes(0,2),0), 
        new_shape=targetSize, is_seg=True)
    writer = sitk.ImageFileWriter()
    writer.SetFileName('/home/abhinav/selfSupervised_kidney/testPredsInv_scratch/prediction_'+fName.split('_')[1]+'.nii.gz')
    writer.Execute(sitk.GetImageFromArray(resampledVol[0].swapaxes(0,2)))
    print('Resized Case '+fName+' from size '+str(currentSize)+' to '+str(resampledVol[0].shape))

def resampleVolume(path,fName):
#     initTime = time.time()
    vol = sitk.ReadImage(path+fName+'/imaging.nii.gz')
    label = sitk.ReadImage(path+fName+'/segmentation.nii.gz')
    volArr = np.expand_dims(sitk.GetArrayFromImage(vol).swapaxes(0,2),0)
    labelArr = np.expand_dims(sitk.GetArrayFromImage(label).swapaxes(0,2),0)

    originalSpacing = vol.GetSpacing()
    targetSpacing = [3.22,1.62,1.62]

    resampledVol,resampledLabel = resample_patient(volArr, labelArr, originalSpacing, targetSpacing)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path+fName+'/resampled_dkfz_vol.nii.gz')
    writer.Execute(sitk.GetImageFromArray(resampledVol[0].swapaxes(0,2)))
    writer.SetFileName(path+fName+'/resampled_dkfz_labels.nii.gz')
    writer.Execute(sitk.GetImageFromArray(resampledLabel[0].swapaxes(0,2)))
    print('Resized Case '+fName+' from size '+str(vol.GetSize())+' to '+str(resampledVol[0].shape))

# fName = 'case_00023'
path = '/home/abhinav/kits_train/'#'/scratch/abhinavdhere/kits_test/'
fList = os.listdir(path)
for fName in fList:
    # inverseResample(path,fName,'Data')
    resampleVolume(path,fName)

'''
Old code using SimpleITK library
'''

# resampledVol = sitk.ReadImage(path+fName+'/resampled_vol.nii.gz')
    # resampledSize = resampledVol.GetSize()
# predVol = sitk.GetArrayFromImage(predVol)

# resizedVol = cropResize(predVol,currentSize,resampledSize)

# def resample(vol,new_spacing,targetSize):
#     orig_spacing = [3.22,1.62,1.62]#vol.GetSpacing()
#     orig_size = np.array(vol.GetSize(),dtype=np.int)

#     resample = sitk.ResampleImageFilter()
#     resample.SetInterpolator(sitk.sitkLinear)
#     resample.SetOutputDirection(vol.GetDirection())
#     resample.SetOutputOrigin(vol.GetOrigin())
#     resample.SetOutputSpacing(new_spacing)

#     # newSize = orig_size*(np.array(orig_spacing)/np.array(new_spacing))
#     # newSize = np.floor(newSize).astype('int')
#     resample.SetSize(targetSize)
    
#     newVol = resample.Execute(vol)
#     return newVol

# def resampleVolume(path,fName,volType):
#     initTime = time.time()
#     new_spacing = [3.22,1.62,1.62]
    
#     if volType=='Data':
#         vol = sitk.ReadImage(path+fName+'/imaging.nii.gz')
#     elif volType=='Labels':
#         vol = sitk.ReadImage(path+fName+'/segmentation.nii.gz')

#     newVol = resample(vol,new_spacing)

#     print('Took '+str(time.time()-initTime)+' seconds to resize.')
#     print('Resized Case '+fName+' from size '+str(vol.GetSize())+' to '+str(newVol.GetSize()))
#     writer = sitk.ImageFileWriter()
#     if volType=='Data':
#         writer.SetFileName(path+fName+'/resampled_vol.nii.gz')
#     elif volType=='Labels':
#         writer.SetFileName(path+fName+'/resampled_labels.nii.gz')
#     writer.Execute(newVol)

# def inverseResample(path,fName,volType):
#     pdb.set_trace()
#     origVol = sitk.ReadImage(path+fName+'/imaging.nii.gz')
#     predVol = sitk.ReadImage('/home/abhinav/selfSupervised_kidney/testPreds/prediction_'+fName+'.nii.gz')
#     new_spacing = origVol.GetSpacing()
#     currentSize = predVol.GetSize()
#     targetSize = origVol.GetSize()
#     invResampledVol = resample(predVol,new_spacing,targetSize)
#     # if not np.equal(np.array(origVol.GetSize()),np.array(invResampledVol.GetSize())).all():
#     #     invResampledVol = cropResize(invResampledVol,invResampledVol.GetSize(),origVol.GetSize())
#     writer = sitk.ImageFileWriter()
#     writer.SetFileName('/home/abhinav/selfSupervised_kidney/testPredsInv/prediction_'+fName+'.nii.gz')
#     if np.equal(np.array(origVol.GetSize()),np.array(invResampledVol.GetSize())).all():
#         writer.Execute(invResampledVol)
#         print('Resized Case '+fName+' from size '+str(currentSize)+' to '+str(invResampledVol.GetSize()))