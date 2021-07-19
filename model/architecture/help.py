import SimpleITK as sitk
import numpy as np
def binarize_numpy_array(array,ids ):
    out = np.zeros(array.shape, dtype=np.uint16)
    for L in ids:
        out = out + np.where(array == L, 1, 0)
    return out

def padd(min,max,padding,size):
    start = 0 if min - padding < 0 else min - padding
    stop = (size - 1) if max + padding > (size - 1) else max + padding
    return slice(start, stop)


def get_bounding_box_by_ids(x,padding=0,ids=[200,1220,2112]):
    res=[]
    # coor = np.nonzero(x)
    if isinstance(x,sitk.Image):
        x=sitk.GetArrayFromImage(x)
    out = binarize_numpy_array( x,ids)
    coor = np.nonzero(out)
    size=np.shape(out)

    for i in range(3):
        if len(coor[i])==0:
            return None
        xmin = np.min(coor[i])
        xmax = np.max(coor[i])
        res.append(padd(xmin,xmax,padding,size[i]))
    return res

def center_pad(center, padding, size):
    start = 0 if center - padding < 0 else center - padding
    stop = (size - 1) if center + padding > (size - 1) else center + padding
    return slice(start, stop)

def get_bbox( lab, pad_ratio=0.25):
    lab_arr=sitk.GetArrayFromImage(lab)
    assert np.argmin(lab_arr)==0
    #extract z slice
    print(lab_arr.shape)
    bbox=get_bounding_box_by_ids(lab_arr,padding=0,ids=[1,2,3])

    if bbox==None:
        return None

    center_bbox=[]
    center_bbox.append(bbox[0])
    for i,slice in enumerate(bbox[1:]):
        center=(slice.start+slice.stop)//2
        center_bbox.append(center_pad(center, int(lab_arr.shape[i+1] * pad_ratio), lab_arr.shape[i + 1]))


    return center_bbox

def bi_project_dataV2(source, target):
    # print(f"{source} {target}")
    target_img=target
    source_img=source
    # source_img=sitkResample3DV2(source_img,sitk.sitkLinear,target_img.GetSpacing())

    new_img=sitk.Image(target_img.GetSize(),source_img.GetPixelID())
    new_img.CopyInformation(target_img)
    size=(source_img.GetSize())

    print(size)
    for x in range(0,size[0]):
        for y in range(0,size[1]):
            for z in range(0,size[2]):


                point=source_img.TransformIndexToPhysicalPoint([x,y,z])
                # p=index2physicalpoint(target_img,[x,y,z])
                # index_la=la_img.TransformPhysicalPointToContinuousIndex(point)
                index_la=target_img.TransformPhysicalPointToIndex(point)
                # index_la=np.round(index_la)
                # i=physicalpoint2index(source_img, point)
                index_la=np.array(index_la)
                if index_la[0]<0 or index_la[0] >= target_img.GetSize()[0]:
                    continue
                if index_la[1] < 0 or index_la[1] >= target_img.GetSize()[1]:
                    continue
                if index_la[2] < 0 or index_la[2] >= target_img.GetSize()[2]:
                    continue
                new_img[int(index_la[0]),int(index_la[1]),int(index_la[2])]=0
                # print(index_la)
                # print(x,y,z)
                # new_img[x,y,z]=source_img.GetPixel(x,y,z)

                new_img[int(index_la[0]),int(index_la[1]),int(index_la[2])]=source_img[x,y,z]
                # new_img[x,y,z]=interplote(la_img,index_la)
    return new_img

def reindex_label( img, ids, to_nn=True):
    arr = sitk.GetArrayFromImage(img)
    new_array = np.zeros(arr.shape, np.int)
    for k in ids.keys():
        for i in ids[k]:
            new_array = new_array + np.where(arr == i, k, 0)
    new_array=new_array.astype(np.uint8)
    new_img = sitk.GetImageFromArray(new_array)
    new_img.CopyInformation(img)

    return new_img

def crop_by_bbox(img,bbox):
    crop_img = img[bbox[2].start:bbox[2].stop+1,bbox[1].start:bbox[1].stop+1,bbox[0].start:bbox[0].stop+1]
    return crop_img
    
def crop_sa_z_by_la(sa,lab):
    cons=bi_project_dataV2(lab,sa)
    bbox=get_bbox(cons)
    cons=reindex_label(cons,{1:[3]})
    bbox2=get_bbox(cons) # only rv for z axis


    if bbox==None or bbox2==None:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return sa
    if bbox2[0].stop!=bbox[0].stop:
        print("deletez")
    bbox[0]=bbox2[0]
    img=crop_by_bbox(sa,bbox)
    return img    

def recast_pixel_val(reference, tobeconverted):
    """
    Recast pixel value to be the same for segmentation and original image, othewise SimpleITK complains.
    :param reference:
    :param tobeconverted:
    :return:
    """
    pixelID = reference.GetPixelID()
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(pixelID)
    tobeconverted = caster.Execute(tobeconverted)
    return tobeconverted

def paste_roi_image(image_source, image_roi, reference_size=None):
    """ Resize ROI binary mask to size, dimension, origin of its source/original img.
        Usage: newImage = paste_roi_image(source_img_plan, roi_mask)
        Use only if the image segmentation ROI has the same spacing as the image source
    """
    # get the size and the origin from the source image
    if reference_size:
        newSize = reference_size
    else:
        newSize = image_source.GetSize()

    newOrigin = image_source.GetOrigin()
    # get the spacing and the direction from the mask or the image if they are identical
    newSpacing = image_source.GetSpacing()
    newDirection = image_source.GetDirection()

    # re-cast the pixel type of the roi mask
    image_roi = recast_pixel_val(image_source, image_roi)

    # black 3D image
    outputImage = sitk.Image(newSize, image_source.GetPixelIDValue())
    outputImage.SetOrigin(newOrigin)
    outputImage.SetSpacing(newSpacing)
    outputImage.SetDirection(newDirection)
    # img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0
    destinationIndex = outputImage.TransformPhysicalPointToIndex(image_roi.GetOrigin())
    # paste the roi mask into the re-sized image
    pasted_img = sitk.Paste(outputImage, image_roi, image_roi.GetSize(), destinationIndex=destinationIndex)

    return pasted_img


def paste_la_to_sa(sa_ed,sa_lab):
    lab=paste_roi_image(sa_ed,sa_lab)
    lab=sitk.Cast(lab,sitk.sitkUInt8)
    lab_array = sitk.GetArrayFromImage(lab)
    lab = sitk.GetImageFromArray(lab_array)
    lab.CopyInformation(sa_ed)
    return lab