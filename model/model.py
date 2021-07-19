'''
M&Ms-2 predictive model.
You must supply 1 mandatory method:
- predict: uses the model to perform predictions over the test folder

Note that for simplicity, in this example a basic U-Net architecture is provided. 
Two 2D U-Nets were trained for SA and LA files without cross-validation nor pre/post-processing 
and with rotation,scaling, translation and affine shearing augmentation.
The current working directory at runtime will be the model folder. 
Any nested imports inside other modules must be realtive to its folder.

'''


import numpy as np   
from os.path import isfile
import os
import nibabel as nib
import skimage.transform as tr
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import sys
import collections
import SimpleITK as sitk

from architecture.generic_UNet import *
from architecture.preprocessing import PreprocessorFor2D,GenericPreprocessor
from architecture.segmentation_export import save_segmentation_nifti_from_softmax
from architecture.dir import mk_or_cleardir,mkdir_if_not_exist,sort_glob
from architecture.help import crop_sa_z_by_la,paste_la_to_sa, reindex_label

class model:
    def __init__(self):
        '''
        IMPORTANT: Initializes the model wrapper WITHOUT PARAMETERS, it will be called as model()
        '''
        self.input_dim = 256
        self.device =  torch.device("cpu" if not torch.cuda.is_available() else "cuda")



    def inference_la(self, net_unet,input_file,output_file):

        '''
        IMPORTANT: Mandatory. This function makes predictions for an entire test folder. 
        '''
        
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # PreprocessorFor2D
        d=collections.OrderedDict()
        d[0]="nonCT"
        normalization_schemes=d
        d=collections.OrderedDict()
        d[0]=False
        use_mask_for_norm=d
        transpose_forward=[0,1,2]
        intensity_properties=None
        preprocessor = PreprocessorFor2D(normalization_schemes, use_mask_for_norm,
                                          transpose_forward, intensity_properties)

        #properties == dct
        data, s, properties = preprocessor.preprocess_test_case([input_file],np.array([999,1,1]))
        #
        # data=torch.FloatTensor(1,1,288,288)
        # data=data.cuda()
        # sa_ed_pred = net_unet(img)
        patch_size=np.array([256,256])
        do_mirroring=True
        mirror_axes=(0,1)
        use_sliding_window=True
        step_size=0.5
        use_gaussian=True
        regions_class_order=None
        pad_border_mode="constant"
        pad_kwargs= {'constant_values': 0}
        all_in_gpu=False
        verbose=True
        mixed_precision=True
        net_unet.eval()
        ret = net_unet.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=patch_size, regions_class_order=regions_class_order,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision)
        
        softmax=ret[1]

        transpose_forward = [0,1,2]
        if transpose_forward is not None:
            transpose_backward = [0,1,2]
            softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0
        region_class_order=None
        npz_file=None
        save_segmentation_nifti_from_softmax(softmax, output_file, properties, interpolation_order, region_class_order,
                                            None, None,
                                            npz_file, None, force_separate_z, interpolation_order_z)
                                          
    def build_la_model(self):
        trained_unet = torch.load(os.path.join('./unet', '2d_model_best.model'))
        num_input_channels=1
        base_num_features=32
        num_classes=3
        net_num_pool_op_kernel_sizes=[[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
        conv_per_stage=2
        conv_op=torch.nn.Conv2d
        norm_op=torch.nn.InstanceNorm2d
        norm_op_kwargs={'eps': 1e-05, 'affine': True}
        dropout_op=torch.nn.Dropout2d
        dropout_op_kwargs={'p': 0, 'inplace': True}
        net_nonlin=torch.nn.LeakyReLU
        net_nonlin_kwargs={'negative_slope': 0.01, 'inplace': True}
        net_conv_kernel_sizes=[[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        net_unet=Generic_UNet(num_input_channels, base_num_features, num_classes,
                    len(net_num_pool_op_kernel_sizes),
                    conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                    dropout_op_kwargs,
                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                    net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True).float()
        net_unet.load_state_dict(trained_unet['state_dict'])
        net_unet.to(self.device)
        return net_unet

    def predictLA(self, output_folder, net_unet, case, la_ed):
        name=os.path.basename(la_ed)
        img=sitk.ReadImage(la_ed)
        print(la_ed)
        img_arr=sitk.GetArrayFromImage(img)

        img_arr=np.squeeze(img_arr)
        new_img = sitk.GetImageFromArray(img_arr.astype(np.float32)[None])
        new_img.SetSpacing([1,1,999])

        sitk.WriteImage(new_img,os.path.join(output_folder,"img_tmp.nii.gz"))

        self.inference_la(net_unet,os.path.join(output_folder,"img_tmp.nii.gz"),os.path.join(output_folder,"lab_tmp.nii.gz"))
            
        pred_lab=sitk.ReadImage(os.path.join(output_folder,"lab_tmp.nii.gz"))
        lab_arr=sitk.GetArrayFromImage(pred_lab)
        new_lab=sitk.GetImageFromArray(lab_arr)
        new_lab.CopyInformation(img)
            
        mkdir_if_not_exist(os.path.join(output_folder,case))
        new_lab=reindex_label(new_lab,{3:[2],1:[1]})
        sitk.WriteImage(new_lab,os.path.join(output_folder,case,name.replace('.nii.gz', '_pred.nii.gz')))
        sitk.WriteImage(img,os.path.join(output_folder,case,name))


    def predict(self, input_folder, output_folder):
        '''
        IMPORTANT: Mandatory. This function makes predictions for an entire test folder.
        '''

        net_unet_la = self.build_la_model()
        
        net_unet_sa = self.build_sa_model()

        for case in os.listdir(input_folder):

            print(case)
            try:
                la_ed = os.path.join(input_folder, case, case + '_LA_ED.nii.gz')
                self.predictLA(output_folder, net_unet_la, case, la_ed)

                la_es = os.path.join(input_folder, case, case + '_LA_ES.nii.gz')
                self.predictLA(output_folder, net_unet_la, case, la_es)
            except Exception as e:
                print(e)

        print("la done")
 

        # for case in ["174"]:
        for case in os.listdir(input_folder):


            print(f"#############################################################################{case}")
            try:
                sa_ed = os.path.join(input_folder, case, case + '_SA_ED.nii.gz')
                la_ed = os.path.join(output_folder, case, case + '_LA_ED_pred.nii.gz')
                self.predictSA(output_folder, net_unet_sa, case, sa_ed,la_ed)

                sa_es = os.path.join(input_folder, case, case + '_SA_ES.nii.gz')
                la_es = os.path.join(output_folder, case, case + '_LA_ES_pred.nii.gz')
                self.predictSA(output_folder, net_unet_sa, case, sa_es,la_es)
            except Exception as e:
                print(e)

        #post processing
        all_pred=sort_glob(f"{output_folder}/*/*pred.nii.gz")
        for p_l in all_pred:
            lab=sitk.ReadImage(p_l)
            lab=reindex_label(lab,{3:[3]})
            sitk.WriteImage(lab,p_l)



    


    def inference_sa(self, net_unet,input_file,output_file):
        
        '''
        IMPORTANT: Mandatory. This function makes predictions for an entire test folder. 
        '''
        
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # PreprocessorFor2D
        d=collections.OrderedDict()
        d[0]="nonCT"
        normalization_schemes=d
        d=collections.OrderedDict()
        d[0]=False
        use_mask_for_norm=d
        transpose_forward=[0,1,2]
        intensity_properties=None
        preprocessor = GenericPreprocessor(normalization_schemes, use_mask_for_norm,
                                          transpose_forward, intensity_properties)

        #properties == dct
        img=sitk.ReadImage(input_file)
        spacing=np.array(img.GetSpacing())[[2, 1, 0]]
        data, s, properties = preprocessor.preprocess_test_case([input_file],spacing)
        #
        # data=torch.FloatTensor(1,1,288,288)
        # data=data.cuda()
        # sa_ed_pred = net_unet(img)
        patch_size=np.array([10,192,192])
        do_mirroring=True
        mirror_axes=(0,1,2)
        use_sliding_window=True
        step_size=0.5
        use_gaussian=True
        regions_class_order=None
        pad_border_mode="constant"
        pad_kwargs= {'constant_values': 0}
        all_in_gpu=False
        verbose=True
        mixed_precision=True
        net_unet.eval()
        ret = net_unet.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=patch_size, regions_class_order=regions_class_order,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision)
        
        softmax=ret[1]

        transpose_forward = [0,1,2]
        if transpose_forward is not None:
            transpose_backward = [0,1,2]
            softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0
        region_class_order=None
        npz_file=None
        save_segmentation_nifti_from_softmax(softmax, output_file, properties, interpolation_order, region_class_order,
                                            None, None,
                                            npz_file, None, force_separate_z, interpolation_order_z)

    def build_sa_model(self):
        trained_unet = torch.load(os.path.join('unet', '3d_model_best.model'))
        num_input_channels=1
        base_num_features=32
        num_classes=3
        net_num_pool_op_kernel_sizes=[[1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2], [1, 2, 2]]
        conv_per_stage=2
        conv_op=torch.nn.Conv3d
        norm_op=torch.nn.InstanceNorm3d
        norm_op_kwargs={'eps': 1e-05, 'affine': True}
        dropout_op=torch.nn.Dropout3d
        dropout_op_kwargs={'p': 0, 'inplace': True}
        net_nonlin=torch.nn.LeakyReLU
        net_nonlin_kwargs={'negative_slope': 0.01, 'inplace': True}
        net_conv_kernel_sizes=[[1, 3, 3], [1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        net_unet=Generic_UNet(num_input_channels, base_num_features, num_classes,
                    len(net_num_pool_op_kernel_sizes),
                    conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                    dropout_op_kwargs,
                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                    net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True).float()
        net_unet.load_state_dict(trained_unet['state_dict'])
        net_unet.to(self.device)
        return net_unet
    


    def predictSA(self, output_folder, net_unet, case, sa_ed,la_ed_lab):
        name=os.path.basename(sa_ed)
        

        img=sitk.ReadImage(sa_ed)
        la_img=sitk.ReadImage(la_ed_lab)
        crop_img=crop_sa_z_by_la(img,la_img)

        sitk.WriteImage(crop_img,os.path.join(output_folder,"img_crop_tmp.nii.gz"))

        self.inference_sa(net_unet,os.path.join(output_folder,"img_crop_tmp.nii.gz"),os.path.join(output_folder,"lab_crop_tmp.nii.gz"))
            
        pred_lab=sitk.ReadImage(os.path.join(output_folder,"lab_crop_tmp.nii.gz"))

        new_lab=paste_la_to_sa(img,pred_lab)
            
        mkdir_if_not_exist(os.path.join(output_folder,case))
        new_lab=reindex_label(new_lab,{3:[2],1:[1]})
        sitk.WriteImage(new_lab,os.path.join(output_folder,case,name.replace('.nii.gz', '_pred.nii.gz')))
        sitk.WriteImage(img,os.path.join(output_folder,case,name))




if __name__=="__main__":
    p=model()
    p.predict('../validation','../validation_lab')