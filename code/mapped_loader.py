from torchvision import datasets
from base import BaseDataLoader
from torch.utils.data.dataset import Dataset
from data_loader import threeD_Transforms
import numpy as np
import torch
import nibabel as nib
import os
from skimage.transform import resize
from scipy.ndimage import binary_dilation, zoom
from torchvision import transforms


class Nifti_Dataset(Dataset):
    def __init__(self, patient_df, params):
        self.patients = np.array(patient_df['pic_ID'])
        # self.labels = np.array(patient_df[params['inputs']['label']].astype(int))
        self.labels = np.array(patient_df[params['inputs']['label']])
        self.pt_sides = np.array(patient_df['Side'])
        self.pt_locations = np.array(patient_df['Location'])
        self.total_patients = len(self.patients)
        self.transform_1 = threeD_Transforms.Compose([
            # threeD_Transforms.RandomNoise(low=-0.1, high=0.1),
            threeD_Transforms.ToTensor()
        ])
        self.modalities = params['inputs']['modalities']
        self.image_input = params['inputs']['image_input']
        self.image_size = tuple(params['inputs']['image_size'])
        self.data_location = params['files']['data_location']
        self.training_path = params['files']['training_path']
        self.projection = False

        preprocess_folder = ''
        for m in self.modalities:
            preprocess_folder += str(m)[:-7]
        for i in self.image_size:
            preprocess_folder += str(i) + 'x'
        preprocess_folder += self.image_input

        self.preprocess_folder = os.path.join(self.training_path, preprocess_folder)

    def __getitem__(self, index):
        idx = self.patients[index]
        label = self.labels[index]
        side = self.pt_sides[index]
        location = self.pt_locations[index]
        # make a preprocessed directory if it doesn't exist
        if not os.path.exists(self.preprocess_folder):
            os.makedirs(self.preprocess_folder)
        idx_file = os.path.join(self.preprocess_folder, str(idx))
        if self.image_input == 'half_ab':
            if not os.path.exists(idx_file + '.pt'):

                series_list = [nib.load(os.path.join(self.data_location, str(idx)[:-1], series)).get_data().T for series
                               in self.modalities]
                series_array = np.array(series_list)
                series_array.astype(np.float32)
                series_array = (series_array - np.min(series_array)) / np.ptp(series_array)
                series_array_L = series_array[:, :, :, :int(series_array.shape[3] / 2)]
                series_array_R = series_array[:, :, :, int(series_array.shape[3] / 2):]
                series_array_R = np.flip(series_array_R, 3)
                series_resized_L = resize(series_array_L, self.image_size, mode='reflect', anti_aliasing=True)
                series_resized_R = resize(series_array_R, self.image_size, mode='reflect', anti_aliasing=True)
                tensor_L = self.transform_1(series_resized_L)
                tensor_R = self.transform_1(series_resized_R)
                torch.save(tensor_R, idx_file[:-1]+"A.pt")
                torch.save(tensor_L, idx_file[:-1]+"B.pt")

                image_tensor = torch.load(idx_file + '.pt')
            else:
                image_tensor = torch.load(idx_file + '.pt')

        elif self.image_input == 'region':
            if not os.path.exists(idx_file + '.pt'):
                series_list = [nib.load(os.path.join(self.data_location, str(idx)[:-1], series)).get_fdata().T for series
                               in self.modalities]
                series_array = np.array(series_list)
                series_array.astype(np.float32)
                series_array = (series_array - np.min(series_array)) / np.ptp(series_array)

                # load the atlas file
                ##TODO: update to location of this file for CT
                mask_file = nib.load(os.path.join(self.training_path, "vasc_terr_r.nii.gz"))
                atlas = mask_file.get_fdata() / 2

                label_terr = ['Whole_Brain',
                              'R_ACA',
                              'R_MCA',
                              'R_PCA',
                              'R_ICA',
                              'R_PICA',
                              'L_ACA',
                              'L_MCA',
                              'L_PCA',
                              'L_ICA',
                              'L_PICA', ]

                # generate compound mask
                mask = np.zeros(atlas.T.shape)
                location = location.split('/')
                for l in location:
                    if l.startswith('M'):
                        l = 'MCA'
                    elif l.startswith('PI'):
                        l = 'PICA'
                    elif l.startswith('P'):
                        l = 'PCA'
                    elif l.startswith('C'):
                        l = 'ICA'
                    index = side + "_" + l
                    mask += atlas.T == label_terr.index(index)

                # dilate the mask
                kernel = np.ones((3, 3, 3), np.uint8)
                img_dil = binary_dilation(mask, kernel, iterations=2)

                # multiply the mask
                series_array = series_array * img_dil

                # split into half
                if side == 'L':
                    series_array = series_array[:, :, :, :int(series_array.shape[3] / 2)]
                else:
                    series_array = series_array[:, :, :, int(series_array.shape[3] / 2):]
                    series_array = np.flip(series_array, 3)

                series_resized = resize(series_array, self.image_size, mode='reflect', anti_aliasing=True)
                image_tensor = self.transform_1(series_resized)

                if self.projection:
                    modal_0 = image_tensor[0, :, :, :]
                    modal_1 = image_tensor[1, :, :, :]
                    modal_2 = image_tensor[2, :, :, :]

                    resize_modal_0 = zoom(modal_0, (224 / modal_0.shape[0], 224 / modal_0.shape[1], 224 / modal_0.shape[2]))
                    resize_modal_1 = zoom(modal_1, (224 / modal_1.shape[0], 224 / modal_1.shape[1], 224 / modal_1.shape[2]))
                    resize_modal_2 = zoom(modal_2, (224 / modal_2.shape[0], 224 / modal_2.shape[1], 224 / modal_2.shape[2]))

                    modal_0_2d = np.zeros((3, 224, 224))
                    modal_1_2d = np.zeros((3, 224, 224))
                    modal_2_2d = np.zeros((3, 224, 224))
                    modal_0_2d[0] = resize_modal_0.sum(axis=0) / 224
                    modal_0_2d[1] = resize_modal_0.sum(axis=1) / 224
                    modal_0_2d[2] = resize_modal_0.sum(axis=2) / 224
                    modal_1_2d[0] = resize_modal_1.sum(axis=0) / 224
                    modal_1_2d[1] = resize_modal_1.sum(axis=1) / 224
                    modal_1_2d[2] = resize_modal_1.sum(axis=2) / 224
                    modal_2_2d[0] = resize_modal_2.sum(axis=0) / 224
                    modal_2_2d[1] = resize_modal_2.sum(axis=1) / 224
                    modal_2_2d[2] = resize_modal_2.sum(axis=2) / 224

                    image_tensor = torch.stack([torch.from_numpy(modal_0_2d), torch.from_numpy(modal_1_2d),
                                                torch.from_numpy(modal_2_2d)], dim=0)
                    image_tensor = image_tensor.float()
                    image_tensor = torch.clamp(image_tensor, min=0, max=1)
                torch.save(image_tensor, idx_file + ".pt")
            else:
                image_tensor = torch.load(idx_file + '.pt')
            image_tensor = image_tensor.float()

        return idx, image_tensor, label

    def __len__(self):
        return self.total_patients  # of how many examples(images?) you have



                # import pdb
                # import matplotlib.pyplot as plt
                # pdb.set_trace()
                # a = image_tensor[0,:,:,:].permute(1,2,0)
                # b = image_tensor[1,:,:,:].permute(1,2,0)
                # c = image_tensor[2,:,:,:].permute(1,2,0)
                # plt.imshow(a)
                # plt.show()