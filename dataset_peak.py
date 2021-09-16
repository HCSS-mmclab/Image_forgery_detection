import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from random import randrange


class genSynthImage():
    def __init__(self, image_size=256):
        self.count = 0
        self.image_size = image_size
        self.im_center = image_size //2
        self.image_shape = (image_size, image_size)

    def get_synth_image(self, noise_power = 10):
        # cv2.getRotationMatrix2D(center, angle, scale)
        # rotation 0~45까지 (0.5 간격)
        # scaling 1.1~2.0까지 (0.05 간격)

        while True:
            rot_factor = randrange(0, 451, 5) * 0.1
            scal_factor = randrange(110, 201, 5) * 0.01
            matrix = cv2.getRotationMatrix2D((self.im_center, self.im_center), rot_factor,scal_factor)
            matrix = [round(x,2) for x in [matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1]]]

            # # shearing
            # shear_factor = np.random.randint(low=-noise_power, high=noise_power, size=4) * 0.01
            # matrix = [matrix_ori[i]+shear_factor[i] for i in range(4)]

            # singular matrix 방지; ad != bc
            if matrix[0]*matrix[3] != matrix[1]*matrix[2]:
                break

        # Matrix 형성 및 inverse 계산
        A = np.array([[matrix[0], matrix[1]],[matrix[2], matrix[3]]])
        A_T_inv = np.linalg.inv(A.T)

        # y = x - round(x)
        peak_image = np.zeros(self.image_shape)
        M = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

        for m_i in range(M.shape[0]):
            m = M[m_i, :]
            A_T_inv_m = np.matmul(A_T_inv, m)

            # y = x - round(x)
            peak_normalized_loc = A_T_inv_m - np.round(A_T_inv_m)

            peak_im_loc =  self.im_center*peak_normalized_loc + self.im_center

            # 이미지 값 기록
            peak_image[int(peak_im_loc[0]), int(peak_im_loc[1])] = 1

        return peak_image, matrix

class resamplingDataset(Dataset):
    def __init__(self, csv, mode, image_size=256, noise_power=10, transform=None):
        # self.csv = pd.concat([csv] * 10, ignore_index=True).reset_index(drop=True)  # image 복사
        self.mode = mode  # train / valid
        self.transform = transform
        self.image_size = image_size
        self.im_center = image_size // 2
        self.image_shape = (image_size, image_size)

        self.image_num = 10000
        self.matrix = []
        self.image_list = []



    def __len__(self):
        return self.image_num

    def __getitem__(self, index):

        while True:
            rot_factor = randrange(0, 451, 5) * 0.1
            scal_factor = randrange(110, 201, 5) * 0.01
            matrix = cv2.getRotationMatrix2D((self.im_center, self.im_center), rot_factor,scal_factor)
            matrix = [round(x,2) for x in [matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1]]]

            # # shearing
            # shear_factor = np.random.randint(low=-noise_power, high=noise_power, size=4) * 0.01
            # matrix = [matrix_ori[i]+shear_factor[i] for i in range(4)]

            # singular matrix 방지; ad != bc
            if matrix[0]*matrix[3] != matrix[1]*matrix[2]:
                break

        # Matrix 형성 및 inverse 계산
        A = np.array([[matrix[0], matrix[1]],[matrix[2], matrix[3]]])
        A_T_inv = np.linalg.inv(A.T)

        # y = x - round(x)
        peak_image = np.zeros(self.image_shape)
        M = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

        for m_i in range(M.shape[0]):
            m = M[m_i, :]
            A_T_inv_m = np.matmul(A_T_inv, m)

            # y = x - round(x)
            peak_normalized_loc = A_T_inv_m - np.round(A_T_inv_m)

            peak_im_loc =  self.im_center*peak_normalized_loc + self.im_center

            # 이미지 값 기록
            peak_image[int(peak_im_loc[0]), int(peak_im_loc[1])] = 1

        image = np.expand_dims(peak_image, axis=0)

        # 학습용 데이터 리턴
        data = torch.tensor(image).float()

        # 변경 값 리턴하기
        target_list = matrix

        return data, torch.tensor(target_list).float()

class resamplingDataset_org(Dataset):
    def __init__(self, csv, mode, image_size=1024, noise_power=10, transform=None):
        # self.csv = pd.concat([csv] * 10, ignore_index=True).reset_index(drop=True)  # image 복사
        self.mode = mode  # train / valid
        self.transform = transform

        self.image_num = 20000
        self.matrix = []
        self.image_list = []

        self.image_size = image_size


        # 시간 절약을 위해 메모리에 데이터셋 미리 읽어들임
        generator = genSynthImage()

        for i in range(self.image_num):
            peak_image, matrix = generator.get_synth_image(noise_power=noise_power)
            self.image_list.append(peak_image)
            self.matrix.append(matrix)
        print('success to make images')


    def __len__(self):
        return self.image_num

    def __getitem__(self, index):
        image = self.image_list[index]
        target = self.matrix[index]
        target_list = []
        # # 변형 값 얻어내기
        # r_test_value = np.random.uniform(low=0, high=180, size=(1,)) // self.r_test_delta
        # s_test_value = np.random.uniform(low=self.s_test_low, high=self.s_test_high, size=(1,))
        # s_test_value -= (s_test_value % self.s_test_delta)
        #
        # # 변형
        # matrix = cv2.getRotationMatrix2D((int(self.image_size / 2), int(self.image_size / 2)), float(r_test_value),
        #                                  float(s_test_value))
        # image = cv2.warpAffine(image, matrix, (self.image_size, self.image_size), cv2.INTER_LINEAR)
        #
        # # albumentation 적용
        # res = self.transform(image=image)
        # image = res['image'].astype(np.float32)

        # 흑백 이미지 변환 후 차원 변경 [1, 1024, 1024]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=0)

        # 학습용 데이터 리턴
        data = torch.tensor(image).float()

        # 변경 값 리턴하기
        target_list = target

        return data, torch.tensor(target_list).float()

'''
###########################################
############## validation #################
###########################################
'''
def get_dataframe_val(data_dir, data_folder, out_dim = 1):

    # data 읽어오기 (pd.read_csv / DataFrame으로 저장)
    data_folder = 'images/'
    df_train = pd.read_csv(os.path.join(data_dir, data_folder, 'csv_val_set1.csv'))
    df_train2 = pd.read_csv(os.path.join(data_dir, data_folder, 'csv_val_set2.csv'))

    # 비어있는 데이터 버리고 데이터 인덱스를 재지정함
    # df_train = df_train[df_train['인덱스 이름'] != -1].reset_index(drop=True)

    # 이미지 이름을 경로로 변환하기 (pd.DataFrame / apply, map, applymap에 대해 공부)
    # df_train['image_name'] = df_train['image_name'] + '_' + df_train['patch'].astype(str)
    # df_train['image_name'] = df_train['image_name'].apply(lambda x: x+'.png')
    df_train['filepath'] = df_train['img_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}val_set1', x))  # f'{x}.jpg'
    df_train2['filepath'] = df_train2['img_name'].apply(
        lambda x: os.path.join(data_dir, f'{data_folder}val_set2', x))  # f'{x}.jpg'

    # # 원본데이터=0, 외부데이터=1
    # df_train['is_ext'] = 0

    '''
    ####################################################
    교차 검증 구현 (k-fold cross-validation)
    ####################################################
    '''
    # 교차 검증을 위해 이미지 리스트들에 분리된 번호를 매김
    # img_ids = len(df_train['img_id'].unique())
    # print(f'Original dataset의 이미지수 : {img_ids}')

    # 데이터 인덱스 : fold 번호. (fold)번 분할뭉치로 간다
    # train.py input arg에서 k-fold를 수정해줘야함 (default:5)
    # print(f'Dataset: {k_fold}-fold cross-validation')
    # img_id2fold = {i: i % k_fold for i in range(img_ids)}
    # df_train['fold'] = df_train['img_id'].map(img_id2fold)

    # org_img_name = df_train['org_filename'].unique()
    # img_name2fold = {img: i % k_fold for i, img in enumerate(org_img_name)}
    # df_train['fold'] = df_train['org_filename'].map(img_name2fold)

    ###############################################
    # manipul_unique = sorted(df_train['manipulation'].unique())
    # mani_class = {str(mani): i for i, mani in enumerate(manipul_unique)}
    ###############################################

    # test data (학습이랑 똑같게 함)
    # df_test = pd.read_csv(os.path.join(data_dir, data_folder, 'test.csv'))
    # df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}test', x)) # f'{x}.jpg'

    return df_train, df_train2



class resamplingDataset_rasie(Dataset):
    def __init__(self, csv, mode, image_size=256, transform=None):
        self.csv = csv
        self.mode = mode # train / valid
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        image = cv2.imread(self.csv.iloc[index].filepath)
        image = cv2.Laplacian(image, cv2.CV_8U, -1)
        target_list = [self.csv.iloc[index].p1, self.csv.iloc[index].p2, self.csv.iloc[index].p3, self.csv.iloc[index].p4]   #scaling factor, scaling시에만 적용된다

        # albumentation 적용
        res = self.transform(image=image)
        image = res['image'].astype(np.float32)

        # 흑백 이미지 변환 후 차원 변경 [1, 1024, 1024]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=0)
        # image = np.transpose(image, (2, 0, 1))

        # 학습용 데이터 리턴
        data = torch.tensor(image).float()

        return data, torch.tensor(target_list).float()





def get_transforms(image_size):
    transforms_train = albumentations.Compose([
        # albumentations.RandomBrightness(limit=0.1, p=0.75),
        # albumentations.RandomContrast(limit=0.1, p=0.75),
        # albumentations.CLAHE(clip_limit=2.0, p=0.3),  # Histogram Equalization
        # albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        # albumentations.Cutout(max_h_size=int(image_size * 0.1), max_w_size=int(image_size * 0.1), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        # albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val

def get_meta_data_stoneproject(df_train, df_test):
    '''
    ####################################################
                        안씀
    ####################################################
    '''

    return 0,0,0,0

