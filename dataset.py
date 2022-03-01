#-*_ coding:utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

def RandomCrop(image, output_size):
    rows,cols = image.shape[:2]
    row_point = np.random.randint(0, rows - output_size)
    col_point = np.random.randint(0, cols - output_size)
    # dst = image[row_point[0]:row_point[0] + output_size, col_point[0]:col_point[0] + output_size]
    dst = image[row_point:row_point + output_size, col_point:col_point + output_size]
    return dst

# Train
def get_dataframe(k_fold, data_dir, data_folder, out_dim = 1):

    # data 읽어오기 (pd.read_csv / DataFrame으로 저장)
    data_folder = 'images/'
    df_train = pd.read_csv(os.path.join(data_dir, data_folder, 'train.csv'))

    # 이미지 이름을 경로로 변환하기 (pd.DataFrame / apply, map, applymap에 대해 공부)
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}train', x))  # f'{x}.jpg'
    # df_train['org_img_name'] = df_train['image_name'].apply(lambda x: (x.split('_')[0]))

    # 원본데이터=0, 외부데이터=1
    df_train['is_ext'] = 0

    '''
    ####################################################
    교차 검증 구현 (k-fold cross-validation)
    ####################################################
    '''
    # 교차 검증을 위해 이미지 리스트들에 분리된 번호를 매김
    img_ids = len(df_train['img_id'].unique())
    print(f'Original dataset의 이미지수 : {img_ids}')

    # 데이터 인덱스 : fold 번호. (fold)번 분할뭉치로 간다
    # train.py input arg에서 k-fold를 수정해줘야함 (default:5)
    # print(f'Dataset: {k_fold}-fold cross-validation')
    # img_id2fold = {i: i % k_fold for i in range(img_ids)}
    # df_train['fold'] = df_train['img_id'].map(img_id2fold)


    # test data (학습이랑 똑같게 함)
    # df_test = pd.read_csv(os.path.join(data_dir, data_folder, 'test.csv'))
    # df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}test', x)) # f'{x}.jpg'

    return df_train#, df_test

# Train
# Gray-Down-Crop(512)-까지 된 상태
class resamplingDataset_orgimg(Dataset): # train dataset 동적으로 만드는 class
    def __init__(self, csv, mode, image_size=256, transform=None):

        self.csv = pd.concat([csv], ignore_index=True)
        self.mode = mode  # train / valid
        self.transform = transform
        self.image_size = image_size

        # self.r_test_delta = 45
        # self.s_test_delta = 0.05

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        ################################################################# - make_val_set
        # image = cv2.imread(path + file_name, cv2.IMREAD_GRAYSCALE)
        #
        # rows, cols = image.shape[:2]
        # image_center = (cols // 2, rows // 2)
        #
        # # scaling list
        # # rot_factor = int(np.random.uniform(low=1, high=45, size=(1,)))
        # scal_factor = int(np.random.uniform(low=22, high=39, size=(1,))) * 0.05
        #
        # m = cv2.getRotationMatrix2D(image_center, 0, float(scal_factor))  # rotation, scaling
        # # manipulation
        # dst = cv2.warpAffine(image, m, (cols, rows), flags=cv2.INTER_LINEAR)  # (bound_w, bound_h)
        # dst = dst[rows // 2 - 128:rows // 2 + 128, cols // 2 - 128:cols // 2 + 128]
        #################################################################

        image = cv2.imread(self.csv.iloc[index].filepath, cv2.IMREAD_GRAYSCALE)
        rows, cols = image.shape[:2]
        image_center = (cols // 2, rows // 2)
        ################################################
        rot = int(np.random.uniform(low=1, high=45, size=(1,)))
        scale = int(np.random.uniform(low=22, high=39, size=(1,)))*0.05 #1.1 ~ 1.9 - 0.05  110/5=22  190/5=38

        rot_factor=rot
        # rot_factor=0
        scal_factor=scale
        # scal_factor = 1
        ################################################
        matrix = cv2.getRotationMatrix2D(image_center, float(rot_factor), float(scal_factor))
        dst = cv2.warpAffine(image, matrix, (cols, rows), cv2.INTER_LINEAR)
        # image = cv2.warpAffine(image, matrix, (cols, rows), flags=cv2.INTER_LINEAR)  # (bound_w, bound_h)
        dst = dst[rows // 2 - 128:rows // 2 + 128, cols // 2 - 128:cols // 2 + 128]
        ################################################
        # center crop

        # Matrix
        target_list = [matrix[0][0],matrix[0][1],matrix[1][0],matrix[1][1]]  # scaling factor, scaling시에만 적용된다
        factor = [scale,rot]
        # Classification & Regression
        # target_list = [scal_factor, rot_factor]


        res = self.transform(image=dst)
        image = res['image'].astype(np.float32)
        #########################################

        image = np.expand_dims(image, axis=0)
        ################
        image = cv2.Laplacian(image, cv2.CV_32F, -1) ###############################################################################################################################
        # image = np.transpose(image, (2, 0, 1))  # GrayScale
        ################
        data = torch.tensor(image).float()

        return data, torch.tensor(target_list).float(), torch.tensor(factor).float()
#########################################

#########################################
# Validation
def get_dataframe_val(data_dir, data_folder, out_dim = 1):

    # data 읽어오기 (pd.read_csv / DataFrame으로 저장)
    data_folder = 'images/'
    df_train = pd.read_csv(os.path.join(data_dir, data_folder, 'valid_S.csv')) # scaling, rotation, all
    df_train2 = pd.read_csv(os.path.join(data_dir, data_folder, 'valid_R.csv'))
    df_train3 = pd.read_csv(os.path.join(data_dir, data_folder, 'valid_S+R.csv'))
    # 비어있는 데이터 버리고 데이터 인덱스를 재지정함
    # df_train = df_train[df_train['인덱스 이름'] != -1].reset_index(drop=True)

    # 이미지 이름을 경로로 변환하기 (pd.DataFrame / apply, map, applymap에 대해 공부)
    # df_train['image_name'] = df_train['image_name'] + '_' + df_train['patch'].astype(str)
    # df_train['image_name'] = df_train['image_name'].apply(lambda x: x+'.png')
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}valid_S', f'{x}.png'))  # f'{x}.jpg'

    # df_train2['image_name'] = df_train2['image_name'] + '_' + df_train2['patch'].astype(str)
    # df_train2['image_name'] = df_train2['image_name'].apply(lambda x: x + '.png')
    df_train2['filepath'] = df_train2['image_name'].apply(
        lambda x: os.path.join(data_dir, f'{data_folder}valid_R', f'{x}.png'))  # f'{x}.jpg'

    df_train3['filepath'] = df_train3['image_name'].apply(
        lambda x: os.path.join(data_dir, f'{data_folder}valid_S+R', f'{x}.png'))  # f'{x}.jpg'
    # 원본데이터=0, 외부데이터=1
    # df_train2['is_ext'] = 0

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
    # print(mani_class)
    # manipul_unique2 = sorted(df_train2['manipulation'].unique())
    # mani_class2 = {str(mani): i for i, mani in enumerate(manipul_unique2)}
    ###############################################

    # test data (학습이랑 똑같게 함)
    # df_test = pd.read_csv(os.path.join(data_dir, data_folder, 'test.csv'))
    # df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}test', x)) # f'{x}.jpg'

    return df_train, df_train2, df_train3#mani_class, df_train2, mani_class2


# Validation - Classification, Regression, Matrix
class resamplingDataset_val(Dataset):
    def __init__(self, csv, mode, image_size=256, transform=None):
        self.csv = csv
        self.mode = mode # train / valid
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        image = cv2.imread(self.csv.iloc[index].filepath, cv2.IMREAD_GRAYSCALE)

        target_list = [np.round(self.csv.iloc[index].p1,2), np.round(self.csv.iloc[index].p2,2), np.round(self.csv.iloc[index].p4,2), np.round(self.csv.iloc[index].p5,2)]   #scaling factor, scaling시에만 적용된다
        target_factor = [np.round(self.csv.iloc[index].scaling,2), np.round(self.csv.iloc[index].rotation,1)]

        # albumentation 적용
        res = self.transform(image=image)
        image = res['image'].astype(np.float32)
        #########################################
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # # 흑백 이미지 변환 후 차원 변경 [1, 1024, 1024]
        image = np.expand_dims(image, axis=0)
        #########################################
        ################
        image = cv2.Laplacian(image, cv2.CV_32F, -1) ###############################################################################################################################
        # image = np.transpose(image, (2, 0, 1))  # GrayScale
        ################

        # 학습용 데이터 리턴
        data = torch.tensor(image).float()

        return data, torch.tensor(target_list).float(), torch.tensor(target_factor).float()

# class resamplingDataset_raise(Dataset):
#     def __init__(self, csv, mode, image_size=256, transform=None):
#         self.csv = csv
#         self.mode = mode # train / valid
#         self.transform = transform
#         self.image_size = image_size
#
#     def __len__(self):
#         return self.csv.shape[0]
#
#     def __getitem__(self, index):
#
#         image = cv2.imread(self.csv.iloc[index].filepath, cv2.IMREAD_GRAYSCALE)
#         # target_list = [self.csv.iloc[index].scale, self.csv.iloc[index].rot]
#         target_list = [self.csv.iloc[index].p1, self.csv.iloc[index].p2, self.csv.iloc[index].p4,self.csv.iloc[index].p5]
#         # albumentation 적용
#         res = self.transform(image=image)
#         image = res['image'].astype(np.float32)
#         #########################################
#         # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # # 흑백 이미지 변환 후 차원 변경 [1, 1024, 1024]
#         # image = cv2.Laplacian(image, cv2.CV_32F, -1)
#         image = np.expand_dims(image, axis=0)
#         #########################################
#
#         # image = np.transpose(image, (2, 0, 1))# GrayScale
#
#         # 학습용 데이터 리턴
#         data = torch.tensor(image).float()
#
#         return data, torch.tensor(target_list).float()

def get_transforms(image_size):
    transforms_train = albumentations.Compose([
        # albumentations.RandomBrightness(limit=0.1, p=0.75),
        # albumentations.RandomContrast(limit=0.1, p=0.75),
        # albumentations.CLAHE(clip_limit=2.0, p=0.3),
        # albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        # albumentations.Cutout(max_h_size=int(image_size * 0.1), max_w_size=int(image_size * 0.1), num_holes=1, p=0.7),
        # albumentations.RandomGamma(gamma_limit=(80, 120), eps=None,always_apply=False, p=0.5),
        albumentations.Normalize(mean=(0.5419184366861979),std=(0.14091745018959045))
        # shift
    ])

    transforms_val = albumentations.Compose([
        albumentations.Normalize(mean=(0.5419184366861979),std=(0.14091745018959045))
    ])

    return transforms_train, transforms_val

def get_meta_data_stoneproject(df_train, df_test):
    '''
    ####################################################
                        안씀
    ####################################################
    '''

    return 0,0,0,0


