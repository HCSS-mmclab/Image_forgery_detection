import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

def get_dataframe(k_fold, data_dir, data_folder, out_dim = 1):

    # data 읽어오기 (pd.read_csv / DataFrame으로 저장)
    data_folder = 'images/'
    df_train = pd.read_csv(os.path.join(data_dir, data_folder, 'train.csv'))

    # 비어있는 데이터 버리고 데이터 인덱스를 재지정함
    # df_train = df_train[df_train['인덱스 이름'] != -1].reset_index(drop=True)

    # 이미지 이름을 경로로 변환하기 (pd.DataFrame / apply, map, applymap에 대해 공부)
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}train', x))  # f'{x}.jpg'

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
    print(f'Dataset: {k_fold}-fold cross-validation')
    # img_id2fold = {i: i % k_fold for i in range(img_ids)}
    # df_train['fold'] = df_train['img_id'].map(img_id2fold)

    org_img_name = df_train['org_filename'].unique()
    img_name2fold = {img: i % k_fold for i, img in enumerate(org_img_name)}
    df_train['fold'] = df_train['org_filename'].map(img_name2fold)

    ###############################################
    manipul_unique = sorted(df_train['manipulation'].unique())
    mani_class = {str(mani): i for i, mani in enumerate(manipul_unique)}
    ###############################################

    # test data (학습이랑 똑같게 함)
    df_test = pd.read_csv(os.path.join(data_dir, data_folder, 'test.csv'))
    df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}test', x)) # f'{x}.png'

    return df_train, df_test, mani_class



class resamplingDataset_modified(Dataset):
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
        target_list = [self.csv.iloc[index].p1, self.csv.iloc[index].p5]   #scaling factor, scaling시에만 적용된다

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


#########################################
# raise dataset
#########################################
def get_dataframe_raise(data_dir, data_folder, out_dim = 1):

    # data 읽어오기 (pd.read_csv / DataFrame으로 저장)
    data_folder = 'images/'
    df_train = pd.read_csv(os.path.join(data_dir, data_folder, 'rasie_dataset.CSV'))

    # 비어있는 데이터 버리고 데이터 인덱스를 재지정함
    # df_train = df_train[df_train['인덱스 이름'] != -1].reset_index(drop=True)

    # 이미지 이름을 경로로 변환하기 (pd.DataFrame / apply, map, applymap에 대해 공부)
    df_train['image_name'] = df_train['image_name'] + '_' + df_train['patch'].astype(str)
    df_train['image_name'] = df_train['image_name'].apply(lambda x: x+'.png')
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}raise', x))  # f'{x}.jpg'

    # 원본데이터=0, 외부데이터=1
    df_train['is_ext'] = 0

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
    manipul_unique = sorted(df_train['manipulation'].unique())
    mani_class = {str(mani): i for i, mani in enumerate(manipul_unique)}
    ###############################################

    # test data (학습이랑 똑같게 함)
    # df_test = pd.read_csv(os.path.join(data_dir, data_folder, 'test.csv'))
    # df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}test', x)) # f'{x}.jpg'

    return df_train, mani_class



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
        target_list = [self.csv.iloc[index].p1, self.csv.iloc[index].p5]   #scaling factor, scaling시에만 적용된다

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
        albumentations.RandomBrightness(limit=0.1, p=0.75),
        albumentations.RandomContrast(limit=0.1, p=0.75),
        albumentations.CLAHE(clip_limit=2.0, p=0.3),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.Cutout(max_h_size=int(image_size * 0.1), max_w_size=int(image_size * 0.1), num_holes=1, p=0.7),
        albumentations.RandomGamma(gamma_limit=(80, 120), eps=None,always_apply=False, p=0.5),
        albumentations.Normalize()
        # shift
    ])

    transforms_val = albumentations.Compose([
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

