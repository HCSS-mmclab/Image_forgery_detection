import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
# import timm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data.sampler import RandomSampler
from matplotlib import pyplot as plt
from utils.util import *
import pandas as pd
# import apex
# from apex import amp
# from dataset_peak import get_transforms, resamplingDataset, resamplingDataset_rasie, get_dataframe_val
from dataset import get_dataframe, get_transforms, resamplingDataset_raise, get_dataframe_raise, resamplingDataset_orgimg #, resamplingDataset_modified
# from models import JUNet, MISLnet, HCSSNet_1d, HCSSNet_2d, PeakEstimator, Resnet50#, DistilledVisionTransformer #, Stream3
from trans_model import Deit
from pytorchtools import EarlyStopping
Precautions_msg = '(주의사항) ---- \n'


'''
- train.py

모델을 학습하는 전과정을 담은 코드

#### 실행법 ####
Terminal을 이용하는 경우 경로 설정 후 아래 코드를 직접 실행
python train.py --kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 30

pycharm의 경우: 
Run -> Edit Configuration -> train.py 가 선택되었는지 확인 
-> parameters 이동 후 아래를 입력 -> 적용하기 후 실행/디버깅
--kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 30

*** def parse_args(): 실행 파라미터에 대한 모든 정보가 있다.  
*** def run(): 학습의 모든과정이 담긴 함수. 이곳에 다양한 trick을 적용하여 성능을 높혀보자. 
** def main(): fold로 나뉜 데이터를 run 함수에 분배해서 실행
* def train_epoch(), def val_epoch() : 완벽히 이해 후 수정하도록


Training list
python train.py --kernel-type 0907peak_test_m1_4e-5_50 --out-dim 4 --data-folder images/ --enet-type PeakEstimator --n-epochs 50 --batch-size 64 --k-fold 4 --image-size 256 --CUDA_VISIBLE_DEVICES 0
python predict.py --kernel-type test20210805 --out-dim 2 --data-folder images/ --enet-type MISLnet --n-epochs 50 --batch-size 8 --k-fold 4 --image-size 1024 --CUDA_VISIBLE_DEVICES 0

'''


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel-type', type=str, required=True)
    # kernel_type : 실험 세팅에 대한 전반적인 정보가 담긴 고유 이름

    parser.add_argument('--data-dir', type=str, default='./data/')
    # base 데이터 폴더 ('./data/')

    parser.add_argument('--data-folder', type=str, required=True)
    # 데이터 세부 폴더 예: 'original_stone/'
    # os.path.join(data_dir, data_folder, 'train.csv')

    parser.add_argument('--image-size', type=int, default='256')
    # 입력으로 넣을 이미지 데이터 사이즈

    parser.add_argument('--enet-type', type=str, required=True)
    # 학습에 적용할 네트워크 이름
    # {resnest101, seresnext101,
    #  tf_efficientnet_b7_ns,
    #  tf_efficientnet_b6_ns,
    #  tf_efficientnet_b5_ns...}

    parser.add_argument('--use-amp', action='store_true')
    # 'A Pytorch EXtension'(APEX)
    # APEX의 Automatic Mixed Precision (AMP)사용
    # 기능을 사용하면 속도가 증가한다. 성능은 비슷
    # 옵션 00, 01, 02, 03이 있고, 01과 02를 사용하는게 적절
    # LR Scheduler와 동시 사용에 버그가 있음 (고쳐지기전까지 비활성화)
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/2309

    parser.add_argument('--use-meta', action='store_true')
    # meta데이터 (사진 외의 나이, 성별 등)을 사용할지 여부

    parser.add_argument('--n-meta-dim', type=str, default='512,256')
    # meta데이터 사용 시 중간레이어 사이즈

    parser.add_argument('--out-dim', type=int, default=2)
    # 모델 출력 output dimension

    parser.add_argument('--DEBUG', action='store_true')
    # 디버깅용 파라미터 (실험 에포크를 5로 잡음)

    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    # 학습에 사용할 GPU 번호

    parser.add_argument('--k-fold', type=int, default=5)
    # data cross-validation
    # k-fold의 k 값을 명시

    parser.add_argument('--log-dir', type=str, default='./SR_logs')
    # Evaluation results will be printed out and saved to ./logs/
    # Out-of-folds prediction results will be saved to ./oofs/

    parser.add_argument('--accumulation-step', type=int, default=1)
    # Gradient accumulation step
    # GPU 메모리가 부족할때, 배치를 잘개 쪼개서 처리한 뒤 합치는 기법
    # 배치가 30이면, 60으로 합쳐서 모델 업데이트함

    parser.add_argument('--model-dir', type=str, default='./SR_weights')
    # weight 저장 폴더 지정
    # best :

    parser.add_argument('--use-ext', action='store_true')
    # 원본데이터에 추가로 외부 데이터를 사용할지 여부
    parser.add_argument('--patience', type=int, default=5)


    parser.add_argument('--batch-size', type=int, default=32) # 배치 사이즈
    parser.add_argument('--num-workers', type=int, default=4) # 데이터 읽어오는 스레드 개수
    parser.add_argument('--init-lr', type=float, default=2e-5) # 초기 러닝 레이트. pretrained를 쓰면 매우 작은값
    parser.add_argument('--n-epochs', type=int, default=100) # epoch 수
    args, _ = parser.parse_known_args()
    return args


def train_epoch(model, loader, optimizer):

    model.train()
    train_loss = []
    y_pred = []
    y = []
    bar = tqdm(loader)
    for i, (data, target) in enumerate(bar):

        optimizer.zero_grad()
        data, target = data.to(device), torch.squeeze(target.to(device))
        logits = model(data)

        loss = criterion(logits, target)

        loss.backward()
        # loss = torch.log10(loss)
        # # 그라디언트가 너무 크면 값을 0.5로 잘라준다 (max_grad_norm=0.5)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # gradient accumulation (메모리 부족할때)
        if args.accumulation_step:
            if (i + 1) % args.accumulation_step == 0:
                optimizer.step()
                #optimizer.zero_grad()
        else:
            optimizer.step()
            #optimizer.zero_grad()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))

        y_pred += logits.detach().cpu().tolist()
        y += target.detach().cpu().tolist()

    train_loss = np.mean(train_loss)
    return train_loss, y_pred, y


def val_epoch(model, loader):
    '''

    Output:
    val_loss, acc,
    auc   : 전체 데이터베이스로 진행한 validation
    auc_no_ext: 외부 데이터베이스를 제외한 validation
    '''

    model.eval()
    val_loss = []

    with torch.no_grad():
        for (data, target) in tqdm(loader):

            data, target = data.to(device), torch.squeeze(target.to(device))
            logits = model(data)

            loss = criterion(logits, target)
            # loss = torch.log10(loss)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)

    return val_loss

def run(df, df_val1, df_val2, transforms_train, transforms_val):
    # fold, df, transforms_train, transforms_val
    '''
    학습 진행 메인 함수
    :param fold: cross-validation에서 valid에 쓰일 분할 번호
    :param df: DataFrame 학습용 전체 데이터 목록
    :param transforms_train, transforms_val: 데이터셋 transform 함수
    '''
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    train_loss_list = []
    valid_loss_set1_list = []
    valid_loss_set2_list = []
    # if args.DEBUG:
    #     args.n_epochs = 5
    #     df_train = df[df['fold'] != fold].sample(args.batch_size * 5)
    #     df_valid = df[df['fold'] == fold].sample(args.batch_size * 5)
    # else:
    #     df_train = df[df['fold'] != fold]
    #     df_valid = df[df['fold'] == fold]
    #
    # if args.k_fold == 1:
    #     df_train = df_valid
    #
    #     # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
    #     # batch_normalization에서 배치 사이즈 1인 경우 에러 발생할 수 있으므로, 데이터 한개 버림
    #     if len(df_train) % args.batch_size == 1:
    #         df_train = df_train.sample(len(df_train)-1)
    #     if len(df_valid) % args.batch_size == 1:
    #         df_valid = df_valid.sample(len(df_valid)-1)

    # 데이터셋 읽어오기
    dataset_valid_set1 = resamplingDataset_raise(df_val1, 'valid', args.image_size, transform=transforms_val)
    dataset_valid_set2 = resamplingDataset_raise(df_val2, 'valid', args.image_size, transform=transforms_val)
    valid_loader_set1 = torch.utils.data.DataLoader(dataset_valid_set1, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_loader_set2 = torch.utils.data.DataLoader(dataset_valid_set2, batch_size=args.batch_size,
                                               num_workers=args.num_workers)

    model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold.pth')
    model_file3 = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold.pth')
    if os.path.isfile(model_file):
        model = ModelClass(
            args.enet_type,
            # out_dim=args.out_dim,
            # pretrained=True,
            # im_size = args.image_size
        )
        model.load_state_dict(torch.load(model_file))
    else:
        model = ModelClass(
            args.enet_type#,
            # out_dim=args.out_dim,
            # pretrained=True,
            # im_size=args.image_size
        )

    # GPU 여러개로 병렬처리
    # if DP:
    #     model = apex.parallel.convert_syncbn_model(model)
    model = model.to(device)

    val_loss_max = 99999.

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # if args.use_amp:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if DP:
        model = nn.DataParallel(model)

    # amp를 사용하면 버그 (use_amp 비활성화)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    # df_train = df[df['fold'] != fold]
    for epoch in range(1, args.n_epochs + 1):

        dataset_train = resamplingDataset_orgimg(df,'train', args.image_size, transform=transforms_train)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                   sampler=RandomSampler(dataset_train), num_workers=args.num_workers)

        print(time.ctime(), f'Epoch {epoch}')
        train_loss, logits1, target1 = train_epoch(model, train_loader, optimizer)
            # df = pd.read_csv(f'./visual/{args.kernel_type}_train.csv')
        # if args.k_fold == 1 :
        #     # skip validation
        #     val_loss_set1, acc, auc, auc_no_ext = [999.0, 0.0, 0.0, 0.0]
        #     if epoch + 5 > args.n_epochs :
        #         model_file4 = os.path.join(args.model_dir, f'{args.kernel_type}_e{args.n_epochs-epoch}.pth')
        #         torch.save(model.state_dict(), model_file4)
        #     val_loss_set2, acc, auc, auc_no_ext = [999.0, 0.0, 0.0, 0.0]
        #     if epoch + 5 > args.n_epochs:
        #         model_file4 = os.path.join(args.model_dir, f'{args.kernel_type}_e{args.n_epochs - epoch}.pth')
        #         torch.save(model.state_dict(), model_file4)
        if epoch > 0:
            val_loss_set1 = val_epoch(model, valid_loader_set1) #, val_r_loss, val_s_loss, acc_score
            val_loss_set2 = val_epoch(model, valid_loader_set2)

        else:
            val_loss_set1 = 1
            val_loss_set2 = 1

        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid set1 loss: {(val_loss_set1):.5f}, valid set2 loss: {(val_loss_set2):.5f}'
        print(content)
        early_stopping(val_loss_set1, model)

        if early_stopping.early_stop:
            print("Early stopping")
            plt.figure(figsize=(10,40))
            plt.subplot(4,1,1)
            train_min = min(train_loss_list)
            train_x = np.argmin(train_loss_list)
            valid_min_set1 = min(valid_loss_set1_list)
            valid_x_set1 = np.argmin(valid_loss_set1_list)
            valid_min_set2 = min(valid_loss_set2_list)
            valid_x_set2 = np.argmin(valid_loss_set2_list)
            plt.plot(train_loss_list)
            plt.text(train_x,train_min, round(train_min,4))
            plt.plot(valid_loss_set1_list)
            plt.text(valid_x_set1, valid_min_set1, round(valid_min_set1,4))
            plt.plot(valid_loss_set2_list)
            plt.text(valid_x_set2, valid_min_set2, round(valid_min_set2, 4))
            plt.legend(['train_loss', 'val_s_loss_set1','val_s_loss_set2'])
            plt.ylabel('loss')
            plt.title(f'{args.kernel_type}')
            plt.grid()

            plt.subplot(4,1,2)
            plt.plot(train_loss_list)
            plt.text(train_x, train_min, round(train_min, 4))
            plt.legend(['train_loss'])
            plt.grid()

            plt.subplot(4, 1, 3)
            plt.plot(valid_loss_set1_list)
            plt.text(valid_x_set1, valid_min_set1, round(valid_min_set1, 4))
            plt.legend(['val_s_loss_set1'])
            plt.grid()

            plt.subplot(4, 1, 4)
            plt.plot(valid_loss_set2_list)
            plt.text(valid_x_set2, valid_min_set2, round(valid_min_set2, 4))
            plt.legend(['val_s_loss_set2'])
            plt.grid()
            plt.savefig(f'./SR_results/{args.kernel_type}.jpg')
            plt.show()
            break
        train_loss_list.append(train_loss)
        valid_loss_set1_list.append(val_loss_set1)
        valid_loss_set2_list.append(val_loss_set2)

        # train, val loss, acc


        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

        scheduler_warmup.step()
        if epoch == 2:
            scheduler_warmup.step() # bug workaround

        if val_loss_set1 < val_loss_max:
            print('val_loss_max ({:.6f} --> {:.6f}). Saving model ...'.format(val_loss_max, val_loss_set1))
            torch.save(model.state_dict(), model_file)
            val_loss_max = val_loss_set1
            if epoch > 60:
                d_t = pd.DataFrame(columns=['a', 'b', 'c', 'd', 'a*', 'b*', 'c*', 'd*'])
                for i in range(len(logits1)):
                    visual_v1 = {
                        'a': target1[i][0],
                        'b': target1[i][1],
                        'c': target1[i][2],
                        'd': target1[i][3],
                        'a*': logits1[i][0],
                        'b*': logits1[i][1],
                        'c*': logits1[i][2],
                        'd*': logits1[i][3],
                    }
                    d_t = d_t.append(visual_v1, ignore_index=True)
            if epoch == args.n_epochs:
                d_t.to_csv(f'./visual/{args.kernel_type}_train.csv')
        if val_loss_set2 < val_loss_max:
            print('val_loss_max ({:.6f} --> {:.6f}). Saving model ...'.format(val_loss_max, val_loss_set2))
            torch.save(model.state_dict(), model_file)
            val_loss_max = val_loss_set2

    # if args.k_fold ==1 :
    #     torch.save(model.state_dict(), model_file)
    torch.save(model.state_dict(), model_file3)



def main():
    # 데이터셋 읽어오기
    df_train, df_test = get_dataframe(args.k_fold, args.data_dir, args.data_folder, args.out_dim)
    df_val1,df_val2 = get_dataframe_raise(args.data_dir, args.data_folder, args.out_dim)

    # 모델 트랜스폼 가져오기
    transforms_train, transforms_val = get_transforms(args.image_size)

    run(df_train, df_val1, df_val2, transforms_train, transforms_val)


if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')

    # argument값 만들기
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    ####################################
    ####################################
    # resnet50 = models.resnet50()
    # for param in resnet50.parameters():
    #     param.requires_grad = False
    # features = list(resnet50.fc.children())[:-1]
    # features.extend([nn.Linear(2048, 64),
    #                  nn.ReLU(),
    #                  nn.Linear(64, 4)])
    # resnet50.fc = nn.Sequential(*features)
    ####################################
    ####################################
    # 네트워크 타입 설정
    # if 'JUNet' in args.enet_type:
    #     ModelClass = JUNet
    # elif 'MISLnet' in args.enet_type:
    #     ModelClass = MISLnet
    # elif 'HCSSNet_1d' in args.enet_type:
    #     ModelClass = HCSSNet_1d
    # elif 'HCSSNet_2d' in args.enet_type:
    #     ModelClass = HCSSNet_2d
    # elif 'PeakEstimator' in args.enet_type:
    #     ModelClass = PeakEstimator
    # elif 'Resnet50' in args.enet_type:
    #     ModelClass = Resnet50
    if 'DistilledVisionTransformer' in args.enet_type:
        ModelClass = DistilledVisionTransformer
    elif 'Deit' in args.enet_type:
        ModelClass = Deit
    else:
        raise NotImplementedError()

    # GPU가 여러개인 경우 멀티 GPU를 사용함
    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    # 실험 재현을 위한 random seed 부여하기
    set_seed(2359)
    device = torch.device('cuda')
    criterion = nn.MSELoss()

    # 메인 기능 수행
    main()