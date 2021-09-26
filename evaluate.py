import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import torch
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import get_dataframe, get_transforms, resamplingDataset_raise, resamplingDataset_modified, get_dataframe_raise, resamplingDataset_orgimg
from models import JUNet, MISLnet, HCSSNet_1d, HCSSNet_2d
from utils.util import *
from utils.torchsummary import summary

Precautions_msg = '(주의사항) Stone dataset의 경우 사람당 4장의 이미지기때문에 batch사이즈를 4의 배수로 해야 제대로 평가 된다.'


'''
- evaluate.py

학습한 모델을 평가하는 코드
Test셋이 아니라 학습때 살펴본 validation셋을 활용한다. 
grad-cam 실행한다. 


#### 실행법 ####
Terminal을 이용하는 경우 경로 설정 후 아래 코드를 직접 실행
python evaluate.py --kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3 --n-epochs 30

pycharm의 경우: 
Run -> Edit Configuration -> evaluate.py 가 선택되었는지 확인 
-> parameters 이동 후 아래를 입력 -> 적용하기 후 실행/디버깅
--kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 30 --k-fold 5



edited by MMCLab, 허종욱, 2020
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--out-dim', type=int, default=4)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--use-ext', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./SR_weights')
    parser.add_argument('--log-dir', type=str, default='./SR_logs')
    parser.add_argument('--oof-dir', type=str, default='./SR_oofs')

    parser.add_argument('--k-fold', type=int, default=5)
    parser.add_argument('--eval', type=str, choices=['best', 'best_no_ext', 'final'], default="best")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')

    args, _ = parser.parse_known_args()
    return args

def dec_matrix(output): # peak로 구한 output 2개
    angle = []
    resize = []
    output = output.reshape(args,2)
    # output = torch.vstack((output1,output2)) # output 2개를 합쳐서 matrix로 구현
    for i in range(output.shape[0]):
        U, S, V = torch.linalg.svd(output)
        resize = S[::-1] # resize
        uv = torch.matmul(U,V)
        radians = torch.arccos(uv[0,0]) # cos->radians
        angle = radians * 180 / np.pi # angle
    return angle, resize

def decomposition_val_epoch(model, loader, scaling_class):
    '''

    Output:
    val_loss, acc,
    auc   : 전체 데이터베이스로 진행한 validation
    auc_no_ext: 외부 데이터베이스를 제외한 validation
    '''
    # scaling_class = {'0.5': 0, '0.6': 1, '0.7': 2, '0.8': 3, '0.9': 4, '1.0': 5, '1.1': 6, '1.2': 7, '1.3': 8, '1.4': 9, '1.5': 10, '1.6': 11, '1.7': 12, '1.8': 13, '1.9': 14, '2.0': 15}
    model.eval()
    val_loss = []
    val_r_loss = []
    val_s_loss = []
    acc = []
    scaling_total = np.zeros(16)
    scaling_correct = np.zeros(16)

    with torch.no_grad():
        for (data, target) in tqdm(loader):

            data, target = data.to(device), torch.squeeze(target.to(device))
            logits = model(data)
            t_angle, t_scale = dec_matrix(target)
            p_angle, p_scale = dec_matrix(logits)
            for i in range(8):
                angle_loss = t_angle[i] - p_angle[i]
            #######################
            for i in range(target.shape[0]):
                if(abs(logits[i,1] - target[i,1])<0.05):
                    acc.append(1)
                else:
                    acc.append(0)
                tmp = scaling_class[str(np.round(target[i, 1].detach().cpu().numpy(), 1))]
                scaling_total[tmp] += 1
                scaling_correct[tmp] += (abs(logits[i, 1] - target[i, 1]) < 0.05).type(torch.float).detach().cpu()

            #######################

    val_loss = np.mean(val_loss)
    val_r_loss = np.mean(val_r_loss)
    val_s_loss = np.mean(val_s_loss)
    acc_score = (np.sum(acc) / len(acc))*100
    return val_loss, val_r_loss, val_s_loss, acc_score, scaling_correct, scaling_total

def val_epoch_fft1(model, loader):#, scaling_class):
    '''

    Output:
    val_loss, acc,
    auc   : 전체 데이터베이스로 진행한 validation
    auc_no_ext: 외부 데이터베이스를 제외한 validation
    '''
    # scaling_class = {'0.5': 0, '0.6': 1, '0.7': 2, '0.8': 3, '0.9': 4, '1.0': 5, '1.1': 6, '1.2': 7, '1.3': 8, '1.4': 9, '1.5': 10, '1.6': 11, '1.7': 12, '1.8': 13, '1.9': 14, '2.0': 15}
    model.eval()
    val_loss = []
    y_pred = []
    y = []
    # val_r_loss = []
    # val_s_loss = []
    # acc = []
    # scaling_total = np.zeros(16)
    # scaling_correct = np.zeros(16)

    with torch.no_grad():
        for (data, target) in tqdm(loader):

            data, target = data, torch.squeeze(target) # to(device)
            # print(data.shape)
            # print(target.shape)
            logits = model(data)

            #######################
            # for i in range(target.shape[0]):
            #     if(abs(logits[i,1] - target[i,1])<0.05):
            #         acc.append(1)
            #     else:
            #         acc.append(0)
            #     tmp = scaling_class[str(np.round(target[i, 1].detach().cpu().numpy(), 1))]
            #     scaling_total[tmp] += 1
            #     scaling_correct[tmp] += (abs(logits[i, 1] - target[i, 1]) < 0.05).type(torch.float).detach().cpu()
            loss = criterion(logits, target)
            val_loss.append(loss)#.detach().cpu().numpy())
            y_pred.append(logits.detach().numpy())
            y.append(target.detach().numpy())
            #######################

    val_loss = np.mean(val_loss)
    print(target.shape)
    print(logits.shape) #14,4
    print(len(target)) #14
    print(len(y_pred))
    # val_r_loss = np.mean(val_r_loss)
    # val_s_loss = np.mean(val_s_loss)
    # acc_score = (np.sum(acc) / len(acc))*100
    return val_loss, y_pred, y#val_r_loss, val_s_loss, acc_score, scaling_correct, scaling_total

def val_epoch_fft2(model, loader):
    '''
    Output:
    val_loss, acc,
    auc   : 전체 데이터베이스로 진행한 validation
    auc_no_ext: 외부 데이터베이스를 제외한 validation
    '''

    model.eval()
    val_loss = []
    y_pred = []
    y = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):

            data, target = data, torch.squeeze(target)
            logits = model(data)

            loss = criterion(logits, target)
            # loss = torch.log10(loss)
            val_loss.append(loss.detach().cpu().numpy())
            y_pred += logits.detach().tolist()
            y += target.detach().tolist()

    val_loss = np.mean(val_loss)

    return val_loss, np.asarray(y_pred), np.asarray(y)

def val_epoch_org(model, loader, target_idx, is_ext=None, n_test=1, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):

            # if args.use_meta:
            data, meta = data
            data, meta, target = data.to(device), meta.to(device), target.to(device)
            logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
            probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
            for I in range(n_test):
                l = model(get_trans(data, I), meta)
                logits += l
                probs += l.softmax(1)
            # else:
            #     data, target = data.to(device), target.to(device)
            #     logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
            #     probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
            #     for I in range(n_test):
            #         l = model(get_trans(data, I))
            #         logits += l
            #         probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return LOGITS, PROBS
    else:
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
        auc = roc_auc_score((TARGETS == target_idx).astype(float), PROBS[:, target_idx])
        auc_no_ext = roc_auc_score((TARGETS[is_ext == 0] == target_idx).astype(float), PROBS[is_ext == 0, target_idx])
        return val_loss, acc, auc, auc_no_ext


def val_epoch_stonedata(model, loader, target_idx, is_ext=None, n_test=1, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []

    with torch.no_grad(): # grad-cam 할때는 꺼야함
        for (data, target) in tqdm(loader):

            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            # 4장 의료 데이터를 묶음으로 확률계산. 평균이용
            for b_i in range(int(data.shape[0] / 4)):
                b_i4 = b_i*4
                logits[0+b_i4:4+b_i4, 0] = torch.mean(logits[0+b_i4:4+b_i4, 0])
                logits[0+b_i4:4+b_i4, 1] = torch.mean(logits[0+b_i4:4+b_i4, 1])
                probs[0+b_i4:4+b_i4, 0] = torch.mean(probs[0+b_i4:4+b_i4, 0])
                probs[0+b_i4:4+b_i4, 1] = torch.mean(probs[0+b_i4:4+b_i4, 1])

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()


    if get_output:
        return LOGITS, PROBS, TARGETS
    else:
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
        auc = roc_auc_score((TARGETS == target_idx).astype(float), PROBS[:, target_idx])
        auc_no_ext = roc_auc_score((TARGETS[is_ext == 0] == target_idx).astype(float), PROBS[is_ext == 0, target_idx])
        return val_loss, acc, auc, auc_no_ext


def pca(X, k):
    Cov = torch.matmul(X.T, X)/X.shape[0]
    print(Cov)
    U, S, V = torch.linalg.svd(Cov)
    return U[:,:k]

def main():

    '''
    ####################################################
    # stone data 데이터셋 : dataset.get_df_stone
    ####################################################
    '''
    # df_train, df_test, meta_features, n_meta_features, target_idx = get_dataframe(
    #     k_fold = args.k_fold,
    #     out_dim = args.out_dim,
    #     data_dir = args.data_dir,
    #     data_folder = args.data_folder,
    #     use_meta = args.use_meta,
    #     use_ext = args.use_ext
    # )

    # df_train, df_test, scaling_class = get_dataframe(args.k_fold, args.data_dir, args.data_folder, args.out_dim)
    d_v1 = pd.DataFrame(columns=['a', 'b', 'c', 'd', 'a*', 'b*', 'c*', 'd*'])
    transforms_train, transforms_val = get_transforms(args.image_size)

    # LOGITS = []
    # PROBS = []
    # TARGETS = []
    # dfs = []
    # y_pred=[]
    # y=[]
    df_val1, df_val2 = get_dataframe_raise(args.data_dir, args.data_folder, args.out_dim)

    # 모델 트랜스폼 가져오기

    dataset_valid_set1 = resamplingDataset_raise(df_val1, 'valid', args.image_size, transform=transforms_val)
    valid_loader_set1 = torch.utils.data.DataLoader(dataset_valid_set1, batch_size=args.batch_size,
                                                    num_workers=args.num_workers)
    ################################################
    ################################################
    ################################################
    '''
    if args.eval == 'best':
        model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
    elif args.eval == 'best_no_ext':
        model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_no_ext_fold{fold}.pth')
    if args.eval == 'final':
        model_file = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

    # model = ModelClass(
    #     args.enet_type,
    #     n_meta_features=n_meta_features,
    #     n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
    #     out_dim=args.out_dim
    # )
    model = ModelClass(
        args.enet_type,
        out_dim=args.out_dim,
        pretrained=True,
        im_size=args.image_size
    )
    model.load_state_dict(torch.load(model_file))

    model = model.to(device)

    try:  # single GPU model_file
        model.load_state_dict(torch.load(model_file), strict=True)
    except:  # multi GPU model_file
        state_dict = torch.load(model_file)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)

    if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
        model = torch.nn.DataParallel(model)

    model.eval()'''
    ################################################
    ################################################
    ################################################
    model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold.pth')
    # model_file3 = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold.pth')
    #if os.path.isfile(model_file):
    model = ModelClass(
        args.enet_type,
        out_dim=args.out_dim,
        pretrained=True,
        im_size=args.image_size
    )
    model.load_state_dict(torch.load(model_file))
    # model = model.to(device)
    try:  # single GPU model_file
        model.load_state_dict(torch.load(model_file), strict=True)
    except:  # multi GPU model_file
        state_dict = torch.load(model_file)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)

    if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
        model = torch.nn.DataParallel(model)

    model.eval()
    # else:
    #     model = ModelClass(
    #         args.enet_type,
    #         out_dim=args.out_dim,
    #         pretrained=True,
    #         im_size=args.image_size
    #     )
    #
    #     model = model.to(device)

        # model summary
        # if args.use_meta:
        #     pass
        #     # 코드 확인이 필요함
        #     # summary(model, [(3, args.image_size, args.image_size), n_meta_features])
        # else:
        #     if fold == 0: # 한번만
        #         summary(model, (3, args.image_size, args.image_size))


    try:  # single GPU model_file
        model.load_state_dict(torch.load(model_file), strict=True)
    except:  # multi GPU model_file
        state_dict = torch.load(model_file)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)

    if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
        model = torch.nn.DataParallel(model)

    model.eval()
    print('validation accuracy')
    '''
    ####################################################
    # stone data를 위한 평가함수 : val_epoch_stonedata
    ####################################################
    '''
    # val1_loss, y_pred, y = val_epoch_fft2(model, valid_loader_set1)
    val_loss_set1, logits1, target1 = val_epoch_fft2(model, valid_loader_set1)
    # print(len(logits1))
    # print(type(target1))
    for i in range(logits1.shape[0]):
        visual_v1 = {
            'a': target1[i, 0],
            'b': target1[i, 1],
            'c': target1[i, 2],
            'd': target1[i, 3],
            'a*': logits1[i, 0],
            'b*': logits1[i, 1],
            'c*': logits1[i, 2],
            'd*': logits1[i, 3],
        }
        d_v1 = d_v1.append(visual_v1, ignore_index=True)
    d_v1.to_csv(f'./visual/{args.kernel_type}_val1.csv')
    df = pd.read_csv(f'./visual/{args.kernel_type}_val1.csv')

    print('---------------------------------------------------------------------')

    # y_pred = pca(torch.tensor(y_pred),1)
    # y = pca(torch.tensor(y),1)
    # plt.plot(data[:,])
    plt.show()
    # val2_loss = val_epoch_fft(model, valid_loader_set2)
    print(f'val1_loss = {val_loss_set1}')
    #, val2_loss = {val2_loss}')
        # this_LOGITS, this_PROBS, this_TARGETS = val_epoch_stonedata(model, valid_loader, target_idx, is_ext=df_valid['is_ext'].values, n_test=8, get_output=True)

        # LOGITS.append(this_LOGITS)
        # PROBS.append(this_PROBS)
        # TARGETS.append(this_TARGETS)
        # dfs.append(df_valid)

        # SCALING = []
        # val_loss, val_r_loss, val_s_loss, acc_score, scaling_correct, scaling_total = val_epoch(model, valid_loader, scaling_class)
        # class_answer = scaling_correct / scaling_total
        ###############################################

        # for _ in range(len(scaling_class.keys())):
            # c = scaling_class[list(scaling_class.keys())[_]]
            # scaling_acc = f'{list(scaling_class.keys())[_]} : {round(class_answer[c] * 100, 2)}%, {int(scaling_correct[c])} / {int(scaling_total[c])}'
            # print(scaling_acc)
            # SCALING.append(scaling_acc)
        ###############################################


    # dfs = pd.concat(dfs).reset_index(drop=True)
    # dfs['pred'] = np.concatenate(PROBS).squeeze()[:, target_idx]
    #
    # Accuracy = (round(dfs['pred'])== dfs['target']).mean() * 100.
    # auc_all_raw = roc_auc_score(dfs['target'] == target_idx, dfs['pred'])
    #
    # dfs2 = dfs.copy()
    # for i in folds:
    #     dfs2.loc[dfs2['fold'] == i, 'pred'] = dfs2.loc[dfs2['fold'] == i, 'pred'].rank(pct=True)
    # auc_all_rank = roc_auc_score(dfs2['target'] == target_idx, dfs2['pred'])
    #
    # if args.use_ext:
    #     # 외부데이터를 사용할 경우, 외부데이터를 제외하고 모델을 따로 평가해본다.
    #     dfs3 = dfs[dfs.is_ext == 0].copy().reset_index(drop=True)
    #     auc_no_ext_raw = roc_auc_score(dfs3['target'] == target_idx, dfs3['pred'])
    #
    #     for i in folds:
    #         dfs3.loc[dfs3['fold'] == i, 'pred'] = dfs3.loc[dfs3['fold'] == i, 'pred'].rank(pct=True)
    #     auc_no_ext_rank = roc_auc_score(dfs3['target'] == target_idx, dfs3['pred'])
    #
    #     content = time.ctime() + ' ' + f'Eval {args.eval}:\nAccuracy : {Accuracy:.5f}\n' \
    #                                    f'auc_all_raw : {auc_all_raw:.5f}\nauc_all_rank : {auc_all_rank:.5f}\n' \
    #                                    f'auc_no_ext_raw : {auc_no_ext_raw:.5f}\nauc_no_ext_rank : {auc_no_ext_rank:.5f}\n'
    # else:
    #     content = time.ctime() + ' ' + f'Eval {args.eval}:\nAccuracy : {Accuracy:.5f}\n' \
    #               f'AUC_all_raw : {auc_all_raw:.5f}\nAUC_all_rank : {auc_all_rank:.5f}\n'
    #
    # # 로그파일 맨 뒤에 결과 추가해줌
    # print(content)
    # with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
    #     appender.write(content + '\n')
    #
    # np.save(os.path.join(args.oof_dir, f'{args.kernel_type}_{args.eval}_oof.npy'), dfs['pred'].values)
    #
    # # 결과 csv 저장
    # dfs[['filepath', 'patient_id', 'target', 'pred']].to_csv(os.path.join(args.oof_dir, f'{args.kernel_type}_{args.eval}_oof.csv'), index=True)


if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')
    args = parse_args()
    os.makedirs(args.oof_dir, exist_ok=True)
    # if args.use_gradcam:
    #     os.makedirs(os.path.join(args.gradcam_dir, args.kernel_type), exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    # 네트워크 타입 설정
    if 'JUNet' in args.enet_type:
        ModelClass = JUNet
    elif 'MISLnet' in args.enet_type:
        ModelClass = MISLnet
    elif 'HCSSNet_1d' in args.enet_type:
        ModelClass = HCSSNet_1d
    elif 'HCSSNet_2d' in args.enet_type:
        ModelClass = HCSSNet_2d
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    device = torch.device('cuda')
    criterion = nn.MSELoss()

    main()
