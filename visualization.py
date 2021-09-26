import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
kernel_type = 'stream1_v1_fixed_batch8'
# mode = ['train', 'val1', 'val2'] # train, val1, val2
mode = ['val1']


def pca(X, k):
    Cov = np.matmul(X.T, X) / X.shape[0]
    U, S, V = np.linalg.svd(Cov)
    return U[:, :k]


def dec_matrix(output): # peak로 구한 output 2개
    angle = []
    resize = []
    output = output.reshape(2,2)
    # output = torch.vstack((output1,output2)) # output 2개를 합쳐서 matrix로 구현
    for i in range(output.shape[0]):
        U, S, V = np.linalg.svd(output)
        resize = S[0] # resize
        uv = np.matmul(U,V)
        radians = np.arccos(uv[0,0]) # cos->radians
        angle = radians * 180 / np.pi # angle
    return angle, resize


for i in mode:
    df = pd.read_csv(f'./visual/{kernel_type}_{i}.csv')
    # R = np.array(df.iloc[:,1:5])
    # P = np.array(df.iloc[:,5:])
    #
    # R_pca = pca(R, 1)
    # P_pca = pca(P, 1)
    #
    # R_pca = np.matmul(R, R_pca)
    # P_pca= np.matmul(P, P_pca)
    #
    # df['R_pca'] = R_pca
    # df['P_pca'] = P_pca
    R_rot = []
    R_scale = []
    P_rot = []
    P_scale = []
    for r in range(df.shape[0]):
        rot,scale = dec_matrix(np.array(df.iloc[r,1:5]).reshape(2,2))
        R_rot.append(rot)
        R_scale.append(scale)
        rot, scale = dec_matrix(np.array(df.iloc[r,5:]).reshape(2,2))
        P_rot.append(rot)
        P_scale.append(scale)
    df['R_rot'] = R_rot
    df['R_scale'] =R_scale
    df['P_rot'] = P_rot
    df['P_scale'] = P_scale

    fig, ax = plt.subplots(1,1,figsize=(15, 15))
    # sns.regplot(x='R_pca', y='P_pca', data=df, label='abcd', fit_reg=False, color='gray')
    sns.regplot(x='R_rot', y='P_rot', data=df, label='rot', fit_reg=False, color='gray')
    # sns.regplot(x='a', y='a*', data=df, label='a',fit_reg=False)
    # sns.regplot(x='b', y='b*', data=df, label='b',fit_reg=False)
    # sns.regplot(x='c', y='c*', data=df, label='c',fit_reg=False)
    # sns.regplot(x='d', y='d*', data=df, label='d',fit_reg=False)

    # coordinate = np.array([-3.5,-1.0])
    # intercept = 0.07
    coordinate = np.array([0, 45])
    # intercept = 0.5


    plt.plot(coordinate,coordinate,color="red", label = 'ground_truth')
    # plt.plot(coordinate, 1*coordinate+intercept, color="blue", label = 'threshold')
    # plt.plot(coordinate, 1 * coordinate - intercept, color="blue", label='threshold')


    plt.title(f'visualization_{kernel_type}_{i}')
    plt.legend()
    plt.savefig(f'./visual/result_{kernel_type}_{i}_Rot_Scale')


# plt.show()

# d_t.to_csv(f'./visual/{args.kernel_type}_train.csv')
#         d_v1.to_csv(f'./visual/{args.kernel_type}_val1.csv')
#         d_v2.to_csv(f'./visual/{args.kernel_type}_val2.csv')