import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

kernel_type = '0924_test04_lr4e-5_ep100'
mode = ['train', 'val1', 'val2']          # train, val1, val2


def pca(X, k):
    Cov = np.matmul(X.T, X) / X.shape[0]
    U, S, V = np.linalg.svd(Cov)
    return U[:, :k]





for i in mode:
    df = pd.read_csv(f'./visual/{kernel_type}_{i}.csv')
    R = np.array(df.iloc[:,1:5])
    P = np.array(df.iloc[:,5:])

    R_pca = pca(R, 1)
    P_pca = pca(P, 1)

    R_pca = np.matmul(R, R_pca)
    P_pca= np.matmul(P, P_pca)

    df['R_pca'] = R_pca
    df['P_pca'] = P_pca



    fig, ax = plt.subplots(1,1,figsize=(15, 15))
    sns.regplot(x='R_pca', y='P_pca', data=df, label='abcd', fit_reg=False, color='gray')

    # sns.regplot(x='a', y='a*', data=df, label='a',fit_reg=False)
    # sns.regplot(x='b', y='b*', data=df, label='b',fit_reg=False)
    # sns.regplot(x='c', y='c*', data=df, label='c',fit_reg=False)
    # sns.regplot(x='d', y='d*', data=df, label='d',fit_reg=False)

    coordinate = np.array([-3.5,-1.0])
    intercept = 0.07


    plt.plot(coordinate,coordinate,color="red", label = 'ground_truth')
    plt.plot(coordinate, 1*coordinate+intercept, color="blue", label = 'threshold')


    plt.title(f'visualization_{kernel_type}_{i}')
    plt.legend()
    plt.savefig(f'./visual/result_{kernel_type}_{i}')


# plt.show()

# d_t.to_csv(f'./visual/{args.kernel_type}_train.csv')
#         d_v1.to_csv(f'./visual/{args.kernel_type}_val1.csv')
#         d_v2.to_csv(f'./visual/{args.kernel_type}_val2.csv')