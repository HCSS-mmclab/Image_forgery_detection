import torch, torch.nn as nn, torch.nn.functional as F
import torch.fft as fft
import torch
import cv2
import matplotlib.pyplot as plt

class MISLnet(nn.Module):

    def __init__(self, backbone_type, im_size=1024, out_dim=2, pretrained = False):
        super(MISLnet, self).__init__()

        self.register_parameter("const_weight", None)
        self.const_weight = nn.Parameter(torch.randn(size=[3, 1, 5, 5]), requires_grad=True)
        self.conv1 = nn.Conv2d(3, 96, 7, stride=2, padding=4)
        self.conv2 = nn.Conv2d(96, 64, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(64, 128, 1, stride=1)

        self.fc1 = nn.Linear(6272, 512)
        self.fc2 = nn.Linear(512, 200)
        self.fc3 = nn.Linear(200, out_dim)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2)

    def normalized_F(self):
        central_pixel = (self.const_weight.data[:, 0, 2, 2])
        for i in range(3):
            sumed = self.const_weight.data[i].sum() - central_pixel[i]
            self.const_weight.data[i] /= sumed
            self.const_weight.data[i, 0, 2, 2] = -1.0

    def forward(self, x):
        # Constrained-CNN
        self.normalized_F()
        x = F.conv2d(x, self.const_weight)
        # CNN
        x = self.conv1(x)
        x = self.max_pool(torch.tanh(x))
        x = self.conv2(x)
        x = self.max_pool(torch.tanh(x))
        x = self.conv3(x)
        x = self.max_pool(torch.tanh(x))
        x = self.conv4(x)
        x = self.max_pool(torch.tanh(x))
        x = self.conv5(x)
        x = self.avg_pool(torch.tanh(x))

        # Fully Connected
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        logist = self.fc3(x)

        return logist



class JUNet(nn.Module):

    def __init__(self, backbone_type, im_size, out_dim=2, pretrained = False):
        super(JUNet, self).__init__()

        self.register_parameter("const_weight", None)
        self.num_const_w = 1
        self.const_weight = nn.Parameter(torch.randn(size=[self.num_const_w, 1, 5, 5]), requires_grad=True)

        # self.conv2d_row = nn.Conv2d(self.num_const_w,1,kernel_size=5,stride=1,padding=2)
        # self.conv2d_col = nn.Conv2d(self.num_const_w,1,kernel_size=5,stride=1,padding=2)

        self.conv1d_row1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.conv1d_row2 = nn.Conv2d(in_channels=16,out_channels=8,kernel_size=5,stride=1,padding=2)
        self.conv1d_row3 = nn.Conv2d(in_channels=8,out_channels=1,kernel_size=3,stride=1,padding=1)

        self.conv1d_col1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.conv1d_col2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5,stride=1, padding=2)
        self.conv1d_col3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3,stride=1, padding=1)

        # layers after fft
        self.conv1d_row = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=2)
        self.conv1d_col = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=2)

        self.row_fc1 = nn.Linear(im_size-4, im_size-4)
        self.row_fc2 = nn.Linear(im_size-4, 64)

        self.col_fc1 = nn.Linear(im_size-4, im_size-4)
        self.col_fc2 = nn.Linear(im_size-4, 64)

        self.fc1 = nn.Linear(128,2)

    def normalized_F(self):
        central_pixel = (self.const_weight.data[:, 0, 2, 2])
        for i in range(self.num_const_w):
            sumed = self.const_weight.data[i].sum() - central_pixel[i]
            # self.const_weight.data[i] /=sumed
            # self.const_weight.data[i,0,2,2] = -1
            self.const_weight.data[i] *= (-8/sumed) # sumed
            self.const_weight.data[i,0,2,2] = 8

    def forward(self, x):

        # Constrained-CNN
        # 이곳의 레이어를 mantranet의 residual 추출 모듈 (pre-trained)로 교체해야함
        self.normalized_F()
        x = F.conv2d(x, self.const_weight)#-4
        x = torch.relu(x)
        # fft layer - Ryu 2011논문에서 착안. 다른 방식을 도입해도 좋음 (e.g. 유사 radon transform)
        fft_row = torch.abs(fft.fft(x, dim=2))#(8,1,1020,1020)

        fft_row = self.conv1d_row1(fft_row)
        fft_row = torch.relu(fft_row)
        fft_row = self.conv1d_row2(fft_row)
        fft_row = torch.relu(fft_row)
        fft_row = self.conv1d_row3(fft_row)

        mean_fft_row = torch.mean(fft_row, dim=3)  # shape = (8,1,1020)

        # col fft
        fft_col = x.transpose(2,3)# transpose해야 conv1d를 x축 방향으로 수행할 수 있음
        fft_col = torch.abs(fft.fft(fft_col, dim=3))

        fft_col = self.conv1d_col1(fft_col)
        fft_col = torch.relu(fft_col)
        fft_col = self.conv1d_col2(fft_col)
        fft_col = torch.relu(fft_col)
        fft_col = self.conv1d_col3(fft_col)

        mean_fft_col = torch.mean(fft_col, dim=2)
        # plt.plot(mean_fft_col[0,:,:].detach().cpu().numpy())
        # plt.show()

        # 열과 행에서 각각 추출한 fft feature를 이용하여 판별
        x_row = self.conv1d_row(mean_fft_row)
        x_row = torch.relu(x_row)
        x_row = self.row_fc1(x_row)
        x_row = torch.relu(x_row)
        x_row = self.row_fc2(x_row)
        x_row = torch.relu(x_row)

        x_col = self.conv1d_col(mean_fft_col)
        x_col = torch.relu(x_col)
        x_col = self.col_fc1(x_col)
        x_col = torch.relu(x_col)
        x_col = self.col_fc2(x_col)
        x_col = torch.relu(x_col)

        # parameter estimator
        # 위에서 뽑은 특징으로 parameter를 추정하는 부분
        # 알려진 optimal 공식을 적용해서 수정하는 방향으로 구상
        out = torch.cat((x_row, x_col), dim=-1) # (8,128)
        logist = self.fc1(out)

        return logist



class HCSSNet(nn.Module):

    def __init__(self, backbone_type, im_size, out_dim=2, pretrained = False):
        super(HCSSNet, self).__init__()
        # self.lap = nn.Parameter(torch.zeros(size=[3, 3, 3, 3]), requires_grad=False)

        self.conv2d_1 = nn.Conv2d(1,16,kernel_size=7,stride=1,padding=3)
        self.conv2d_2 = nn.Conv2d(16,32,kernel_size=5,stride=1,padding=2)
        self.conv2d_3 = nn.Conv2d(32,32,kernel_size=5,stride=1,padding=2)

        self.conv2d_4 = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3)
        self.conv2d_5 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv2d_6 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)

        self.row_fc1 = nn.Linear(im_size*32, im_size)
        self.row_fc2 = nn.Linear(im_size, 128)
        self.row_fc3 = nn.Linear(128,64)

        self.col_fc1 = nn.Linear(im_size*32, im_size)
        self.col_fc2 = nn.Linear(im_size, 128)
        self.col_fc3 = nn.Linear(128,64)

        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        fft_row = torch.abs(fft.fft(x,dim=2))#(8,1,1024,1024) dim=(-2,-1)

        # aa = fft_row.detach().cpu().numpy()
        # aa[:,:,:20,:20]=0
        # aa[:, :, 1000:, 1000:] = 0
        # plt.plot(aa[1,0,500,:])
        # plt.show()
        # plt.plot(aa[1,0,:,500])
        # plt.show()
        fft_row = self.conv2d_1(fft_row)
        fft_row = torch.relu(fft_row)

        fft_row = self.conv2d_2(fft_row)
        fft_row = torch.relu(fft_row)

        fft_row = self.conv2d_3(fft_row)

        fft_col = torch.abs(fft.fft(x, dim=3))
        fft_col = self.conv2d_4(fft_col)
        fft_col = torch.relu(fft_col)

        fft_col = self.conv2d_5(fft_col)
        fft_col = torch.relu(fft_col)

        fft_col = self.conv2d_6(fft_col)


        mean_fft_row = torch.squeeze(torch.mean(fft_row, dim=2))
        mean_fft_col = torch.squeeze(torch.mean(fft_col, dim=3))
        mean_fft_row = mean_fft_row.view(mean_fft_row.shape[0],-1)
        mean_fft_col = mean_fft_col.view(mean_fft_col.shape[0],-1)
        # plt.plot(mean_fft_col[0,:,:].detach().cpu().numpy())
        # plt.show()

        # 열과 행에서 각각 추출한 fft feature를 이용하여 판별
        # print(mean_fft_row)
        x_row = self.row_fc1(mean_fft_row)
        x_row = torch.relu(x_row)
        x_row = self.row_fc2(x_row)
        x_row = torch.relu(x_row)
        x_row = self.row_fc3(x_row)

        x_col = self.col_fc1(mean_fft_col)
        x_col = torch.relu(x_col)
        x_col = self.col_fc2(x_col)
        x_col = torch.relu(x_col)
        x_col = self.col_fc3(x_col)

        # parameter estimator
        # 위에서 뽑은 특징으로 parameter를 추정하는 부분
        # 알려진 optimal 공식을 적용해서 수정하는 방향으로 구상
        logits = torch.cat((x_row, x_col), dim=-1)
        logits = self.fc1(logits)
        logits = self.fc2(logits)
        #fc layer
        logits = torch.squeeze(logits)

        return logits
