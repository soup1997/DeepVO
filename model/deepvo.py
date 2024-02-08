import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from utils.util import *

class DeepVO(nn.Module):
    def __init__(self, batchNorm=False):
        super(DeepVO,self).__init__()
        
        self.batchNorm = batchNorm
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_hidden_size = 1000
        self.rnn_dropout_ratio = 0.2
        self.rnn_dropout_between = 0.5
        
        self.conv1   = self.conv(self.batchNorm,   6,   64, kernel_size=7, stride=2, dropout=self.conv_dropout[0])
        self.conv2   = self.conv(self.batchNorm,  64,  128, kernel_size=5, stride=2, dropout=self.conv_dropout[1])
        self.conv3   = self.conv(self.batchNorm, 128,  256, kernel_size=5, stride=2, dropout=self.conv_dropout[2])
        self.conv3_1 = self.conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=self.conv_dropout[3])
        self.conv4   = self.conv(self.batchNorm, 256,  512, kernel_size=3, stride=2, dropout=self.conv_dropout[4])
        self.conv4_1 = self.conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=self.conv_dropout[5])
        self.conv5   = self.conv(self.batchNorm, 512,  512, kernel_size=3, stride=2, dropout=self.conv_dropout[6])
        self.conv5_1 = self.conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=self.conv_dropout[7])
        self.conv6   = self.conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=self.conv_dropout[8])

        # Comput the shape based on diff image size
        __tmp = torch.zeros(1, 6, 192, 640)
        __tmp = self.encode_image(__tmp)
        
        # RNN
        self.rnn = nn.LSTM(
                    input_size=int(torch.numel(__tmp)), 
                    hidden_size=self.rnn_hidden_size, 
                    num_layers=2,
                    bidirectional=False, 
                    dropout=self.rnn_dropout_between, 
                    batch_first=True)

        self.fc1 = nn.Sequential(
            nn.Dropout(self.rnn_dropout_ratio),
            nn.Linear(in_features=self.rnn_hidden_size, out_features=256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=3)
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(self.rnn_dropout_ratio),
            nn.Linear(in_features=self.rnn_hidden_size, out_features=256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=3)
        )

        # Initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                # layer 1
                kaiming_normal_(m.weight_ih_l0)  #orthogonal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)

                # layer 2
                kaiming_normal_(m.weight_ih_l1)  #orthogonal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n//4, n//2
                m.bias_hh_l1.data[start:end].fill_(1.)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        self.k1 = 100.0

    def conv(self, batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
        if batchNorm:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)
            )
        
    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6
    
    def load_pretrained_weight(self):
        pretrained_flownet = './trained/flownets_EPE1.951.pth'
        pretrained_w = torch.load(pretrained_flownet, map_location='cpu')

        model_dict = self.state_dict()
        update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
        model_dict.update(update_dict)
        self.load_state_dict(model_dict)
        
        print("Pretrained weights were loaded and updated...")
    
    def criterion(self, pred, local_pose):
        q_hat, q_local_gt = pred[:, :, :3], local_pose[:, :, :3]
        p_hat, p_local_gt = pred[:, :, 3:], local_pose[:, :, 3:]

        q_local_error = nn.MSELoss()(q_local_gt, q_hat)
        p_local_error = nn.MSELoss()(p_local_gt, p_hat)

        local_loss = p_local_error + (self.k1 * q_local_error)
        
        # global_loss = 0.0

        # for batch in range(global_pose.size(0)):
            
        #     first_global_pose = global_pose[batch, 0, :]
        #     last_global_pose = global_pose[batch, -1, :]

        #     H1 = torch_to_SE3(first_global_pose)
        #     res  = H1

        #     for sequence in range(local_pose.size(1)):
        #         Hrel = torch_to_SE3(local_pose[batch, sequence, :]).to(global_pose.device)
        #         res = torch.matmul(res.to(Hrel.device), Hrel)

        #     q_global_hat = torch_to_euler_angles(res[:3, :3])
        #     p_global_hat = res[:3, 3]

        #     q_global_error = self.mse(q_global_hat, last_global_pose[:3].to(q_global_hat.device))
        #     p_global_error = self.mse(p_global_hat, last_global_pose[3:].to(p_global_hat.device))
        #     global_loss += p_global_error + (self.k2 * q_global_error)
        
        # total_loss = global_loss + local_loss

        return local_loss
    
    def calc_err(self, pred, gt):
        r_mse = nn.MSELoss()(pred[:, :, :3], gt[:, :, :3])
        p_mse = nn.MSELoss()(pred[:, :, 3:], gt[:, :, 3:])
        r_rmse, p_rmse = torch.sqrt(r_mse), torch.sqrt(p_mse)

        return r_rmse, p_rmse

    def normalize(self, x):
        x_reshaped = x.view(-1, 3, 192, 640)
        rgb_mean = x_reshaped.mean(dim=(0, 2, 3)).view(1, 1, 3, 1, 1)
        rgb_std = x_reshaped.std(dim=(0, 2, 3)).view(1, 1, 3, 1, 1)
        
        x = (x - rgb_mean) / rgb_std

        return x

    def forward(self, x):
        x = self.normalize(x)
        
        x = torch.cat((x[:, :-1, :, :, :], x[:, 1:, :, :, :]), dim=2) # torch.Size([B=8, L=4, C=6, H=192, W=640])
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # CNN
        x = x.view(batch_size*seq_len, x.size(2), x.size(3), x.size(4))
        x = self.encode_image(x)
        x = x.view(batch_size, seq_len, -1)

        # RNN
        out, _ = self.rnn(x)
            
        rotation = self.fc1(out)
        translation = self.fc2(out)

        pose = torch.cat((rotation, translation), dim=2)

        return pose