import torch
import torch.nn as nn
import numpy as np 
import math 
import matplotlib.pyplot as plt 

class Linear_diagonal_weight(torch.nn.Module):
    def __init__(self, input_size, output_size, device):
        super(Linear_diagonal_weight,self).__init__()
        self.device=device
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        self.diagonal_mask = torch.eye(output_size, input_size)
    def forward(self, input):
        # use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        # if use_cuda:
        self.diagonal_mask = self.diagonal_mask.to(self.device)
        return  torch.mm(input, self.weight * self.diagonal_mask) + self.bias 


class IRLSTM(torch.nn.Module):
    def __init__(self, dim_hid, length_out=21, num_bio=5, num_classes=3, device="cuda"):
        super(IRLSTM, self).__init__()
        self.pi = torch.Tensor([np.pi])

        ####### LSTM #######
        self.input_size = num_bio + num_classes + 3 # 3 consists of age, sex, apoe4
        self.hidden_size = dim_hid
        self.length_out = length_out
        self.device = device
        self.num_bio = num_bio
        self.num_classes = num_classes

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # Decay weights
        # self.w_dg_x = Linear_diagonal_weight(self.input_size, self.input_size, config.DEVICE)
        self.w_dg_h = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)

        # x weights
        self.w_x = torch.nn.Linear(self.hidden_size, self.input_size, bias=True)
        self.w_xz = Linear_diagonal_weight(self.input_size, self.input_size, device)
        
        #beta weight
        self.w_b_dg = torch.nn.Linear(self.input_size*2, self.input_size, bias=True)
   
        # c weights
        self.w_uc = torch.nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.w_hc = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False) 
        self.w_mc = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)

        # o weights
        self.w_uo = torch.nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.w_ho = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False) 
        self.w_mo = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)

        # f weights
        self.w_uf = torch.nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.w_hf = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_mf = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)

        # r weights
        self.w_ur = torch.nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.w_hr = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)   
        self.w_mr = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)

        # PREDICTION MODULE
        self.w_dxh = torch.nn.Linear(self.hidden_size, num_classes, bias=True)
        self.w_gh = torch.nn.Linear(self.hidden_size, num_bio, bias=True)

        self.reset_parameters()
      
    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            torch.nn.init.uniform_(weight, -stdv, stdv)
    
    # def imputation_module(self, x_t, m_t, x_t_hat):

    #     x_t[x_t != x_t] = 0
    #     u_t_hat = x_t_hat 
    #     u_t = m_t * x_t + (1 - m_t) * u_t_hat    

    #     return u_t, u_t_hat

    def encoder_module(self, delta_t, m_t, u_t, h_t, c_t):

        self.pi = self.pi.to(self.device)

        gamma_h = torch.exp(-1* self.relu( self.w_dg_h(delta_t)))
        
        h_t_hat = gamma_h * h_t
        f_t = self.sigmoid(self.w_uf(u_t) + self.w_hf(h_t_hat) + self.w_mf(m_t))
        # print("beffore")
        # plt.hist(f_t.reshape(-1, 1).detach().cpu().numpy(), bins=100, facecolor="orange")
        # plt.rc('xtick', labelsize=12) 
        # plt.rc('ytick', labelsize=12) 
        # plt.xlim([0, 1])
        # plt.savefig("before.png")
        # plt.show()
        g_t = f_t - torch.sin(f_t * self.pi) * torch.cos(f_t * self.pi) / self.pi
        # print("after")
        # plt.hist(g_t.reshape(-1, 1).detach().cpu().numpy(), bins=100, facecolor="orange")
        # plt.rc('xtick', labelsize=12) 
        # plt.rc('ytick', labelsize=12) 
        # plt.xlim([0, 1])
        # plt.savefig("after.png")
        # plt.show()
        c_t_hat = self.tanh(self.w_uc(u_t) + self.w_hc(h_t_hat) + self.w_mc(m_t))
        c_t = g_t*c_t + (1-g_t)*c_t_hat
        o_t = self.sigmoid(self.w_uo(u_t) + self.w_ho(h_t_hat) + self.w_mo(m_t))
        h_t = o_t * self.tanh(c_t)  
        return h_t, c_t

    def prediction_module(self, g_t, h_t):
        x_t_plus_1_hat = self.w_gh(h_t) + g_t
        x_dx_t_plus_1_hat = self.w_dxh(h_t)
        return x_t_plus_1_hat, x_dx_t_plus_1_hat

    def forward(self, x_t):
        """
        data:   B x D
        out:    B x L x D

        """
        # data_bio, data_dg, data_dx = data[0].clone(), data[1].clone(), data[2].clone()
        # mask_bio, mask_dg, mask_dx = [ele.clone() for ele in mask]
        # delta_bio, delta_dg, delta_dx = [ele.clone() for ele in delta]

        B = x_t.shape[0]
        ############# 
        x_bio_hat = [] #torch.empty(data_i.shape)
        x_dx_hat = []
        ### Initialize h_t and c_t in LSTM
        h_t = torch.zeros((B, self.hidden_size), dtype=torch.float, device=self.device)
        c_t = torch.zeros((B, self.hidden_size), dtype=torch.float, device=self.device)

        # if self.training:
        #     num_hist_year = torch.randint(self.config.NUM_HISTORY_VISIT, L-1, (1, ))
        # else:
        #     num_hist_year = self.config.NUM_HISTORY_VISIT

        for tps in range(self.length_out):
            delta_t = torch.ones_like(x_t) * tps

            if tps == 0:
                u_t = x_t
                m_t = torch.ones_like(x_t)
            else:
                m_t = torch.zeros_like(x_t)
                m_t[:, self.num_bio: -self.num_classes] = 1  # DG is always observed
                delta_t[:, self.num_bio: -self.num_classes] = 1

            # Encoder Module
            m_t[:, self.num_bio: -self.num_classes] = 1 

            h_t, c_t = self.encoder_module(delta_t, m_t, u_t, h_t, c_t)
            # Prediction Module
            x_t_plus_1_hat, x_dx_t_plus_1_hat = self.prediction_module(u_t[:, :self.num_bio], h_t)

            u_t = torch.cat([x_t_plus_1_hat, x_t[:, self.num_bio: -self.num_classes], self.softmax(x_dx_t_plus_1_hat)], dim=1) # [bio, dg, dx]

            x_bio_hat.append(x_t_plus_1_hat)
            x_dx_hat.append(x_dx_t_plus_1_hat)
        ######### stack multiple timepoints
        x_bio_hat  = torch.stack(x_bio_hat, dim=1)
        x_dx_hat  = torch.stack(x_dx_hat, dim=1)

        return x_dx_hat, x_bio_hat