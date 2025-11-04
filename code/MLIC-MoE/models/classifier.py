import json
import math
import torch
import torch.nn as nn

class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, dataname=None ,bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias
        
        # coocurrence_path ="/media/data2/maleilei/MLIC/OT-SSML/preproc/partitionv3/data/coco_coocurrence5_0.json"
        # cluster_path = "/media/data2/maleilei/MLIC/OT-SSML/preproc/partitionv3/data/coco_cluster5_0.json"
        
        if dataname == 'coco':
            coocurrence_path ="/media/data2/maleilei/MLIC/OT-SSML/preproc/partitionv3/data/coco_coocurrence5_0.json"
            cluster_path = "/media/data2/maleilei/MLIC/OT-SSML/preproc/partitionv3/data/coco_cluster5_0.json"
        elif dataname == 'voc2007':
            coocurrence_path ="/media/data2/maleilei/MLIC/OT-SSML/preproc/partitionv3/data/voc_coocurrence5_0.json"
            cluster_path = "/media/data2/maleilei/MLIC/OT-SSML/preproc/partitionv3/data/voc_cluster5_0.json"
        else:
            raise ValueError('dataset_name must be coco or voc')

        with open(coocurrence_path, 'r') as f:
            self.coocurrence = json.load(f)
        self.len_coocurrence = len(self.coocurrence)
        self.len_co_list = [len(self.coocurrence[i]) for i in range(self.len_coocurrence)]
        self.len_co_list_index = [0] + [(sum(self.len_co_list[:i+1])) for i in range(self.len_coocurrence)]

        
        with open(cluster_path, 'r') as f:
            self.cluster = json.load(f)
        self.len_cluster = len(self.cluster)
        self.len_cl_list = [len(self.cluster[i]) for i in range(self.len_cluster)]
        self.len_cl_list_index = [0] + [(sum(self.len_cl_list[:i+1])) for i in range(self.len_cluster)]

        # print(self.coocurrence)
        self.co_index = []
        for i in range(self.len_coocurrence):
            tmp = [j for j in range(len(self.coocurrence[i]))]
            for k, v, in self.coocurrence[i].items():
                tmp[v] = int(k)
            self.co_index += tmp
        self.co_index = torch.tensor(self.co_index)

        self.cl_index = []
        for i in range(self.len_cluster):
            tmp = [j for j in range(len(self.cluster[i]))]
            for k, v, in self.cluster[i].items():
                tmp[v] = int(k)
            self.cl_index += tmp
        self.cl_index = torch.tensor(self.cl_index)


        self.co_W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        self.cl_W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.co_b = nn.Parameter(torch.Tensor(1, num_class))
            self.cl_b = nn.Parameter(torch.Tensor(1, num_class))

        self.group_len = self.len_coocurrence
        self.set_device = True
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)

        for i in range(self.num_class):
            self.co_W[0][i].data.uniform_(-stdv, stdv)
            self.cl_W[0][i].data.uniform_(-stdv, stdv)

        if self.bias:
            for i in range(self.num_class):
                self.co_b[0][i].data.uniform_(-stdv, stdv)
                self.cl_b[0][i].data.uniform_(-stdv, stdv)

    def set_to_device(self, device):

        self.co_index = self.co_index.to(device)
        self.cl_index = self.cl_index.to(device)
        for i in range(self.len_coocurrence):
            self.co_W[i] = self.co_W[i].to(device)
            if self.bias:
                self.co_b[i] = self.co_b[i].to(device)
        for i in range(self.len_cluster):
            self.cl_W[i] = self.cl_W[i].to(device)
            if self.bias:
                self.cl_b[i] = self.cl_b[i].to(device)

    def forward(self, x):

        device = x.device

        co_x, cl_x = x[:, 0:self.group_len, :], x[:, self.group_len:, :]

        co_output, cl_output = [], []
        for i in range(self.len_coocurrence):
            co_wa = (self.co_W[:, self.len_co_list_index[i]:self.len_co_list_index[i+1], :] * co_x[:, i, :].unsqueeze(1)).sum(-1)
            if self.bias:
                co_b = self.co_b[:, self.len_co_list_index[i]:self.len_co_list_index[i+1] ]
                co_wa = co_wa + co_b
            co_output.append(co_wa)

        for i in range(self.len_cluster):
            cl_wa = (self.cl_W[:, self.len_cl_list_index[i]:self.len_cl_list_index[i+1], :] * cl_x[:, i, :].unsqueeze(1)).sum(-1)
            if self.bias:
                cl_b = self.cl_b[:, self.len_cl_list_index[i]:self.len_cl_list_index[i+1] ]
                cl_wa = cl_wa + cl_b
            cl_output.append(cl_wa)


        co_output = torch.cat(co_output, dim=1)
        cl_output = torch.cat(cl_output, dim=1)

        co_output = torch.index_select(co_output, 1, self.co_index.to(device))
        cl_output = torch.index_select(cl_output, 1, self.cl_index.to(device))

        return co_output, cl_output


if __name__ == '__main__':
    GroupWiseLinear(num_class=80, hidden_dim=384, bias=True)