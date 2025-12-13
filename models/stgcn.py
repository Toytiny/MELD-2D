import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.gcn_utils import mstcn, unit_gcn, unit_tcn
import copy as cp
import copy
from utils.graph import Graph
# from mmcv.cnn import constant_init, kaiming_init
# from mmcv.utils import _BatchNorm

EPS = 1e-4


class STGCNBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 **kwargs):
        super().__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in [
            'gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels,
                                9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels,
                             stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(
                in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


class STGCN(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=1,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32,
                         requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()


        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels,
                                  A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels *
                               self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels,
                           A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained
        
    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])
        return x


class STGCN_2(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=1,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32,
                         requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels,
                                  A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels *
                               self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels,
                                      A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        # for i in range(self.num_stages):
        #     x = self.gcn[i](x)
        #print (x.shape)

        stage_1_out = self.gcn[0](x)
        stage_2_out = self.gcn[1](stage_1_out)
        stage_3_out = self.gcn[2](stage_2_out)
        stage_4_out = self.gcn[3](stage_3_out)
        stage_5_out = self.gcn[4](stage_4_out)
        stage_6_out = self.gcn[5](stage_5_out)
        stage_7_out = self.gcn[6](stage_6_out)
        stage_8_out = self.gcn[7](stage_7_out)
        stage_9_out = self.gcn[8](stage_8_out)
        stage_10_out = self.gcn[9](stage_9_out)
        # print(stage_1_out.shape)
        # print(stage_2_out.shape)
        # print(stage_3_out.shape)
        # print(stage_4_out.shape)
        # print(stage_5_out.shape)
        # print(stage_6_out.shape)
        # print(stage_7_out.shape)
        # print(stage_8_out.shape)
        # print(stage_9_out.shape)
        # print(stage_10_out.shape)
        # print (stop)
        x = stage_10_out.reshape((N, M) + stage_10_out.shape[1:])
        #print(x.shape)
        #print (stop)

        # x = x.reshape((N, M) + x.shape[1:])
        return stage_1_out, stage_2_out, stage_3_out, stage_4_out, stage_5_out, stage_6_out, stage_7_out, stage_8_out,stage_9_out, stage_10_out, x



class STGCN_SSF(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=1,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.num_vertex = A.size(1)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs
        self.use_new_ssf = False

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

        # Initialize SSF parameters
        # self.ssf_scales = nn.ParameterList()
        # self.ssf_shifts = nn.ParameterList()
        # for i in range(self.num_stages):
        #     out_channels = self.gcn[i].tcn.out_channels
        #     scale, shift = self.init_ssf_scale_shift(out_channels)
        #     self.ssf_scales.append(scale)
        #     self.ssf_shifts.append(shift)

        self.ssf_scales = nn.ParameterList()
        self.ssf_shifts = nn.ParameterList()
        for i in range(self.num_stages):
            out_channels = self.gcn[i].tcn.out_channels
            scale, shift = self.init_ssf_scale_shift(out_channels, self.num_vertex)
            self.ssf_scales.append(scale)
            self.ssf_shifts.append(shift)

    def reinit_ssf(self):
        del self.ssf_scales
        del self.ssf_shifts
        self.ssf_scales = nn.ParameterList()
        self.ssf_shifts = nn.ParameterList()
        for i in range(self.num_stages):
            out_channels = self.gcn[i].tcn.out_channels
            scale, shift = self.init_ssf_scale_shift(out_channels, self.num_vertex)
            self.ssf_scales.append(scale)
            self.ssf_shifts.append(shift)

    def add_new_ssf(self):
        self.use_new_ssf = True
        self.ssf_scales_new = nn.ParameterList()
        self.ssf_shifts_new = nn.ParameterList()
        for i in range(self.num_stages):
            out_channels = self.gcn[i].tcn.out_channels
            scale, shift = self.init_ssf_scale_shift(out_channels, self.num_vertex)
            self.ssf_scales_new.append(scale)
            self.ssf_shifts_new.append(shift)

            

    # def init_ssf_scale_shift(self, dim):
    #     scale = nn.Parameter(torch.ones(dim))
    #     shift = nn.Parameter(torch.zeros(dim))
    #     nn.init.normal_(scale, mean=1, std=.02)
    #     nn.init.normal_(shift, std=.02)
    #     return scale, shift
    #
    # def ssf_ada(self, x, scale, shift):
    #     return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)

    def init_ssf_scale_shift(self, out_channels, num_vertex):
        # Scale/shift per channel and vertex
        scale = nn.Parameter(torch.ones(out_channels, num_vertex))
        shift = nn.Parameter(torch.zeros(out_channels, num_vertex))
        nn.init.normal_(scale, mean=1, std=.02)
        nn.init.normal_(shift, std=.02)
        return scale, shift

    def ssf_ada(self, x, scale, shift):
        # x shape: (N*M, C, T, V)
        # scale/shift shape: (C, V)
        # Need to explicitly specify the dimensions
        return x * scale.view(1, scale.size(0), 1, scale.size(1)) + shift.view(1, shift.size(0), 1, shift.size(1))
    
    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        outputs = []
        for i in range(self.num_stages):
            x = self.gcn[i](x)
            # parallel -------------
            # if self.use_new_ssf:
            #     x1 = self.ssf_ada(x, self.ssf_scales[i], self.ssf_shifts[i])
            #     x2 = self.ssf_ada(x, self.ssf_scales_new[i], self.ssf_shifts_new[i])
            #     x = x1 + x2

            # serial --------------

            x = self.ssf_ada(x, self.ssf_scales[i], self.ssf_shifts[i])
            if self.use_new_ssf:
                x = self.ssf_ada(x, self.ssf_scales_new[i], self.ssf_shifts_new[i])
            outputs.append(x)

        x = x.reshape((N, M) + x.shape[1:])
        return outputs + [x]

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             kaiming_init(m)
    #         elif isinstance(m, (_BatchNorm, nn.BatchNorm2d)):
    #             constant_init(m, 1)


class STGCN_SSF_CL(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=1,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.num_vertex = A.size(1)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs
        self.use_new_ssf = True


        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages


        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained


        self.ssf_scale_list = nn.ModuleList()
        self.ssf_shift_list = nn.ModuleList()

        self.ssf_scales = nn.ParameterList()
        self.ssf_shifts = nn.ParameterList()
        for i in range(self.num_stages):
            out_channels = self.gcn[i].tcn.out_channels
            scale, shift = self.init_ssf_scale_shift(out_channels, self.num_vertex)
            self.ssf_scales.append(scale)
            self.ssf_shifts.append(shift)
            
        # Initialize SSF parameters
        # self.ssf_scales = nn.ParameterList()
        # self.ssf_shifts = nn.ParameterList()
        # for i in range(self.num_stages):
        #     out_channels = self.gcn[i].tcn.out_channels
        #     scale, shift = self.init_ssf_scale_shift(out_channels)
        #     self.ssf_scales.append(scale)
        #     self.ssf_shifts.append(shift)

        self.ssf_scales = nn.ParameterList()
        self.ssf_shifts = nn.ParameterList()
        for i in range(self.num_stages):
            out_channels = self.gcn[i].tcn.out_channels
            scale, shift = self.init_ssf_scale_shift(out_channels, self.num_vertex)
            self.ssf_scales.append(scale)
            self.ssf_shifts.append(shift)

    def reinit_ssf(self):
        del self.ssf_scales
        del self.ssf_shifts
        self.ssf_scales = nn.ParameterList()
        self.ssf_shifts = nn.ParameterList()
        for i in range(self.num_stages):
            out_channels = self.gcn[i].tcn.out_channels
            scale, shift = self.init_ssf_scale_shift(out_channels, self.num_vertex)
            self.ssf_scales.append(self.ssf_scales)
            self.ssf_shifts.append(self.ssf_shifts)

    def add_new_ssf(self):
        #copy.deepcopy(self.cur_adapter).requires_grad_(False)

        self.ssf_scale_list.append(copy.deepcopy(self.ssf_scales).requires_grad_(False))
        self.ssf_shift_list.append(copy.deepcopy(self.ssf_shifts).requires_grad_(False))
        del self.ssf_scales
        del self.ssf_shifts

        self.ssf_scales = nn.ParameterList()
        self.ssf_shifts = nn.ParameterList()
        for i in range(self.num_stages):
            out_channels = self.gcn[i].tcn.out_channels
            scale, shift = self.init_ssf_scale_shift(out_channels, self.num_vertex)
            self.ssf_scales.append(scale)
            self.ssf_shifts.append(shift)

    # def init_ssf_scale_shift(self, dim):
    #     scale = nn.Parameter(torch.ones(dim))
    #     shift = nn.Parameter(torch.zeros(dim))
    #     nn.init.normal_(scale, mean=1, std=.02)
    #     nn.init.normal_(shift, std=.02)
    #     return scale, shift
    #
    # def ssf_ada(self, x, scale, shift):
    #     return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)

    def init_ssf_scale_shift(self, out_channels, num_vertex):
        # Scale/shift per channel and vertex
        scale = nn.Parameter(torch.ones(out_channels, num_vertex))
        shift = nn.Parameter(torch.zeros(out_channels, num_vertex))
        nn.init.normal_(scale, mean=1, std=.02)
        nn.init.normal_(shift, std=.02)
        return scale, shift

    def ssf_ada(self, x, scale, shift):
        # x shape: (N*M, C, T, V)
        # scale/shift shape: (C, V)
        # Need to explicitly specify the dimensions
        return x * scale.view(1, scale.size(0), 1, scale.size(1)) + shift.view(1, shift.size(0), 1, shift.size(1))

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        outputs = []
        for i in range(self.num_stages):
            
            x_init = self.gcn[i](x)

            # parallel --------------
            x = x_init + self.ssf_ada(x_init, self.ssf_scales[i], self.ssf_shifts[i])
            if len(self.ssf_shift_list) > 0:
                for t in range (len(self.ssf_shift_list)):
                    x += self.ssf_ada(x_init, self.ssf_scale_list[t][i], self.ssf_shift_list[t][i])
            #x += self.ssf_ada(x, self.ssf_scales[i], self.ssf_shifts[i])
            outputs.append(x)

        x = x.reshape((N, M) + x.shape[1:])
        return outputs + [x]
