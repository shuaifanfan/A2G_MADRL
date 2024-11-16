# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 01:56:13 2022

@author: 86153
"""

import time
import os
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from source_code.algorithms.utils import collect, mem_report

from tqdm.std import trange
# from algorithms.algorithm import ReplayBuffer
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import pickle
from copy import deepcopy as dp

import random
import multiprocessing as mp
# import torch.multiprocessing as mp
from torch import distributed as dist
import argparse

adjacent_name_list = ['uav-carrier', 'uav-poi', 'carrier-uav', 'carrier-poi', 'carrier-road', 'poi-uav', 'poi-carrier',
                      'road-carrier', 'uav-epoi', 'uav-road', 'carrier-epoi']
simple_adjacent_name_list = ['uav-carrier', 'uav-poi', 'carrier-uav', 'carrier-poi', 'poi-uav', 'poi-carrier','poi','uav','carrier','id']
normal_feature_list = ["s", "share_s", "a", "r", "share_s1", "s1", "d", "a_mask", "logp", 'h', 'ae_h', 't_emb', 'a0',
                       'r0', 'cost', 'nodes', 'edges', 'cols', 'memory']

class MultiCollect:
    def __init__(self, adjacency, device='cuda'):
        """
        Method: 'gather', 'reduce_mean', 'reduce_sum'.
        Adjacency: torch Tensor.
        Everything outward would be in the same device specifed in the initialization parameter.
        """
        self.device = device
        n = adjacency.size()[0]
        adjacency = adjacency > 0  # Adjacency Matrix, with size n_agent*n_agent.
        adjacency = adjacency | torch.eye(n,
                                          device=device).bool()  # Should contain self-loop, because an agent should utilize its own info.
        adjacency = adjacency.to(device)
        # print('a=',adjacency)
        self.degree = adjacency.sum(dim=1)  # Number of information available to the agent.
        self.indices = []
        index_full = torch.arange(n, device=device)
        for i in range(n):
            self.indices.append(torch.masked_select(index_full, adjacency[i]))  # Which agents are needed.
        

    def gather(self, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape: [[batch_size, dim_i] for i in range(n_agent)]
        """
        return self._collect('gather', tensor)

    def reduce_mean(self, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape: [[batch_size, dim] for i in range(n_agent)]
        """
        return self._collect('reduce_mean', tensor)

    def reduce_sum(self, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape: [[batch_size, dim] for i in range(n_agent)]
        """
        return self._collect('reduce_sum', tensor)

    def _collect(self, method, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape: 
            gather: [[batch_size, dim_i] for i in range(n_agent)]
            reduce: [batch_size, n_agent, dim]  # same as input
        """
        tensor = tensor.to(self.device)
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(-1)
        b, n, depth = tensor.shape
        result = []
        for i in range(n):
            if method == 'gather':
                # self.indices[i]是agent i的邻居列表
                # 例如当agent组成链时，self.indices[1] = [0, 1, 2]
                result.append(torch.index_select(tensor, dim=1, index=self.indices[i]).view(b, -1))
            elif method == 'reduce_mean':
                result.append(torch.index_select(tensor, dim=1, index=self.indices[i]).mean(dim=1))
            else:
                result.append(torch.index_select(tensor, dim=1, index=self.indices[i]).sum(dim=1))
        if method != 'gather':
            result = torch.stack(result, dim=1)
        return result

class Trajectory:
    def __init__(self, **kwargs):
        """
        Data are of size [T, n_agent, dim].
        """
        self.names = normal_feature_list + adjacent_name_list
        self.dict = {name: kwargs[name] for name in kwargs.keys()}
        self.length = self.dict["s"].size()[0]
          
    def getFraction(self, length, start=None):
        # print('l1=',length)
        # print('l2=',self.length)
        if self.length < length:
            length = self.length
        start_max = self.length - length
        if start is None:
            start = torch.randint(low=0, high=start_max + 1, size=(1,)).item()

        start = min(max(start, 0), start_max) 
        
        # if start > start_max:
        #     start = start_max
        # if start < 0:
        #     start = 0

        new_dict = {name: self.dict[name][start:start + length] for name in self.names}
        return Trajectory(**new_dict)

    def __getitem__(self, key):
        assert key in self.names
        return self.dict[key]

    @classmethod
    def names(cls):
        return normal_feature_list + adjacent_name_list.copy()
    

class TrajectoryBuffer:
    def __init__(self, alg_args, device="cuda"):
        self.device = device
        self.type = alg_args.agent_args.agent_type
        self.poi_decision_mode = alg_args.poi_decision_mode
        if self.poi_decision_mode:
            self.type += ['poi_carrier','poi_uav']
        self.n_agent = alg_args.agent_args.n_agent
        self.use_lambda = alg_args.agent_args.use_lambda
        self.use_mate = alg_args.agent_args.use_mate
        self.use_gcn = alg_args.agent_args.use_gcn
        self.gcn_nodes_num = alg_args.agent_args.gcn_nodes_num
        self.use_hgcn = alg_args.agent_args.use_hgcn
        self.random_permutation = alg_args.agent_args.random_permutation
        self.use_temporal_type = alg_args.agent_args.use_temporal_type
        self.lambda_range = [0, 1]
        self.lambda_num = 10

        if self.use_lambda:
            self.lambda_ = {key: [] for key in self.type}
        self.share_s, self.share_s1 = {key: [] for key in self.type}, {key: [] for key in self.type}
        self.a_mask = {key: [] for key in self.type}
        self.s, self.a, self.r, self.s1, self.d, self.logp = {key: [] for key in self.type}, {key: [] for key in
                                                                                              self.type}, {key: [] for
                                                                                                           key in
                                                                                                           self.type}, {
            key: [] for key in self.type}, {key: [] for key in self.type}, {key: [] for key in self.type}
        self.h, self.ae_h, self.t_emb = {key: [] for key in self.type}, {key: [] for key in self.type}, {key: [] for key
                                                                                                         in self.type}
        self.a0, self.r0 = {key: [] for key in self.type}, {key: [] for key in self.type}
        self.cost = {key: [] for key in self.type}
        self.nodes, self.edges = {key: [] for key in self.type}, {key: [] for key in self.type}
        self.cols, self.memory = {key: [] for key in self.type}, {key: [] for key in self.type}
        self.step = []
        for item in adjacent_name_list:
            setattr(self, item, {key: [] for key in self.type})
        # self.s, self.a, self.r, self.s1, self.d, self.logp = [], [], [], [], [], []

    def store(self, state, observations, action, reward, state_1, observations_1, done, action_mask, logprob,
              hidden=None, ae_hidden=None, task_embedding=None, action_0=None, reward_0=None, cost=None, nodes=None,
              edges=None, adj=None, cols=None, memory=None, step=None):
        """
        Would be converted into [batch_size, n_agent, dim].
        """
        device = self.device
        self.step.append(step)
        for type in self.type:
            [share_s, s, r, share_s1, s1, a_mask, logp] = [torch.as_tensor(item, device=device, dtype=torch.float) for
                                                           item in [state, observations[type], reward[type], state_1,
                                                                    observations_1[type], action_mask[type],
                                                                    logprob[type]]]
            d = torch.as_tensor(done[type], device=device, dtype=torch.bool)
            a = torch.as_tensor(action[type], device=device)
            while s.dim() <= 2:
                s = s.unsqueeze(dim=0)  # 添加threads维度
            b, n, dim = s.size()
            # if d.dim() <= 1:
            #     d = d.unsqueeze(0)
            # d = d[:, :n]
            # if r.dim() <= 1:
            #     r = r.unsqueeze(0)
            # r = r[:, :n]
            share_s = share_s.view(b, -1).unsqueeze(1).repeat(1, n, 1)
            share_s1 = share_s1.view(b, -1).unsqueeze(1).repeat(1, n, 1)
            [s, a, r, s1, d, a_mask, logp] = [item.view(b, n, -1) for item in [s, a, r, s1, d, a_mask, logp]]
            self.share_s[type].append(share_s)
            self.s[type].append(s)
            self.a[type].append(a)
            self.r[type].append(r)
            self.share_s1[type].append(share_s1)
            self.s1[type].append(s1)
            self.d[type].append(d)
            self.logp[type].append(logp)   
            self.a_mask[type].append(a_mask)

            if self.use_temporal_type:
                self.memory[type].append(memory)

            if self.use_mate:
                ae_h, t_emb, a0, r0 = [torch.as_tensor(item, device=device, dtype=torch.float).view(b, n, -1) for item
                                       in [ae_hidden[type], task_embedding[type], action_0[type], reward_0[type]]]
                self.ae_h[type].append(ae_h)
                self.t_emb[type].append(t_emb)
                self.a0[type].append(a0)
                self.r0[type].append(r0)

            if self.use_gcn:
                # nos = torch.as_tensor(nodes[type], device=device, dtype=torch.float).view(b,n, self.gcn_nodes_num,3)
                # es = torch.as_tensor(edges[type], device=device, dtype=torch.float).view(b,n, 2, -1)
                # es = torch.as_tensor(edges, device=device, dtype=torch.float).view(b, self.gcn_nodes_num, self.gcn_nodes_num).unsqueeze(dim=1).repeat(1,n,1,1)
                nos = torch.as_tensor(nodes, device=device, dtype=torch.float).view(b, self.gcn_nodes_num, 3)
                es = torch.as_tensor(edges, device=device, dtype=torch.float).view(b, 2, -1)
                self.nodes[type].append(nos)
                self.edges[type].append(es)

            if self.use_hgcn:
                for key in adjacent_name_list:
                    # getattr(self,key)[type].append(torch.cat([torch.as_tensor(adj[key], device=device, dtype=torch.float).unsqueeze(1) for _ in range(self.n_agent[type])],dim=1))
                    getattr(self, key)[type].append(torch.as_tensor(adj[key], device=device, dtype=torch.float))
                    
            if self.random_permutation:
                cols = torch.as_tensor(cols, device=device, dtype=torch.float).view(b,-1)
                self.cols[type].append(cols)

    def retrieve(self):
        """
        Returns trajectories with s, a, r, s1, d, logp.
        Data are of size [T, n_thread, n_agent, dim]
        返回n_thread条traj
        """
        names = ["s", "share_s", "a", "r", "share_s1", "s1", "d", "a_mask", "logp"]
        if self.use_mate:
            names.extend(['ae_h', 't_emb', 'a0', 'r0'])
        if self.use_lambda:
            names.extend(['cost'])
        if self.use_gcn:
            names.extend(['nodes', 'edges'])
        if self.use_hgcn:
            names.extend(adjacent_name_list)
        if self.random_permutation:
            names.extend(['cols'])
        if self.use_temporal_type and self.use_hgcn:
            names.extend(['memory'])

        trajs_type = {}
        for type in self.type:
            trajs = []
            traj_all = {}
            if self.s[type] == []:
                trajs_type[type] = []
                continue
            for name in names:
                traj_all[name] = torch.stack(self.__getattribute__(name)[type],
                                             dim=1)  # stack后，shape = (n_thread, T, n_agent, dim)
            n = traj_all['s'].size()[0]
            for i in range(n):
                traj_dict = {}
                for name in names:
                    if names in adjacent_name_list:
                        traj_dict[name] = traj_all[name][i]
                    else:
                        traj_dict[name] = traj_all[name][i]
                trajs.append(Trajectory(**traj_dict))
            trajs_type[type] = trajs
        trajs_type['step'] = np.array(self.step)
        return trajs_type



class EpisodicTrajectory:
    def __init__(self, **kwargs):
        """
        Data are of size [T, n_agent, dim].
        """
        self.names = ["s", "a", "r", "s1", "d", "a_mask", "logp"] + simple_adjacent_name_list
        self.dict = {name: kwargs[name] for name in kwargs.keys()}
        self.length = self.dict["s"].size()[0]
          
    def getFraction(self, length, start=None):
        # print('l1=',length)
        # print('l2=',self.length)
        if self.length < length:
            length = self.length
        start_max = self.length - length
        if start is None:
            start = torch.randint(low=0, high=start_max + 1, size=(1,)).item()

        start = min(max(start, 0), start_max) 
        
        # if start > start_max:
        #     start = start_max
        # if start < 0:
        #     start = 0

        new_dict = {name: self.dict[name][start:start + length] for name in self.names}
        return EpisodicTrajectory(**new_dict)

    def __getitem__(self, key):
        assert key in self.names
        return self.dict[key]

    @classmethod
    def names(cls):
        return ["s", "a", "r", "s1", "d", "a_mask", "logp"] + simple_adjacent_name_list
    

class EpisodicTrajectoryBuffer:
    def __init__(self, alg_args, device="cuda"):
        self.device = device

        # self.poi_decision_mode = alg_args.poi_decision_mode
        # self.use_gcn = alg_args.agent_args.use_gcn
        # self.gcn_nodes_num = alg_args.agent_args.gcn_nodes_num
       
        self.s, self.a, self.r, self.s1, self.d, self.logp = [],[],[],[],[],[]
        self.a_mask = []
        self.nodes, self.edges = [],[]
        for item in simple_adjacent_name_list:
            setattr(self, item, [])
        # self.s, self.a, self.r, self.s1, self.d, self.logp = [], [], [], [], [], []

    def store(self, state_all, action, reward, state_1_all, done, logp):
        """
        Would be converted into [batch_size, n_agent, dim].
        """
        state = state_all['state']
        state_1 = state_1_all['state']
        action_mask = state_all['mask']
        
        device = self.device

        [s, r, s1, a_mask, logp] = [torch.as_tensor(item, device=device, dtype=torch.float) for
                                                        item in [state, reward, state_1,action_mask,logp]]
        d = torch.as_tensor(done, device=device, dtype=torch.bool)
        a = torch.as_tensor(action, device=device)
        while s.dim() <= 1:
            s = s.unsqueeze(dim=0)  # 添加threads维度
        b, dim = s.size()

        [s, a, r, s1, d, a_mask, logp] = [item.view(b, -1) for item in [s, a, r, s1, d, a_mask, logp]]
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s1.append(s1)
        self.d.append(d)
        self.logp.append(logp)   
        self.a_mask.append(a_mask)



        for key in simple_adjacent_name_list:
            # getattr(self,key)[type].append(torch.cat([torch.as_tensor(adj[key], device=device, dtype=torch.float).unsqueeze(1) for _ in range(self.n_agent[type])],dim=1))
            getattr(self, key).append(torch.as_tensor(state_all[key], device=device, dtype=torch.float))
                

    def retrieve(self):
        """
        Returns trajectories with s, a, r, s1, d, logp.
        Data are of size [T, n_thread, n_agent, dim]
        返回n_thread条traj
        """
        names = ["s", "a", "r", "s1", "d", "a_mask", "logp"]
        names.extend(simple_adjacent_name_list)
       
        trajs = []
        traj_all = {}

        for name in names:
            traj_all[name] = torch.stack(self.__getattribute__(name),
                                            dim=1)  # stack后，shape = (n_thread, T, n_agent, dim)
        # n = traj_all['s'].size()[0]
        # for i in range(n):
        #     traj_dict = {}
        #     for name in names:
        #         traj_dict[name] = traj_all[name][i]
        #     trajs.append(EpisodicTrajectory(**traj_dict))

        return traj_all

class ModelBuffer:
    def __init__(self, max_traj_num):
        self.max_traj_num = max_traj_num
        self.trajectories = []
        self.ptr = -1
        self.count = 0

    def storeTraj(self, traj):
        if self.count < self.max_traj_num:
            self.trajectories.append(traj)
            self.ptr = (self.ptr + 1) % self.max_traj_num
            self.count = min(self.count + 1, self.max_traj_num)
        else:
            self.trajectories[self.ptr] = traj
            self.ptr = (self.ptr + 1) % self.max_traj_num

    def storeTrajs(self, trajs):
        for traj in trajs:
            self.storeTraj(traj)

    def sampleTrajs(self, n_traj):
        traj_idxs = np.random.choice(range(self.count), size=(n_traj,), replace=True)
        return [self.trajectories[i] for i in traj_idxs]