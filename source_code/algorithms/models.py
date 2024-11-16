from re import S
from math import log
import numpy as np
import itertools
from gym.spaces import Box, Discrete
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch_geometric.nn import GCNConv

# from .base_util import batch_to_seq, init_layer, one_hot

def MLP(sizes, activation=nn.ReLU, output_activation=nn.Identity, have_last_branch=True, **kwargs):
    layers = []
    for j in range(len(sizes) - 1):
        if have_last_branch:
            act = activation if j < len(sizes) - 2 else output_activation
        else:
            act = activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)




class YyxMLPModel(nn.Module):  # 在更深入地重复造轮子之前，先浏览一下这个脚本里有没有能用的东西~
    def __init__(self, logger, obs_dim, p_args, multi_mlp=False):
        super().__init__()

        if multi_mlp:
            self.pos_mlp = MLP(sizes=(33 * 2, 128,))
        else:
            hd = 256
            self.fc = MLP(sizes=(obs_dim + 1, 256, hd))
            self.s1s_branch = MLP(sizes=(hd, obs_dim))
            self.rs_branch = MLP(sizes=(hd, 1))
        self.multi_mlp = multi_mlp

        self.logger = logger
        self.reward_coeff = p_args.reward_coeff

        self.MSE = nn.MSELoss(reduction='none')
        self.BCE = nn.BCELoss(reduction='none')

    def forward(self, s, a):
        input = torch.cat([s, a], dim=-1)
        h = self.fc(input)
        rs = self.rs_branch(h)
        s1s = self.s1s_branch(h)
        return rs, s1s

    def train(self, s, a, r, s1, length=1):
        '''和GCN的train函数完全一致，只是不预测无意义的done了~'''
        pred_s, pred_r = [], []
        s0 = s.select(dim=1, index=0)
        length = min(length, s.shape[1])
        for t in range(length):
            r_, s_, = self.forward(s0, a.select(dim=1, index=t))
            pred_r.append(r_)
            pred_s.append(s_)
            s0 = s_
        reward_pred = torch.stack(pred_r, dim=1)
        state_pred = torch.stack(pred_s, dim=1)

        state_loss = self.MSE(state_pred, s1).mean()
        s1_view = s1.view(-1, s1.shape[-1])
        state_var = self.MSE(s1_view, s1_view.mean(dim=0, keepdim=True).expand(*s1_view.shape))
        rel_state_loss = state_loss / (state_var.mean() + 1e-7)
        self.logger.log(state_loss=state_loss, state_var=state_var.mean(), rel_state_loss=rel_state_loss)
        loss = state_loss

        reward_loss = self.MSE(reward_pred, r)
        loss += self.reward_coeff * reward_loss.mean()
        r_view = r.view(-1, r.shape[-1])
        reward_var = self.MSE(r_view, r_view.mean(dim=0, keepdim=True).expand(*r_view.shape)).mean()
        rel_reward_loss = reward_loss.mean() / (reward_var.mean() + 1e-7)

        self.logger.log(reward_loss=reward_loss,
                        reward_var=reward_var,
                        reward=r,
                        reward_norm=torch.norm(r),
                        rel_reward_loss=rel_reward_loss)

        return (loss, rel_state_loss)


class GraphConvolutionalModel(nn.Module):
    class EdgeNetwork(nn.Module):
        def __init__(self, i, j, sizes, activation=nn.ReLU, output_activation=nn.Identity):
            super().__init__()
            self.i = i  # 牛蛙，这种方式相当于给nn.Module额外存信息了，值得学习
            self.j = j
            layers = []
            for t in range(len(sizes) - 1):  # FC
                act = activation if t < len(sizes) - 2 else output_activation
                layers += [nn.Linear(sizes[t], sizes[t + 1]), act()]
            self.net = nn.Sequential(*layers)

        def forward(self, s: torch.Tensor):
            """  # 把边两端的结点的feature cat起来送入FC
            Input: [batch_size, n_agent, node_embed_dim] # raw input
            Output: [batch_size, edge_embed_dim]
            """
            s1 = s.select(dim=1, index=self.i)
            s2 = s.select(dim=1, index=self.j)
            s = torch.cat([s1, s2], dim=-1)
            return self.net(s)

    class NodeNetwork(nn.Module):
        def __init__(self, sizes, n_embedding=0, action_dim=0, activation=nn.ReLU, output_activation=nn.ReLU):
            super().__init__()
            layers = []
            for t in range(len(sizes) - 1):
                act = activation if t < len(sizes) - 2 else output_activation
                layers += [nn.Linear(sizes[t], sizes[t + 1]), act()]
            self.net = nn.Sequential(*layers)

            if n_embedding != 0:
                self.action_embedding_fn = nn.Embedding(action_dim, n_embedding)
                self.action_embedding = lambda x: self.action_embedding_fn(x.squeeze(-1))
            else:
                self.action_embedding = nn.Identity()

        def forward(self, h_ls, a):
            """
            Input: 
                h_ls: list of tensors with sizes of [batch_size, edge_embed_dim]
                a: [batch_size, action_dim]
            Output: 
                h: [batch_size, node_embed_dim]
            """

            # embedding = 0
            # hear 12 is edge_embed_dim
            embedding = torch.zeros([a.size()[0], 12], dtype=torch.float32, device=a.device)
            for h in h_ls:
                embedding += h
            a = self.action_embedding(a)
            while a.ndim < embedding.ndim:
                a = a.unsqueeze(-1)
            embedding = torch.cat([embedding, a], dim=-1)
            return self.net(embedding)

    class NodeWiseEmbedding(nn.Module):
        def __init__(self, n_agent, input_dim, output_dim, output_activation):
            super().__init__()
            self.nets = nn.ModuleList()
            self.n_agent = n_agent
            for _ in range(n_agent):
                self.nets.append(nn.Sequential(*[nn.Linear(input_dim, output_dim), output_activation()]))

        def forward(self, h):
            # input dim = 3, output the same
            items = []
            for i in range(self.n_agent):
                items.append(self.nets[i](h.select(dim=1, index=i)))
            items = torch.stack(items, dim=1)
            return items

    def __init__(self, logger, adj, state_dim, action_dim, n_agent, p_args):
        super().__init__()
        self.logger = logger.child("p")
        self.adj = adj > 0  # for default, 所有agent组成一条链，也即0-1-2-3-4-5-6-7
        # print('adj=',adj)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agent = n_agent
        self.n_conv = p_args.n_conv
        self.n_embedding = p_args.n_embedding
        self.residual = p_args.residual
        self.edge_embed_dim = p_args.edge_embed_dim
        self.edge_hidden_size = p_args.edge_hidden_size
        self.node_embed_dim = p_args.node_embed_dim
        self.node_hidden_size = p_args.node_hidden_size
        self.reward_coeff = p_args.reward_coeff

        self.node_nets = self._init_node_nets()
        self.edge_nets = self._init_edge_nets()
        self.node_embedding, self.state_head, self.reward_head, self.done_head = self._init_node_embedding()
        self.MSE = nn.MSELoss(reduction='none')
        self.BCE = nn.BCELoss(reduction='none')

    def predict(self, s, a):
        """
            输入s,a 输出
            Input: 
                s: [batch_size, n_agent, state_dim]
                a: [batch_size, n_agent, action_dim]
            Output: [batch_size, n_agent, state_dim] # same as input state
        """
        with torch.no_grad():
            r1, s1, d1 = self.forward(s, a)
            done = torch.clamp(d1, 0., 1.)
            done = torch.cat([1 - done, done], dim=-1)
            done = Categorical(done).sample() > 0  # [b]
            return r1, s1, done

    def train(self, s, a, r, s1, d, length=1):

        """
        Input shape: [batch_size, T_horizon, n_agent, dim]
        loss: 综合包含了预测的s' r done的损失。损失函数就是MSE~~
        rel_state_loss: 针对预测的s'的损失 除以标准差进行某种程度的归一化~
        """

        '''前向传播预测'''
        pred_s, pred_r, pred_d = [], [], []
        s0 = s.select(dim=1, index=0)  # 时序上最靠前的那个~
        length = min(length, s.shape[1])  # 就是T_horizon
        for t in range(length):
            # yyx 这里进行model-based中env model的前向传播
            # 输入是s和a 输出是预测的s' r done
            r_, s_, d_ = self.forward(s0, a.select(dim=1, index=t))
            pred_r.append(r_)
            pred_s.append(s_)
            pred_d.append(d_)
            s0 = s_
        reward_pred = torch.stack(pred_r, dim=1)
        state_pred = torch.stack(pred_s, dim=1)
        done_pred = torch.stack(pred_d, dim=1)

        state_loss = self.MSE(state_pred, s1).mean()
        s1_view = s1.view(-1, s1.shape[-1])  # shape = (48, 238)  # 其中48意为将前三个维度堆叠在一起
        state_var = self.MSE(s1_view, s1_view.mean(dim=0, keepdim=True).expand(*s1_view.shape))  # shape = (48, 238)
        rel_state_loss = state_loss / (state_var.mean() + 1e-7)
        # yyx: 这个state_var是什么意思？同理下面的reward_var
        self.logger.log(state_loss=state_loss, state_var=state_var.mean(), rel_state_loss=rel_state_loss)
        loss = state_loss

        reward_loss = self.MSE(reward_pred, r)
        loss += self.reward_coeff * reward_loss.mean()
        r_view = r.view(-1, r.shape[-1])
        reward_var = self.MSE(r_view, r_view.mean(dim=0, keepdim=True).expand(*r_view.shape)).mean()
        rel_reward_loss = reward_loss.mean() / (reward_var.mean() + 1e-7)

        self.logger.log(reward_loss=reward_loss,
                        reward_var=reward_var,
                        reward=r,
                        reward_norm=torch.norm(r),
                        rel_reward_loss=rel_reward_loss)

        d = d.float()
        done_loss = self.BCE(done_pred, d)
        loss = loss + done_loss.mean()
        done = done_pred > 0
        done_true_positive = (done * d).mean()
        d = d.mean()
        self.logger.log(done_loss=done_loss, done_true_positive=done_true_positive, done=d, rolling=100)

        return (loss, rel_state_loss)

    def forward(self, s, a):
        """
            Input: [batch_size, n_agent, state_dim]
            例如s.shape = [4, 3, 238]; a.shape = [4, 3, 238]
            Output: [batch_size, n_agent, state_dim]
        """
        embedding = self.node_embedding(s)  # dim = 3
        for _ in range(self.n_conv):
            edge_info_of_nodes = [[] for __ in range(self.n_agent)]  # 每个agent预先初始化一个列表
            for edge_net in self.edge_nets:
                edge_info = edge_net(embedding)  # dim = 2
                edge_info_of_nodes[edge_net.i].append(edge_info)
                edge_info_of_nodes[edge_net.j].append(edge_info)
            node_preds = []
            for i in range(self.n_agent):
                node_net = self.node_nets[i]
                node_pred = node_net(edge_info_of_nodes[i], a.select(dim=1, index=i))  # dim = 2
                node_preds.append(node_pred)
            embedding = torch.stack(node_preds, dim=1)  # dim = 3
        state_pred = self.state_head(embedding)
        if self.residual:
            state_pred += s
        reward_pred = self.reward_head(embedding)
        done_pred = self.done_head(embedding)
        return reward_pred, state_pred, done_pred

    def _init_node_nets(self):
        node_nets = nn.ModuleList()
        action_dim = self.n_embedding if self.n_embedding > 0 else self.action_dim
        sizes = [self.edge_embed_dim + action_dim] + self.node_hidden_size + [self.node_embed_dim]
        for i in range(self.n_agent):  # 每个agent存在一个node_net
            node_nets.append(GraphConvolutionalModel.NodeNetwork(sizes=sizes, n_embedding=self.n_embedding, action_dim=self.action_dim))
        return node_nets

    def _init_edge_nets(self):
        edge_nets = nn.ModuleList()
        sizes = [self.node_embed_dim * 2] + self.edge_hidden_size + [self.edge_embed_dim]
        # print('adj=',self.adj)
        for i in range(self.n_agent):  # 连通的两个agent之间存在一个node_net
            for j in range(i + 1, self.n_agent):
                if self.adj[i][j]:
                    edge_nets.append(GraphConvolutionalModel.EdgeNetwork(i, j, sizes))
        return edge_nets

    def _init_node_embedding(self):
        node_embedding = GraphConvolutionalModel.NodeWiseEmbedding(self.n_agent, self.state_dim, self.node_embed_dim, output_activation=nn.ReLU)
        state_head = GraphConvolutionalModel.NodeWiseEmbedding(self.n_agent, self.node_embed_dim, self.state_dim, output_activation=nn.Identity)
        reward_head = GraphConvolutionalModel.NodeWiseEmbedding(self.n_agent, self.node_embed_dim, 1, nn.Identity)
        done_head = GraphConvolutionalModel.NodeWiseEmbedding(self.n_agent, self.node_embed_dim, 1, nn.Sigmoid)
        return node_embedding, state_head, reward_head, done_head

class MultiCategoricalActor(nn.Module):
    """ 
    always returns a distribution
    """

    def __init__(self, **net_args):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        net_fn = net_args['network']  # MLP

        self.network = net_fn(**net_args)
        self.use_lambda = net_args['use_lambda']
        self.task_emb_dim = net_args['task_emb_dim']
        self.use_mate = net_args['use_mate']

        self.use_gcn = net_args['use_gcn']
        self.gcn_emb_dim = net_args['gcn_emb_dim']
        self.gcn_nodes_num = net_args['gcn_nodes_num']
        self.n_embd = net_args['n_embd']
        self.n_branch = net_args['n_branch']

        extra_features = 0
        if self.use_lambda:
            extra_features += 1
        if self.use_mate:
            extra_features += self.task_emb_dim*2
        if self.use_gcn:
            extra_features += self.n_embd

        if self.use_gcn:
            # self.gcn1 = GCNConv(3,3)
            # self.gcn2 = GCNConv(3,self.gcn_emb_dim)
            # self.gcn3 = nn.Linear(self.gcn_emb_dim*self.gcn_nodes_num,self.n_embd)
            self.gcn_model = net_args['gcn_model']


        #self.branch1 = nn.Linear(net_args['sizes'][-1]+extra_features,)
        self.branchs = nn.ModuleList([nn.Sequential(
                                        nn.Linear(net_args['sizes'][-1]+extra_features, net_args['sizes'][-1]),
                                        nn.ReLU(),
                                        nn.Linear(net_args['sizes'][-1],  net_args['poi_branchs'][net_args['type']].nvec[i]),) 
                                      for i in range(self.n_branch)])



        #self.branch2 = nn.Linear(net_args['sizes'][-1] , net_args['branchs'][1])
        self.eps = 1e-5


    def forward(self, obs, lambda_=None,task_emb=None,graph_inputs=None):  # 多维度动作确认OK
        # obs [B,S]  #
        # if self.snrmap_features > 0:
        #     # x = self.network(obs[:, 0:-self.snrmap_features])
        #     # snrmap = obs[:, -self.snrmap_features:]
        #     # embed = self.snr_network(torch.cat([x, snrmap], dim=-1))
        # else:
        #     assert self.snrmap_features == 0
        #     embed = self.network(obs)

        embed = self.network(obs)

        if self.use_lambda:
            embed = torch.cat([embed,lambda_],dim=1)

        if self.use_gcn:
            gcn_emb = self.gcn_model.forward(graph_inputs)
            embed = torch.cat([embed,gcn_emb],dim=1)

        probs = []
        for i in range(self.n_branch):
            logit = self.branchs[i](embed)
            # logits = torch.stack([logit1, logit2], dim=1)
            prob = self.softmax(logit) + self.eps
            prob = prob / prob.sum(dim=-1, keepdim=True)
            probs.append(prob)

        # prob2 = self.softmax(logit2) + self.eps
        # prob2 = prob2 / prob2.sum(dim=-1, keepdim=True)
        return torch.cat(probs,dim=-1)  # shape = (-1, act_dim1+act_dim2)



class CategoricalActor(nn.Module):
    """ 
    always returns a distribution
    """

    def __init__(self, **net_args):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        net_fn = net_args['network']  # MLP

        self.network = net_fn(**net_args)
        self.use_lambda = net_args['use_lambda']
        self.task_emb_dim = net_args['task_emb_dim']
        self.use_mate = net_args['use_mate']

        self.use_gcn = net_args['use_gcn']
        self.gcn_emb_dim = net_args['gcn_emb_dim']
        self.gcn_nodes_num = net_args['gcn_nodes_num']
        self.n_embd = net_args['n_embd']

        extra_features = 0
        if self.use_lambda:
            extra_features += 1
        if self.use_mate:
            extra_features += self.task_emb_dim*2
        if self.use_gcn:
            extra_features += self.n_embd

        if self.use_gcn:
            # self.gcn1 = GCNConv(3,3)
            # self.gcn2 = GCNConv(3,self.gcn_emb_dim)
            # self.gcn3 = nn.Linear(self.gcn_emb_dim*self.gcn_nodes_num,self.n_embd)
            self.gcn_model = net_args['gcn_model']


        #self.branch1 = nn.Linear(net_args['sizes'][-1]+extra_features,)
        self.branch1 = nn.Sequential(
            nn.Linear(net_args['sizes'][-1]+extra_features, net_args['sizes'][-1]),
            nn.ReLU(),
            nn.Linear(net_args['sizes'][-1],  net_args['branchs'][net_args['type']].n),
        )


        #self.branch2 = nn.Linear(net_args['sizes'][-1] , net_args['branchs'][1])
        self.eps = 1e-5


    def forward(self, obs, lambda_=None,task_emb=None,graph_inputs=None):  # 多维度动作确认OK
        # obs [B,S]  #
        # if self.snrmap_features > 0:
        #     # x = self.network(obs[:, 0:-self.snrmap_features])
        #     # snrmap = obs[:, -self.snrmap_features:]
        #     # embed = self.snr_network(torch.cat([x, snrmap], dim=-1))
        # else:
        #     assert self.snrmap_features == 0
        #     embed = self.network(obs)

        embed = self.network(obs)

        if self.use_lambda:
            embed = torch.cat([embed,lambda_],dim=1)

        if self.use_gcn:
            gcn_emb = self.gcn_model.forward(graph_inputs)
            embed = torch.cat([embed,gcn_emb],dim=1)

        if self.use_mate:
            embed = torch.cat([embed,task_emb],dim=1)

        logit1 = self.branch1(embed)
        # logits = torch.stack([logit1, logit2], dim=1)
        prob1 = self.softmax(logit1) + self.eps
        prob1 = prob1 / prob1.sum(dim=-1, keepdim=True)

        # prob2 = self.softmax(logit2) + self.eps
        # prob2 = prob2 / prob2.sum(dim=-1, keepdim=True)
        return prob1  # shape = (-1, act_dim1+act_dim2)


    def gcn_forward(self,graph_inputs):
        AssertionError
        nodes, edges = graph_inputs
        gcn_emb = self.gcn1(nodes,edges[0].to(torch.int64))
        gcn_emb =  F.leaky_relu(gcn_emb, negative_slope=0.1)
        gcn_emb = self.gcn2(gcn_emb,edges[0].to(torch.int64))
        gcn_emb = gcn_emb.view(-1,self.gcn_emb_dim*self.gcn_nodes_num)
        gcn_emb = self.gcn3(gcn_emb).view(-1,self.n_embd)

        return gcn_emb



class Critic(nn.Module):

    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity, have_last_branch=True, **kwargs):
        super().__init__()
        self.use_lambda = kwargs['use_lambda']
        self.task_emb_dim = kwargs['task_emb_dim']
        self.use_mate = kwargs['use_mate']
        self.use_gcn = kwargs['use_gcn']
        self.gcn_emb_dim = kwargs['gcn_emb_dim']
        self.gcn_nodes_num = kwargs['gcn_nodes_num']

        self.n_embd = kwargs['n_embd']

        extra_features = 0
        if self.use_lambda:
            extra_features += 1
        if self.use_mate:
            extra_features += self.task_emb_dim*2
        if self.use_gcn:
            extra_features+= self.n_embd

        if self.use_gcn:
            self.gcn_model = kwargs['gcn_model']
            # self.gcn1 = GCNConv(3,3)
            # self.gcn2 = GCNConv(3,self.gcn_emb_dim)
            # self.gcn3 = nn.Linear(self.gcn_emb_dim*self.gcn_nodes_num,self.n_embd)

        layers = []
        for j in range(len(sizes) - 2):
            act = activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]

        self.critic = nn.Sequential(*layers)

        self.output = nn.Sequential(
            nn.Linear(sizes[-2]+extra_features, sizes[-2]),
            activation(),
            nn.Linear(sizes[-2], sizes[-1]),
            output_activation()
        )

    def forward(self,obs,lambda_=None,task_emb=None,graph_inputs=None):

        obs = self.critic(obs)

        if self.use_lambda:
            obs = torch.cat([obs,lambda_],dim=1)

        if self.use_gcn:
            gcn_emb = self.gcn_model.forward(graph_inputs)
            obs = torch.cat([obs, gcn_emb],dim=1)

        if self.use_mate:
            obs = torch.cat([obs,task_emb],dim=1)

        result = self.output(obs)

        return result


    def gcn_forward(self,graph_inputs):
        AssertionError
        nodes, edges = graph_inputs
        gcn_emb = self.gcn1(nodes,edges[0].to(torch.int64))
        gcn_emb =  F.leaky_relu(gcn_emb, negative_slope=0.1)
        gcn_emb = self.gcn2(gcn_emb,edges[0].to(torch.int64))
        gcn_emb = gcn_emb.view(-1,self.gcn_emb_dim*self.gcn_nodes_num)
        gcn_emb = self.gcn3(gcn_emb).view(-1,self.n_embd)

        return gcn_emb



class GCN(nn.Module):
    def __init__(self, gcn_emb_dim, gcn_nodes_num, n_embd,device):
        super(GCN, self).__init__()
        self.gcn_emb_dim = gcn_emb_dim
        self.gcn_nodes_num = gcn_nodes_num
        self.n_embd = n_embd
        self.gcn1 = GCNConv(3,3)
        self.gcn2 = GCNConv(3,self.gcn_emb_dim)
        self.gcn3 = nn.Linear(self.gcn_emb_dim*self.gcn_nodes_num,n_embd)
        self.to(device=device)

    def forward(self, graph_inputs, ma_dim =False):
        nodes, edges = graph_inputs
        if ma_dim:
            B,N,_,_ = nodes.shape
            nodes = nodes.view(-1, *nodes.shape[2:])
            edges = edges.view(-1, *edges.shape[2:])
            gcn_emb = self.gcn1(nodes,edges[0].to(torch.int64))
            gcn_emb =  F.leaky_relu(gcn_emb, negative_slope=0.1)
            gcn_emb = self.gcn2(gcn_emb,edges[0].to(torch.int64))
            gcn_emb = gcn_emb.view(-1,self.gcn_emb_dim*self.gcn_nodes_num)
            gcn_emb = self.gcn3(gcn_emb).view(B,N,self.n_embd)
        else:
            gcn_emb = self.gcn1(nodes,edges[0].to(torch.int64))
            gcn_emb =  F.leaky_relu(gcn_emb, negative_slope=0.1)
            gcn_emb = self.gcn2(gcn_emb,edges[0].to(torch.int64))
            gcn_emb = gcn_emb.view(-1,self.gcn_emb_dim*self.gcn_nodes_num)
            gcn_emb = self.gcn3(gcn_emb).view(-1,self.n_embd)
        return gcn_emb

    def forward_with_origin_output(self, graph_inputs, ma_dim =False):
        nodes, edges = graph_inputs
        if ma_dim:
            B,N,_,_ = nodes.shape
            nodes = nodes.view(-1, *nodes.shape[2:])
            edges = edges.view(-1, *edges.shape[2:])
            gcn_emb = self.gcn1(nodes,edges[0].to(torch.int64))
            gcn_emb =  F.leaky_relu(gcn_emb, negative_slope=0.1)
            gcn_emb = self.gcn2(gcn_emb,edges[0].to(torch.int64)).view(-1,N,self.gcn_emb_dim,self.gcn_nodes_num)
        else:
            gcn_emb = self.gcn1(nodes,edges[0].to(torch.int64))
            gcn_emb =  F.leaky_relu(gcn_emb, negative_slope=0.1)
            gcn_emb = self.gcn2(gcn_emb,edges[0].to(torch.int64)).view(-1,self.gcn_emb_dim,self.gcn_nodes_num)
        return gcn_emb