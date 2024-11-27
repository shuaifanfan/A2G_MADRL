# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 02:04:27 2022

@author: 86153
"""

import time
import os
from algorithms.models import Critic, GCN
from source_code.algorithms.algo.agent_base import AgentBase
# from algorithms.algorithm import ReplayBuffer

from gym.spaces.box import Box
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
from algorithms.models import MultiCategoricalActor
from algorithms.algo.buffer import Trajectory,EpisodicTrajectory
from torch.distributions import Categorical, Distribution
from source_code.algorithms.model.hgcn import HeteGCNLayer
from torch.nn import functional as F
from typing import List

simple_adjacent_name_list = ['uav-carrier', 'uav-poi', 'carrier-uav', 'carrier-poi', 'poi-uav', 'poi-carrier','poi','uav','carrier','id']

class MultiCategorical(Distribution):

    def __init__(self, dists: List[Categorical]):
        super().__init__()
        self.dists = dists

    def log_prob(self, value):
        ans = []
        for d, v in zip(self.dists, torch.split(value, 1, dim=-1)):
            ans.append(d.log_prob(v.squeeze(-1)))
        return torch.stack(ans, dim=-1)

    def entropy(self):
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([d.sample(sample_shape) for d in self.dists], dim=-1)

class EpisodicAgent(nn.ModuleList, AgentBase):
    """Everything in and out is torch Tensor."""

    def __init__(self, logger, device, agent_args, input_args):
        super().__init__()
        self.input_args = input_args

        self.logger = logger  # LogClient类对象
        self.device = device
        self.gamma = agent_args.gamma
        self.lamda = agent_args.lamda
        self.clip = agent_args.clip
        self.target_kl = agent_args.target_kl
        self.v_coeff = agent_args.v_coeff
        self.v_thres = agent_args.v_thres
        self.entropy_coeff = agent_args.entropy_coeff
        self.lr = agent_args.lr  # 5e-5
        self.lr_v = agent_args.lr_v  # 5e-4
        self.n_update_v = agent_args.n_update_v
        self.n_update_pi = agent_args.n_update_pi
        self.n_minibatch = agent_args.n_minibatch
        self.use_reduced_v = agent_args.use_reduced_v
        self.batch_size = agent_args.batch_size
        self.use_temporal_type = agent_args.use_temporal_type
        self.use_rtg = agent_args.use_rtg
        self.use_gae_returns = agent_args.use_gae_returns
        self.env_name = input_args.env
        self.algo_name = input_args.algo
        self.advantage_norm = agent_args.advantage_norm
        
        self.decoupled_actor = input_args.decoupled_actor
        self.use_graph_feature = input_args.use_graph_feature
        self.observation_dim =  agent_args.observation_dim
        self.action_space = agent_args.action_space
        self.agent_type = 'Episodic_agent'
        self.n_agent = sum([agent_args.n_agent[key] for key in agent_args.agent_type])
        self.poi_num = agent_args.n_poi if not self.input_args.near_selection_mode else 14
        self.uav_num = agent_args.n_agent['uav']
        self.carrier_num = agent_args.n_agent['carrier']
        self.poi_feature_dim = 6 #图神经网络中，poi作为node的特征的维度是6，表示poi的观测值，分别是：[x,y,数据量,当前是否被选过，与当前agent距离，与当前agent的relay-car距离]
        self.uav_feature_dim = 4 + input_args.channel_num*4 #图神经网络中，agent（uav和car）作为node的特征的维度是4+channel_num*4，表示agent的观测值，分别是：[包括位置x，y，收集时间，是否是当前agent，收集历史(4*channel)]

        # self.action_dim = sum([dim.n for dim in self.action_space])
        self.pi_args = agent_args.pi_args
        self.v_args = agent_args.v_args
        self.agent_args = agent_args
        self.share_parameters = False
        
        self.grapher_parameters = []
        if self.use_graph_feature:
            self.n_embd = 32
            self.hgcn_number = input_args.hgcn_layer_num
            self._init_graph()
            self.grapher_parameters = []
            for i in range(self.hgcn_number):
                self.grapher_parameters.extend(list(self.grapher[i].parameters()))
         
        else:
            self.graph_output_size = 1
        
        self.actors = self._init_actors()  # collect_pi和collect_v应该一样吧？
        self.vs = self._init_vs()
    
            
        self.optimizer_v = Adam(list(set(list(self.vs.parameters())+self.grapher_parameters)), lr=self.lr_v)
        self.optimizer_pi = Adam(list(set(list(self.actors.parameters())+self.grapher_parameters)), lr=self.lr)


    def log_model(self):
        prefix = f"{self.agent_type}/"
        self.log_parameters(f"{prefix}/Critic_", self.vs.named_parameters())
        self.log_parameters(f"{prefix}/Actor_", self.actors.named_parameters())

    
    def log_parameters(self, prefix, n_params):
        log_dict = {}
        for p_name, param in n_params:
            p_name = prefix + p_name
            log_dict[p_name] = torch.norm(param).item()
            if param.grad is not None:
                log_dict[p_name + ".grad"] = torch.norm(param.grad).item()
        self.logger.log(**log_dict)


    def act(self, state,graph=None):
        """
        非向量环境：Requires input of [batch_size, n_agent, dim] or [n_agent, dim].
        向量环境：Requires input of [batch_size*n_thread, n_agent, dim] or [n_thread, n_agent, dim].
        其中第一维度的值在后续用-1表示
        This method is gradient-free. To get the gradient-enabled probability information, use get_logp().
        Returns a distribution with the same dimensions of input.
        """
        with torch.no_grad():
            action_mask = torch.as_tensor(state['mask'], dtype=torch.float32, device=self.device)
            s = torch.as_tensor(state['state'], dtype=torch.float32, device=self.device)
            assert s.dim() == 2
            if self.use_graph_feature:
                graph_feature,output_dict = self.hgcn_forward(state)
                s = graph_feature

            if self.use_graph_feature and self.decoupled_actor:#这个
                prob = self.actors(output_dict['poi'])
            else:
                prob = self.actors(s)
            prob[action_mask == 0] = 0
            probs = Categorical(prob)
            return probs

    def get_logp(self, state, action,feed_dict=None):
        """
        Requires input of [batch_size, n_agent, dim] or [n_agent, dim].
        Returns a tensor whose dim() == 3.
        """
      
        s = torch.as_tensor(state, dtype=torch.float32,
                            device=self.device) 
        a = action
        while s.dim() <= 1:
            s = s.unsqueeze(0)
            a = a.unsqueeze(0)
        while a.dim() < s.dim():
            a = a.unsqueeze(-1)
        
        if self.use_graph_feature:
            graph_feature,output_dict = self.hgcn_forward(feed_dict)
            s = graph_feature

        if self.use_graph_feature and self.decoupled_actor:
            probs = self.actors(output_dict['poi'])
        else:
            probs = self.actors(s)
            
        index = action
   
        log_prob = torch.log(torch.gather(probs, dim=-1, index=index))  # [320,2,1]
        while log_prob.dim() < 2:
            log_prob = log_prob.unsqueeze(-1)

        return log_prob 

    def _evalV(self, state, feed_dict=None):
        s = state.to(self.device)
        if self.use_graph_feature:
            graph_feature,output_dict = self.hgcn_forward(feed_dict)
            s = graph_feature
        value = self.vs(s)
        return value

    def updateAgent(self, trajs, clip=None):
        time_t = time.time()
        if clip is None:
            clip = self.clip
        n_minibatch = self.n_minibatch

        names = EpisodicTrajectory.names()  # ['s', 'a', 'r', 's1', 'd', 'logp']
      
        # traj_all = {name:[] for name in names}
        # for traj in trajs:
        #     for name in names:
        #         traj_all[name].append(traj[name])
        # # should be 4-dim [batch * step * n_agent * dim]
        # traj = {name:torch.stack(value, dim=0) for name, value in traj_all.items()}
      
        traj = trajs
        for i_update in range(1):
            s, a, r, s1, d, logp = traj['s'], traj['a'], traj['r'], traj['s1'], traj['d'], traj['logp']
            s, a, r, s1, d, logp = [item.to(self.device) for item in [s, a, r, s1, d, logp]]
            
            feed_dict = {
                key:traj[key]  for key in simple_adjacent_name_list
            }

            value_old, returns, advantages, _ = self._process_traj(traj['s'], traj['a'], traj['r'], traj['s1'],
                                                                    traj['d'], traj['logp'],feed_dict)  # 不同traj分开计算adv和return
            advantages_old = advantages

            b, T, d_s = s.size()
            d_a = a.size()[-1]
            s = s.view(-1, d_s)
            a = a.view(-1, d_a)
            logp = logp.view(-1, d_a)
            advantages_old = advantages_old.view(-1, 1)
            returns = returns.view(-1, 1)
            value_old = value_old.view(-1, 1)
            # 关键：s, a, logp, adv, ret, v are now all in shape [-1, n_agent, dim] 因为计算完adv和return后可以揉在一起做mini_batch训练
            batch_total = logp.size()[0]
            #batch_size = int(batch_total / n_minibatch)
            batch_size = self.batch_size
            

            if self.use_graph_feature:
                feed_dict['id'] = feed_dict['id'].view(b*T,1)
                feed_dict['poi'] = feed_dict['poi'].view(b*T,self.poi_num,self.poi_feature_dim)
                feed_dict['carrier'] = feed_dict['carrier'].view(b*T,self.carrier_num,self.uav_feature_dim)
                feed_dict['uav'] = feed_dict['uav'].view(b*T,self.uav_num,self.uav_feature_dim)
                feed_dict['uav-carrier'] = feed_dict['uav-carrier'].view(b*T,self.uav_num,self.carrier_num)
                feed_dict['uav-poi'] = feed_dict['uav-poi'].view(b*T,self.uav_num,self.poi_num)
                feed_dict['carrier-uav'] = feed_dict['carrier-uav'].view(b*T,self.carrier_num,self.uav_num)
                feed_dict['carrier-poi'] = feed_dict['carrier-poi'].view(b*T,self.carrier_num,self.poi_num)
                feed_dict['poi-uav'] = feed_dict['poi-uav'].view(b*T,self.poi_num,self.uav_num)
                feed_dict['poi-carrier'] = feed_dict['poi-carrier'].view(b*T,self.poi_num,self.carrier_num)
    
            
            
            kl_all = []
            batch_feed_dict = None
            for i_pi in range(self.n_update_pi):
                batch_state, batch_action, batch_logp, batch_advantages_old = [s, a, logp, advantages_old]
          
                if n_minibatch > 1:
                    idxs = np.random.choice(range(batch_total), size=batch_size, replace=False)
                    [batch_state, batch_action, batch_logp, batch_advantages_old] = [item[idxs] for item in
                                                                                        [batch_state, batch_action,
                                                                                        batch_logp,
                                                                                        batch_advantages_old]]
                    if self.use_graph_feature:
                        batch_feed_dict = {key:value[idxs] for key,value in feed_dict.items()}
                    
                
                batch_logp_new = self.get_logp(batch_state, batch_action,feed_dict=batch_feed_dict)

                logp_diff = batch_logp_new.sum(-1, keepdim=True) - batch_logp.sum(-1, keepdim=True)
                kl = logp_diff.mean()  # 这里魔改的，不一定对
                ratio = torch.exp(logp_diff)
                surr1 = ratio * batch_advantages_old
                surr2 = ratio.clamp(1 - clip, 1 + clip) * batch_advantages_old
                loss_surr = torch.min(surr1, surr2).mean()
                loss_entropy = - torch.mean(batch_logp_new)
                loss_pi = - loss_surr - self.entropy_coeff * loss_entropy
                self.optimizer_pi.zero_grad()
                loss_pi.backward()
                self.optimizer_pi.step()

                log_dict = {"surr_loss": loss_surr, "entropy": loss_entropy, "kl_divergence": kl, "is_ratio":ratio.mean(),
                            'pi_update_step': i_pi, 'pi_update': None}
                self.logger.log(**{'episodic/' + '_' + key: log_dict[key] for key in log_dict.keys()})
                kl_all.append(kl.abs().item())
                # if self.target_kl is not None and kl.abs() > 1.5 * self.target_kl:
                #     break

            for i_v in range(self.n_update_v):
                batch_returns = returns
                batch_state = s

                if n_minibatch > 1:
                    idxs = np.random.choice(range(batch_total), size=batch_size, replace=False)
                    [batch_returns, batch_state] = [item[idxs] for item in [batch_returns, batch_state]]
                    
                    if self.use_graph_feature:
                        batch_feed_dict = {key:value[idxs] for key,value in feed_dict.items()}
    
                batch_v_new = self._evalV(batch_state,feed_dict=batch_feed_dict)
                loss_v = ((batch_v_new - batch_returns) ** 2).mean()
                self.optimizer_v.zero_grad()
                loss_v.backward()
                self.optimizer_v.step()

                var_v = ((batch_returns - batch_returns.mean()) ** 2).mean()
                rel_v_loss = loss_v / (var_v + 1e-8)
    
                log_dict = {'v_loss': loss_v, 'v_update': None, 'v_var': var_v, 'rel_v_loss': rel_v_loss,
                            'v_update_step': i_v}
                self.logger.log(**{'episodic_loss/' +  key: log_dict[key] for key in log_dict.keys()})

            log_dict = {"update": None, "reward": r, "value": value_old, "clip": clip, "returns": returns,
                        "advantages": advantages_old.abs()}
            self.logger.log(**{'episodic_loss/' + key: log_dict[key] for key in log_dict.keys()})

        return [r.mean().item(), loss_entropy.item(), max(kl_all)]

    @torch.no_grad()
    def _process_traj(self, s, a, r, s1, d, logp, feed_dict=None):
        # 过网络得到value_old， 使用GAE计算adv和return
        """
        Input are all in shape [batch_size, T, n_agent, dim]
        """
     
        b, T, dim_s = s.shape
        s, a, r, s1, d, logp = [item.to(self.device) for item in [s, a, r, s1, d, logp]]

        new_feed_dict = {}
        feed_dict_t_1 = {}
        if self.use_graph_feature:
            new_feed_dict['id'] = feed_dict['id'].view(b*T,1)
            new_feed_dict['poi'] = feed_dict['poi'].view(b*T,self.poi_num,self.poi_feature_dim)
            new_feed_dict['carrier'] = feed_dict['carrier'].view(b*T,self.carrier_num,self.uav_feature_dim)
            new_feed_dict['uav'] = feed_dict['uav'].view(b*T,self.uav_num,self.uav_feature_dim)
            new_feed_dict['uav-carrier'] = feed_dict['uav-carrier'].view(b*T,self.uav_num,self.carrier_num)
            new_feed_dict['uav-poi'] = feed_dict['uav-poi'].view(b*T,self.uav_num,self.poi_num)
            new_feed_dict['carrier-uav'] = feed_dict['carrier-uav'].view(b*T,self.carrier_num,self.uav_num)
            new_feed_dict['carrier-poi'] = feed_dict['carrier-poi'].view(b*T,self.carrier_num,self.poi_num)
            new_feed_dict['poi-uav'] = feed_dict['poi-uav'].view(b*T,self.poi_num,self.uav_num)
            new_feed_dict['poi-carrier'] = feed_dict['poi-carrier'].view(b*T,self.poi_num,self.carrier_num)
            feed_dict_t_1 = {key:value.select(1,T-1) for key,value in feed_dict.items()}
            
       
        # 过网络前先merge前两个维度，过网络后再复原
        value = self._evalV(s.view(-1, dim_s),feed_dict=new_feed_dict).view(b, T, -1)
        # 在evalV中实现了具体的扩展值函数逻辑
        returns = torch.zeros(value.size(), device=self.device)
        deltas, advantages = torch.zeros_like(returns), torch.zeros_like(returns)

        prev_value = self._evalV(s1.select(1, T - 1),feed_dict = feed_dict_t_1)
        if not self.use_rtg:
            prev_return = prev_value
        else:
            prev_return = torch.zeros_like(prev_value)
        prev_advantage = torch.zeros_like(prev_return)
        d_mask = d.float()
        for t in reversed(range(T)):
            deltas[:, t,  :] = r.select(1, t) + self.gamma * (1 - d_mask.select(1, t)) * prev_value - value.select(1,
                                                                                                                     t).detach()
            advantages[:, t,  :] = deltas.select(1, t) + self.gamma * self.lamda * (
                        1 - d_mask.select(1, t)) * prev_advantage
            if self.use_gae_returns:
                returns[:, t, :] = value.select(1, t).detach() + advantages.select(1, t)
            else:
                returns[:, t,  :] = r.select(1, t) + self.gamma * (1 - d_mask.select(1, t)) * prev_return

            prev_return = returns.select(1, t)
            prev_value = value.select(1, t)
            prev_advantage = advantages.select(1, t)
        if self.advantage_norm:
            advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / (
                        advantages.std(dim=1, keepdim=True) + 1e-5)

        return value.detach(), returns, advantages.detach(), None
    

    def _init_actors(self):
        if self.decoupled_actor:#这个
            actor = DecoupledEpisodicActor(**{
            "use_graph_feature":self.use_graph_feature,
            "graph_output_size":self.n_embd,
            **self.pi_args._toDict()}).to(self.device)
            return actor
        
        self.pi_args.sizes[0] = self.observation_dim
        actor = EpisodicActor(**{
            "use_graph_feature":self.use_graph_feature,
            "graph_output_size":self.graph_output_size,
            **self.pi_args._toDict()}).to(self.device)
        return actor

    def _init_vs(self):
        self.v_args.sizes[0] = self.observation_dim
        vs = EpisodicCritic(**{
            "use_graph_feature":self.use_graph_feature,
            "graph_output_size":self.graph_output_size,
            **self.v_args._toDict()}).to(self.device)
        return vs

    def save_nets(self, dir_name, iter=0, is_newbest=False):
        if not os.path.exists(dir_name + '/Models'):
            os.mkdir(dir_name + '/Models')
        prefix = 'best' if is_newbest else str(iter)
        torch.save(self.actors.state_dict(), dir_name + '/Models/' + prefix + '_episodic_actor.pt')
        torch.save(self.vs.state_dict(), dir_name + '/Models/' + prefix + '_episodic_critic.pt')
        print('RL saved successfully')

    def load_nets(self, dir_name, iter=0, best=False):
        prefix = 'best' if best else str(iter)
        self.actors.load_state_dict(torch.load(dir_name + '/Models/' + prefix + '_episodic_actor.pt'))
        self.vs.load_state_dict(torch.load(dir_name + '/Models/' + prefix + '_episodic_critic.pt'))
        print('load networks successfully')

    def _init_graph(self):
        n_embd = self.n_embd
        type_att_size = n_embd
        type_fusion = 'mean'
        net_schema = {
            'uav': ['carrier', 'poi'],  # n x n, n x poi
            'carrier': ['uav', 'poi'],  # n x n, n x poi, n x node
            'poi': ['uav', 'carrier'],  # poi x n, poi x n
        }
        # 只用了一层
        self.grapher = []
        if self.decoupled_actor:
            self.graph_output_size =  self.n_embd + self.poi_num*16
            layer_shape = [
                {'uav': self.uav_feature_dim, 'carrier': self.uav_feature_dim, 'poi': self.poi_feature_dim},
                {'uav': n_embd, 'carrier': n_embd, 'poi': 16},
                {'uav': n_embd, 'carrier': n_embd, 'poi': 16},
                {'uav': n_embd, 'carrier': n_embd, 'poi': 16},
                ]
            for i in range(self.hgcn_number):
                self.grapher.append(HeteGCNLayer(net_schema, layer_shape[i], layer_shape[i+1], type_fusion, type_att_size,
                                            self.input_args).to(self.device))
        else:
            self.graph_output_size =  self.n_embd + self.poi_num*self.poi_feature_dim
            layer_shape = [
                {'uav': self.uav_feature_dim, 'carrier': self.uav_feature_dim, 'poi': self.poi_feature_dim},
                {'uav': n_embd, 'carrier': n_embd, 'poi': n_embd},
                {'uav': n_embd, 'carrier': n_embd, 'poi': n_embd},  
                {'uav': n_embd, 'carrier': n_embd, 'poi': n_embd},       
                ]
            
            for i in range(self.hgcn_number):
                self.grapher.append(HeteGCNLayer(net_schema, layer_shape[i], layer_shape[i+1], type_fusion, type_att_size,
                                            self.input_args).to(self.device))
    
    def hgcn_forward(self,state):
        """_summary_

        Args:
            state (dict): 是所有子进程的poi_get_obs，相应键的内容stack之后，组成的大的obs，键包括（key-key邻接矩阵的键，id，state，mask，uav，carrier，poi）

        Returns:
            _type_: _description_
        """
        ft_dict = {key:torch.as_tensor(state[key], dtype=torch.float32,
                            device=self.device) for key in ['carrier','uav','poi']}
        adj_dict = dict_convert({key:torch.as_tensor(state[key], dtype=torch.float32,
                            device=self.device) for key in simple_adjacent_name_list})
      
        memory = None 
        
        for i in range(self.hgcn_number):
            output_dict, memory = self.grapher[i](ft_dict, adj_dict, memory)
            output_dict = self.non_linear(output_dict)
            ft_dict = output_dict
   
        weighted_feature =  torch.cat([output_dict['carrier'], output_dict['uav']], dim=1)  # B,4,16
        B,N,S = weighted_feature.shape
        if not self.decoupled_actor:
            assert N == self.n_agent and S == self.n_embd
        index  = torch.as_tensor(state['id'], dtype=torch.float32,device=self.device)
        index = index.unsqueeze(-1).expand(-1, -1, weighted_feature.shape[-1]).long()
        weighted_feature = torch.gather(weighted_feature,1,index).squeeze(1)

        weighted_feature = torch.cat([weighted_feature,output_dict['poi'].view(B,-1)],dim=1)
        #weighted_feature = weighted_feature.view(B,N*S)
        return weighted_feature, output_dict
    
    def non_linear(self, x_dict):
        y_dict = {}
        for k, v in x_dict.items():
            y_dict[k] = F.elu(v)
        return y_dict
    
     
class DecoupledEpisodicActor(nn.Module):

    def __init__(self, **net_args):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.use_graph_feature = net_args['use_graph_feature']
        self.graph_output_size = net_args['graph_output_size']
        
        assert self.use_graph_feature == True
        self.embedding = nn.Sequential(
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
    
        self.eps = 1e-5


    def forward(self, nodes):  # 多维度动作确认OK
        B,N,D = nodes.shape  #B是n_thread,N是poi_num,D是poi_feature_dim
        empty = torch.zeros([B,1,D]).to(nodes.device) 
        nodes = torch.cat([empty,nodes],dim=1) #在poi_num这个维度，添加一个空的poi，feature全为0
        embed = self.embedding(nodes)#B,N,1
        logit1 = embed.squeeze(-1)#B,N,每个poi只有一个实数
        prob1 = self.softmax(logit1) + self.eps
        prob1 = prob1 / prob1.sum(dim=-1, keepdim=True)

        return prob1  # shape = (-1, act_dim1+act_dim2)
       
    
class EpisodicActor(nn.Module):

    def __init__(self, **net_args):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        net_fn = net_args['network']  # MLP
        self.use_graph_feature = net_args['use_graph_feature']
        self.graph_output_size = net_args['graph_output_size']

        if self.use_graph_feature:
            self.network = nn.Identity()
            self.branch = nn.Sequential(
                nn.Linear(self.graph_output_size, net_args['sizes'][-1]),
                nn.ReLU(),
                nn.Linear(net_args['sizes'][-1],  net_args['branchs']),
            )
        else:
            self.network = net_fn(**net_args)
            self.branch = nn.Sequential(
                nn.Linear(net_args['sizes'][-1], net_args['sizes'][-1]),
                nn.ReLU(),
                nn.Linear(net_args['sizes'][-1],  net_args['branchs']),
            )
        self.eps = 1e-5


    def forward(self, obs):  # 多维度动作确认OK

        embed = self.network(obs)
        logit1 = self.branch(embed)
        prob1 = self.softmax(logit1) + self.eps
        #prob1 = prob1 / prob1.sum(dim=-1, keepdim=True)

        return prob1  # shape = (-1, act_dim1+act_dim2)



class EpisodicCritic(nn.Module):

    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity, **kwargs):
        super().__init__()

        
        self.use_graph_feature = kwargs['use_graph_feature']
        self.graph_output_size = kwargs['graph_output_size']

        if self.use_graph_feature:
            self.critic = nn.Identity()
            self.output = nn.Sequential(
                nn.Linear(self.graph_output_size, sizes[-2]),
                activation(),
                nn.Linear(sizes[-2], sizes[-1]),
                output_activation()
            )

        else:
            layers = []
            for j in range(len(sizes) - 2):
                act = activation
                layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]

            self.critic = nn.Sequential(*layers)
            self.output = nn.Sequential(
                nn.Linear(sizes[-2], sizes[-2]),
                activation(),
                nn.Linear(sizes[-2], sizes[-1]),
                output_activation()
            )

    def forward(self,obs):
        
        obs = self.critic(obs)
        result = self.output(obs)
        
        return result
    
    
    



def dict_convert(old_dict):
    adj_dict = {
        'uav': {},
        'carrier': {},
        'poi': {},
    }
    adj_dict['uav']['carrier'] = old_dict['uav-carrier']
    adj_dict['uav']['poi'] = old_dict['uav-poi']
    adj_dict['carrier']['uav'] = old_dict['carrier-uav']
    adj_dict['carrier']['poi'] = old_dict['carrier-poi']
    adj_dict['poi']['uav'] = old_dict['poi-uav']
    adj_dict['poi']['carrier'] = old_dict['poi-carrier']

    return adj_dict