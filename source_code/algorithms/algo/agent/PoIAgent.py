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
from algorithms.algo.buffer import Trajectory
from torch.distributions import Categorical, Distribution
from typing import List


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

class PoIAgent(nn.ModuleList, AgentBase):
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
        
 
        self.observation_dim =  {'poi_'+key:value for key,value in agent_args.observation_dim.items()}
        self.action_space = agent_args.poi_action_space
        self.n_branch = len(self.action_space['poi_carrier'].nvec)
        self.branch_dim = self.action_space['poi_carrier'].nvec[0]
    
        if input_args.use_stack_frame:
            self.observation_dim *= 4

        # self.action_dim = sum([dim.n for dim in self.action_space])
        self.pi_args = agent_args.pi_args
        self.v_args = agent_args.v_args
        self.share_parameters = False
        self.agent_type = ['poi_'+type for type in agent_args.agent_type]
        self.n_agent = {'poi_'+key:value for key,value in agent_args.n_agent.items()}

      

        self.use_hgcn = agent_args.use_hgcn
        self.use_lambda = agent_args.use_lambda
        self.use_mate = agent_args.use_mate
        self.task_emb_dim = agent_args.task_emb_dim
        self.mate_type = agent_args.mate_type
        self.mate_rl_gradient = agent_args.mate_rl_gradient
        self.restore_mate = True if input_args.mate_path != '' else False
        self.share_mate = input_args.share_mate
        self.use_gcn = False
        self.gcn_emb_dim = agent_args.gcn_emb_dim
        self.gcn_nodes_num = agent_args.gcn_nodes_num
        self.n_embd = input_args.n_embd
        self.random_permutation = input_args.random_permutation

        self.gcn_model = {type: [] for type in self.agent_type}
        self.gcn_parameters = []
        for type in self.agent_type:
            for n in range(self.n_agent[type]):
                if self.use_gcn:
                    model = GCN(self.gcn_emb_dim, self.gcn_nodes_num, self.n_embd, device=self.device)
                    self.gcn_parameters += model.parameters()
                else:
                    model = None
                self.gcn_model[type].append(model)

        self.actors = self._init_actors()  # collect_pi和collect_v应该一样吧？
        self.vs = self._init_vs()
        self.optimizer_v = Adam(list(set(list(self.vs.parameters()) + self.gcn_parameters)), lr=self.lr_v)
        self.optimizer_pi = Adam(list(set(list(self.actors.parameters()) + self.gcn_parameters)), lr=self.lr)


    def log_model(self):
        for type in self.agent_type:
            for agent_id in range(self.n_agent[type]):
                prefix = f"{type}_{agent_id}/"
                self.log_parameters(f"{prefix}/Critic_", self.vs[type][agent_id].named_parameters())
                self.log_parameters(f"{prefix}/Actor_", self.actors[type][agent_id].named_parameters())

            if self.use_mate:
                if self.mate_type in ['mix', 'ind']:
                    self.log_parameters(f"{prefix}/Encoders0_", self.mate[type].encoders[0].named_parameters())
                    self.log_parameters(f"{prefix}/Encoders1_", self.mate[type].encoders[1].named_parameters())
                else:
                    self.log_parameters(f"{prefix}/Encoders_", self.mate[type].encoders.named_parameters())

                self.log_parameters(f"{prefix}/Decoders_", self.mate[type].decoder.named_parameters())

    def log_parameters(self, prefix, n_params):
        log_dict = {}
        for p_name, param in n_params:
            p_name = prefix + p_name
            log_dict[p_name] = torch.norm(param).item()
            if param.grad is not None:
                log_dict[p_name + ".grad"] = torch.norm(param.grad).item()
        self.logger.log(**log_dict)


    def act(self, state, task_emb=None):
        """
        非向量环境：Requires input of [batch_size, n_agent, dim] or [n_agent, dim].
        向量环境：Requires input of [batch_size*n_thread, n_agent, dim] or [n_thread, n_agent, dim].
        其中第一维度的值在后续用-1表示
        This method is gradient-free. To get the gradient-enabled probability information, use get_logp().
        Returns a distribution with the same dimensions of input.
        """
        with torch.no_grad():
            all_probs = {}
            for type in self.agent_type:
                action_mask = torch.as_tensor(state['mask_' + type], dtype=torch.float32, device=self.device).unsqueeze(2).repeat(1,1,self.n_branch,1)
                s = torch.as_tensor(state[type], dtype=torch.float32, device=self.device)
                if self.use_gcn:
                    nodes = torch.as_tensor(np.stack([state['Nodes']] * self.n_agent[type], axis=1),
                                            dtype=torch.float32, device=self.device)
                    edges = torch.as_tensor(np.stack([state['Edges']] * self.n_agent[type], axis=1),
                                            dtype=torch.float32, device=self.device)
                    graph = [nodes, edges]
                else:
                    graph = None
                assert s.dim() == 3
                s = s.to(self.device)
                probs = []
                for i in range(self.n_agent[type]):
                    g_in = [graph[0][:, i, ...], graph[1][:, i, ...]] if self.use_gcn else None
                    if task_emb is not None:
                        t_emb = task_emb[type].to(self.device)
                        prob = self.actors[type][i](s[:, i, :], task_emb=t_emb[:, i, :], graph_inputs=g_in) # TODO 形状
                    else:
                        prob = self.actors[type][i](s[:, i, :], graph_inputs=g_in)
                    probs.append(prob)
                probs = torch.stack(probs, dim=1)  # shape = (-1, NUM_AGENT, act_dim1+act_dim2)  
                probs = probs.view(-1,self.n_agent[type],self.n_branch,self.branch_dim)  
                probs[action_mask == 0] = 0
                all_probs[type] = MultiCategorical([Categorical(probs[:,:,i,:]) for i in range(self.n_branch)])

            return all_probs

    def get_logp(self, state, action, type=None, task_emb=None, graph=None):
        """
        Requires input of [batch_size, n_agent, dim] or [n_agent, dim].
        Returns a tensor whose dim() == 3.
        """
        all_type = self.agent_type if type is None else [type]

        log_prob_all = {}

        for type in all_type:
            s = torch.as_tensor(state[type], dtype=torch.float32,
                                device=self.device) if type is None else torch.as_tensor(state, dtype=torch.float32,
                                                                                         device=self.device)
            a = action[type] if type is None else action
            while s.dim() <= 2:
                s = s.unsqueeze(0)
                a = a.unsqueeze(0)
            while a.dim() < s.dim():
                a = a.unsqueeze(-1)

            # Now s[i].dim() == 2, a.dim() == 3
            log_prob = []
            for i in range(self.n_agent[type]):
                g_in = [graph[0][:, i, ...], graph[1][:, i, ...]] if self.use_gcn else None
                if task_emb is not None:
                    t_emb = task_emb[type].to(self.device) if type is None else task_emb.to(self.device)
                    probs = self.actors[type][i](s[:, i, :], task_emb=t_emb[:, i, :], graph_inputs=g_in)
                else:
                    probs = self.actors[type][i](s[:, i, :], graph_inputs=g_in)  # [320,2,9]
                index = torch.select(a, dim=1, index=i).long()[:, 0]
                ans_list = []
                for i in range(self.n_branch):
                    prob = probs[:,i*self.branch_dim:(i+1)*self.branch_dim]
                    ans = torch.log(torch.gather(prob, dim=-1, index=index.unsqueeze(-1)))  # [320,2,1]
                    ans_list.append(ans.squeeze(-1))
                log_prob.append(torch.stack(ans_list,dim=-1))
            log_prob = torch.stack(log_prob, dim=1)
            while log_prob.dim() < 3:
                log_prob = log_prob.unsqueeze(-1)
            log_prob_all[type] = log_prob

        return log_prob_all if type is None else log_prob_all[type]

    def _evalV(self, state, type=None, task_emb=None, graph=None):
        network = self.vs
        all_type = self.agent_type if type is None else [type]
        values_all = {}
        for type in all_type:
            s = state[type].to(self.device) if type is None else state.to(self.device)
            values = []
            for i in range(self.n_agent[type]):
                g_in = [graph[0][:, i, ...], graph[1][:, i, ...]] if self.use_gcn else None
                if task_emb is not None:
                    t_emb = task_emb[type].to(self.device) if type is None else task_emb.to(self.device)
                    values.append(network[type][i](s[:, i, :], task_emb=t_emb[:, i, :], graph_inputs=g_in))
                else:
                    values.append(network[type][i](s[:, i, :], task_emb=None, graph_inputs=g_in))
            values_all[type] = torch.stack(values, dim=1)
        return values_all if type is None else values_all[type]

    def updateAgent(self, trajs, clip=None):
        time_t = time.time()
        if clip is None:
            clip = self.clip
        n_minibatch = self.n_minibatch

        names = Trajectory.names()  # ['s', 'a', 'r', 's1', 'd', 'logp']

        names.remove('h')
        names.remove("nodes")
        if not self.use_mate:
            for n in ['ae_h', 't_emb', 'a0', 'r0']:
                if n in names: names.remove(n)
        if not self.use_lambda:
            for n in ['cost']:
                if n in names: names.remove(n)

        if not self.use_gcn:
            for n in ['nodes', 'edges']:
                if n in names: names.remove(n)

        if not self.random_permutation:
            for n in ['cols']:
                if n in names: names.remove(n)

        if not self.use_hgcn:
            for n in ['uav-carrier', 'uav-poi', 'carrier-uav', 'carrier-poi', 'carrier-road', 'poi-uav', 'poi-carrier',
                      'road-carrier', 'uav-epoi', 'uav-road', 'carrier-epoi']:
                if n in names: names.remove(n)

        if not self.use_temporal_type or not self.use_hgcn:
            names.remove('memory')

        for type in self.agent_type:
            traj_all = {name: [] for name in names}

            for traj in trajs[type]:  # len(trajs) == n_thread
                for name in names:
                    traj_all[name].append(traj[name])
            traj = {name: torch.stack(value, dim=0) for name, value in traj_all.items()}

            for i_update in range(1):
                s, a, r, s1, d, logp = traj['s'], traj['a'], traj['r'], traj['s1'], traj['d'], traj['logp']
                s, a, r, s1, d, logp = [item.to(self.device) for item in [s, a, r, s1, d, logp]]

                if self.use_mate:
                    ae_h, t_emb, a0, r0 = [traj[key].to(self.device) for key in ['ae_h', 't_emb', 'a0', 'r0']]
                else:
                    ae_h, t_emb, a0, r0 = [None for _ in range(4)]

                if self.use_gcn:
                    nodes, edges = [traj[key].to(self.device) for key in ['nodes', 'edges']]
                    nodes = torch.stack([nodes] * self.n_agent[type], dim=1)
                    edges = torch.stack([edges] * self.n_agent[type], dim=1)
                else:
                    nodes, edges = None, None
                # 关键：all in shape [n_thread, T, n_agent, dim]
                value_old, returns, advantages, _ = self._process_traj(traj['s'], traj['a'], traj['r'], traj['s1'],
                                                                       traj['d'], traj['logp'], traj.get('h', None),
                                                                       traj.get('ae_h', None), traj.get('t_emb', None),
                                                                       traj.get('a0', None), traj.get('r0', None),
                                                                       traj.get('nodes', None), traj.get('edges', None),
                                                                       type)  # 不同traj分开计算adv和return
                advantages_old = advantages

                _, T, n, d_s = s.size()
                d_a = a.size()[-1]
                s = s.view(-1, n, d_s)
                a = a.view(-1, n, d_a)
                logp = logp.view(-1, n, d_a)
                advantages_old = advantages_old.view(-1, n, 1)
                returns = returns.view(-1, n, 1)
                value_old = value_old.view(-1, n, 1)
                # 关键：s, a, logp, adv, ret, v are now all in shape [-1, n_agent, dim] 因为计算完adv和return后可以揉在一起做mini_batch训练
                batch_total = logp.size()[0]
                batch_size = int(batch_total / n_minibatch)
                # batch_size = self.batch_size

                if self.use_mate:
                    t_emb = t_emb.view(-1, n, self.task_emb_dim * 2)
                    ae_h = ae_h.view(-1, n, ae_h.size()[-1])
                    a0 = a0.view(-1, n)
                    r0 = r0.view(-1, n)


                if self.use_gcn:
                    nodes = nodes.view(-1, n, self.gcn_nodes_num, 3)
                    edges = edges.view(-1, n, 2, edges.size()[-1])
                    # edges = edges.view(-1,n,self.gcn_nodes_num,self.gcn_nodes_num)

                kl_all = []
                for i_pi in range(self.n_update_pi):
                    batch_state, batch_action, batch_logp, batch_advantages_old = [s, a, logp, advantages_old]
                    if self.use_gcn:
                        batch_nodes = nodes
                        batch_edges = edges
                    else:
                        batch_edges = batch_nodes = None

                    if self.use_mate:
                        if self.mate_rl_gradient:
                            batch_task_emb, _ = self.encode(s, a0, r0, ae_h, no_grads=False, type=type,
                                                            graph_input=[batch_nodes, batch_edges])
                        else:
                            batch_task_emb = t_emb
                    else:
                        batch_task_emb = None

                    if n_minibatch > 1:
                        idxs = np.random.choice(range(batch_total), size=batch_size, replace=False)
                        [batch_state, batch_action, batch_logp, batch_advantages_old] = [item[idxs] for item in
                                                                                         [batch_state, batch_action,
                                                                                          batch_logp,
                                                                                          batch_advantages_old]]
                        if self.use_mate:
                            batch_task_emb = t_emb[idxs]
                        if self.use_gcn:
                            batch_nodes = batch_nodes[idxs]
                            batch_edges = batch_edges[idxs]
                    batch_logp_new = self.get_logp(batch_state, batch_action, type=type, task_emb=batch_task_emb,
                                                   graph=[batch_nodes, batch_edges])

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
  
                    log_dict = {"surr_loss": loss_surr, "entropy": loss_entropy, "kl_divergence": kl,
                                'pi_update_step': i_pi, 'pi_update': None}
                    self.logger.log(**{'poi_loss/' + type + '_' + key: log_dict[key] for key in log_dict.keys()})
                    kl_all.append(kl.abs().item())
                    # if self.target_kl is not None and kl.abs() > 1.5 * self.target_kl:
                    #     break

                for i_v in range(self.n_update_v):
                    batch_returns = returns
                    batch_state = s
                    if self.use_gcn:
                        batch_nodes = nodes
                        batch_edges = edges
                    if self.use_mate:
                        if self.mate_rl_gradient:
                            batch_task_emb, _ = self.encode(s, a0, r0, ae_h, no_grads=False, type=type,
                                                            graph_input=[batch_nodes, batch_edges])
                        else:
                            batch_task_emb = t_emb
                    else:
                        batch_task_emb = None

                    if n_minibatch > 1:
                        idxs = np.random.choice(range(batch_total), size=batch_size, replace=False)
                        [batch_returns, batch_state] = [item[idxs] for item in [batch_returns, batch_state]]
                        if self.use_mate:
                            batch_task_emb = batch_task_emb[idxs]
                        if self.use_gcn:
                            batch_nodes = batch_nodes[idxs]
                            batch_edges = batch_edges[idxs]
                    batch_v_new = self._evalV(batch_state, type=type, task_emb=batch_task_emb,
                                              graph=[batch_nodes, batch_edges])
                    loss_v = ((batch_v_new - batch_returns) ** 2).mean()
                    self.optimizer_v.zero_grad()
                    loss_v.backward()
                    self.optimizer_v.step()

                    var_v = ((batch_returns - batch_returns.mean()) ** 2).mean()
                    rel_v_loss = loss_v / (var_v + 1e-8)
        
                    log_dict = {'v_loss': loss_v, 'v_update': None, 'v_var': var_v, 'rel_v_loss': rel_v_loss,
                                'v_update_step': i_v}
                    self.logger.log(**{'poi_loss/' + type + '_' + key: log_dict[key] for key in log_dict.keys()})

                    # if rel_v_loss < self.v_thres:
                    #     break


                log_dict = {"update": None, "reward": r, "value": value_old, "clip": clip, "returns": returns,
                            "advantages": advantages_old.abs()}
                self.logger.log(**{'poi_loss/' + type + '_' + key: log_dict[key] for key in log_dict.keys()})

        return [r.mean().item(), loss_entropy.item(), max(kl_all)]

    def checkConverged(self, ls_info):
        return False

    def _init_actors(self):
        
        self.pi_args.type = 'poi_carrier'
        self.pi_args.use_gcn = False
        self.pi_args.sizes[0] = self.observation_dim['poi_carrier']
        single_actor = MultiCategoricalActor(**{
                'n_branch':self.n_branch, 
                'gcn_model': self.gcn_model['poi_carrier'][0],
                'n_embd': self.n_embd,
                **self.pi_args._toDict()}).to(self.device)
        
        actors = nn.ModuleDict()
        for type in self.agent_type:
            actor = nn.ModuleList()
            self.pi_args.type = type
            self.pi_args.sizes[0] = self.observation_dim[type]
            for i in range(self.n_agent[type]):
                if self.share_parameters:
                    actor.append(single_actor)
                else:
                    actor.append(MultiCategoricalActor(**{
                        'n_branch':self.n_branch, 
                        'gcn_model': self.gcn_model[type][i],
                        'n_embd': self.n_embd,
                        **self.pi_args._toDict()}).to(self.device))
            actors[type] = actor
        return actors

    def _init_vs(self):
        critics = nn.ModuleDict()
        
        self.v_args.use_gcn = False
        self.v_args.sizes[0] = self.observation_dim['poi_carrier']
        single_v = Critic(**{
                    'gcn_model': self.gcn_model['poi_carrier'][0],
                    'n_embd': self.n_embd,
                    **self.v_args._toDict()}).to(self.device)
        
        for type in self.agent_type:
            vs = nn.ModuleList()
            for i in range(self.n_agent[type]):
                self.v_args.sizes[0] = self.observation_dim[type]
                if self.share_parameters:
                    vs.append(single_v)
                else:
                    vs.append(Critic(**{
                        'gcn_model': self.gcn_model[type][i],
                        'n_embd': self.n_embd,
                        **self.v_args._toDict()}).to(self.device))
            critics[type] = vs
        return critics

    @torch.no_grad()
    def _process_traj(self, s, a, r, s1, d, logp, h=None, ae_h=None, t_emb=None, a0=None, r0=None, nodes=None,
                      edges=None, type=None):
        # 过网络得到value_old， 使用GAE计算adv和return
        """
        Input are all in shape [batch_size, T, n_agent, dim]
        """
        type = 'carrier' if type is None else type
        b, T, n, dim_s = s.shape
        s, a, r, s1, d, logp = [item.to(self.device) for item in [s, a, r, s1, d, logp]]
        if self.use_mate:
            b_e, T_e, n_e, dim_e = t_emb.shape
            t_emb = t_emb.to(self.device).view(-1, n_e, dim_e)
        if self.use_gcn:
            # nodes = nodes.to(self.device).view(-1, T, n, self.gcn_nodes_num, 3)
            # edges = edges.to(self.device).view(-1, T, n, 2, edges.size()[-1])
            # nodes_all = nodes.view(-1, n, self.gcn_nodes_num, 3)
            # edges_all = edges.to(self.device).view(-1, n, 2, edges.size()[-1])
            nodes_all = torch.stack([nodes] * n, dim=1).view(-1, n, self.gcn_nodes_num, 3)
            edges_all = torch.stack([edges] * n, dim=1).view(-1, n, 2, edges.size()[-1])
            graph_all = [nodes_all, edges_all]
            nodes_t = nodes.select(1, T - 1)
            edges_t = edges.select(1, T - 1)
        else:
            graph_all = None
            nodes_t = edges_t = None
            # edges_all = edges.to(self.device).view(-1,n,self.gcn_nodes_num,self.gcn_nodes_num)
        # 过网络前先merge前两个维度，过网络后再复原
        value = self._evalV(s.view(-1, n, dim_s), type=type,
                            task_emb=t_emb, graph=graph_all).view(b, T, n, -1)
        # 在evalV中实现了具体的扩展值函数逻辑
        returns = torch.zeros(value.size(), device=self.device)
        deltas, advantages = torch.zeros_like(returns), torch.zeros_like(returns)

        # t_emb_next = {key:torch.zeros((b,n,self.task_emb_dim)) for key in all_type} if type is None else torch.zeros((b,n,self.task_emb_dim))
        if self.use_mate:
            t_emb_next, _ = self.encode(s1.select(1, T - 1), a.select(1, T - 1).squeeze(-1),
                                        r.select(1, T - 1).squeeze(-1), ae_h.select(1, T - 1), no_grads=True, type=type,
                                        graph_input=[nodes_t, edges_t])
        else:
            t_emb_next = None
        if self.use_gcn:
            graph_next = [torch.stack([nodes.select(1, T - 1)] * n, dim=1), torch.stack([edges.select(1, T - 1)] * n, dim=1)]
        else:
            graph_next = None

        prev_value = self._evalV(s1.select(1, T - 1), type=type, task_emb=t_emb_next, graph=graph_next)
        if not self.use_rtg:
            prev_return = prev_value
        else:
            prev_return = torch.zeros_like(prev_value)
        prev_advantage = torch.zeros_like(prev_return)
        d_mask = d.float()
        for t in reversed(range(T)):
            deltas[:, t, :, :] = r.select(1, t) + self.gamma * (1 - d_mask.select(1, t)) * prev_value - value.select(1,
                                                                                                                     t).detach()
            advantages[:, t, :, :] = deltas.select(1, t) + self.gamma * self.lamda * (
                        1 - d_mask.select(1, t)) * prev_advantage
            if self.use_gae_returns:
                returns[:, t, :, :] = value.select(1, t).detach() + advantages.select(1, t)
            else:
                returns[:, t, :, :] = r.select(1, t) + self.gamma * (1 - d_mask.select(1, t)) * prev_return

            prev_return = returns.select(1, t)
            prev_value = value.select(1, t)
            prev_advantage = advantages.select(1, t)
        if self.advantage_norm:
            advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / (
                        advantages.std(dim=1, keepdim=True) + 1e-5)

        return value.detach(), returns, advantages.detach(), None
        # else:
        #     reduced_advantages = self.collect_v.reduce_sum(advantages.view(-1, n, 1)).view(advantages.size())
        #     if reduced_advantages.size()[1] > 1:
        #         reduced_advantages = (reduced_advantages - reduced_advantages.mean(dim=1, keepdim=True)) / (reduced_advantages.std(dim=1, keepdim=True) + 1e-5)
        #     return value.detach(), returns, advantages.detach(), reduced_advantages.detach()

    def zero_grad(self):
        for type in self.agent_type:
            self.mate[type].zero_grad()
        return

    def save_nets(self, dir_name, iter=0, is_newbest=False):
        if not os.path.exists(dir_name + '/Models'):
            os.mkdir(dir_name + '/Models')
        prefix = 'best' if is_newbest else str(iter)
        torch.save(self.actors.state_dict(), dir_name + '/Models/' + prefix + '_poi_actor.pt')
        torch.save(self.vs.state_dict(), dir_name + '/Models/' + prefix + '_poi_critic.pt')
        if self.use_mate:
            for type in self.agent_type:
                self.mate[type].save(dir_name + '/Models/' + prefix + '_mate_' + type + '.pt')
        print('RL saved successfully')

    def load_nets(self, dir_name, iter=0, best=False):
        prefix = 'best' if best else str(iter)
        self.actors.load_state_dict(torch.load(dir_name + '/Models/' + prefix + '_poi_actor.pt'))
        self.vs.load_state_dict(torch.load(dir_name + '/Models/' + prefix + '_poi_critic.pt'))
        if self.use_mate:
            for type in self.agent_type:
                self.mate[type].restore(dir_name + '/Models/' + prefix + '_mate_' + type + '.pt')
        print('load networks successfully')
