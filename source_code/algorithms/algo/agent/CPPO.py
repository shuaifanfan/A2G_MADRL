# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 02:04:27 2022

@author: 86153
"""
import copy
import time
import os
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from source_code.algorithms.utils import collect, mem_report
from source_code.algorithms.models import MLP, Critic, GCN
from source_code.algorithms.algo.agent_base import AgentBase
from tqdm.std import trange
# from algorithms.algorithm import ReplayBuffer

from gym.spaces.box import Box
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import pickle
from copy import deepcopy as dp
from source_code.algorithms.models import CategoricalActor
import random
import multiprocessing as mp
from torch import distributed as dist
import argparse
from source_code.algorithms.algo.buffer import MultiCollect, Trajectory, TrajectoryBuffer, ModelBuffer, \
    adjacent_name_list
from source_code.algorithms.mat.utils.util import check
from source_code.algorithms.mat.utils.valuenorm import ValueNorm
from source_code.algorithms.mat.utils.util import get_grad_norm, huber_loss, mse_loss
from torch_geometric.nn import GCNConv


class CPPOAgent(nn.ModuleList, AgentBase):
    """Everything in and out is torch Tensor."""

    def __init__(self, logger, device, agent_args, input_args):
        super().__init__()
        self.input_args = input_args
        self.test = input_args.test
        self.logger = logger  # LogClient类对象
        self.device = device
        self.agent_args = agent_args
        self.n_agent = agent_args.n_agent
        self.n_poi = agent_args.n_poi
        self.n_node = agent_args.n_node
        self.gamma = agent_args.gamma
        self.lamda = agent_args.lamda
        self.each_ig_step_num = agent_args.each_ig_step_num
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

        self.use_rtg = agent_args.use_rtg
        self.use_gae_returns = agent_args.use_gae_returns
        self.env_name = input_args.env
        self.algo_name = input_args.algo
        self.advantage_norm = agent_args.advantage_norm

        self.observation_dim = agent_args.observation_dim['uav']
        self.state_dim = agent_args.observation_dim['State']
        self.n_agent_all = sum([value for value in self.n_agent.values()])
        self.action_space = agent_args.action_space
        self.act_dim = agent_args.action_space['uav'].n
        if input_args.use_stack_frame:
            self.observation_dim *= 4

        self.action_dim = [dim.n for dim in self.action_space.values()]
        self.pi_args = agent_args.pi_args
        self.v_args = agent_args.v_args
        self.agent_type = agent_args.agent_type

        self.use_lambda = agent_args.use_lambda
        self.use_mate = agent_args.use_mate
        self.task_emb_dim = agent_args.task_emb_dim
        self.mate_rl_gradient = agent_args.mate_rl_gradient
        self.mate_type = agent_args.mate_type
        self.restore_mate = True if input_args.mate_path != '' else False
        self.use_gcn = agent_args.use_gcn
        self.gcn_emb_dim = agent_args.gcn_emb_dim
        self.gcn_nodes_num = agent_args.gcn_nodes_num
        self.share_mate = input_args.share_mate
        self.cat_position = input_args.cat_position
        self.rep_learning = input_args.rep_learning
        self.rep_iter = input_args.rep_iter
        self.lr_scheduler = input_args.lr_scheduler
        self.init_lr = self.current_lr = input_args.lr
        self.decay_eps = input_args.decay_eps
        self.use_hgcn = input_args.use_hgcn
        self.random_permutation = input_args.random_permutation
        self.permutation_strategy = input_args.permutation_strategy
        self.permutation_eps = input_args.permutation_eps
        self.deno_eps = 1e-8
        if "ucb" in self.permutation_strategy:
            self.permute_count = np.ones(self.n_agent_all)
            self.confidence = input_args.ucb_confidence
            
        self.agent_sequence = np.random.permutation(self.n_agent_all)
        # self.agent_sequence = np.array([0, 2, 3, 1])
        self.use_sequential_update = input_args.use_sequential_update
        self.act_update_seq_diff = input_args.act_update_seq_diff
        self.use_temporal_type = input_args.use_temporal_type
        self.hgcn_rollout_len = input_args.hgcn_rollout_len
        self.credit_assign = input_args.credit_assign
        self.matx_type = 'mat'
        self.n_block = 2
        self.n_embd = input_args.n_embd
        self.n_head = 2
        self.encode_state = False
        self.dec_actor = True if self.matx_type == 'mat_dec' else False
        self.share_actor = True if self.matx_type == 'mat_dec' else False
        self.opti_eps = 1e-5
        self.weight_decay = 0

        self.clip_param = input_args.clip_param
        self.ppo_epoch = 10
        self.num_mini_batch = 1
        self.data_chunk_length = 10
        self.value_loss_coef = 1
        self.entropy_coef = 0.01
        self.max_grad_norm = 10
        self.huber_delta = 10

        self._use_recurrent_policy = False
        self._use_naive_recurrent = False
        self._use_max_grad_norm = input_args.use_max_grad_norm
        self._use_clipped_value_loss = input_args.use_clipped_value_loss
        self._use_huber_loss = input_args.use_huber_loss
        self._use_valuenorm = input_args.use_valuenorm
        self._use_value_active_masks = False
        self._use_policy_active_masks = False


        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

        if self.matx_type in ["mat", "mat_dec"]:
            from source_code.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer as MAT
        elif self.matx_type == "mat_gru":
            from source_code.algorithms.mat.algorithm.mat_gru import MultiAgentGRU as MAT
        elif self.matx_type == "mat_decoder":
            from source_code.algorithms.mat.algorithm.mat_decoder import MultiAgentDecoder as MAT
        elif self.matx_type == "mat_encoder":
            from source_code.algorithms.mat.algorithm.mat_encoder import MultiAgentEncoder as MAT
        else:
            raise NotImplementedError

        self.all_parameters = []
        if self.use_gcn:
            self.gcn_model = GCN(self.gcn_emb_dim, self.gcn_nodes_num, self.n_embd, device=self.device)
            self.all_parameters += list(self.gcn_model.parameters())
        else:
            self.gcn_model = None
        #added by zf,下面参数分别是state_dim:(车+飞机）*3+兴趣点*3，observation_dim:(车+飞机）*4+2+兴趣点数*5，act_dim: 15还是飞机的动作空间9？
        #n_agent_all: 车+飞机数，n_embed:32 ，dec_actor:False, share_actor:False    
        self.transformer = MAT(self.state_dim, self.observation_dim, self.act_dim, self.n_agent_all,
                               n_block=self.n_block, n_embd=self.n_embd, n_head=self.n_head,
                               encode_state=self.encode_state, device=self.device,
                               action_type='Discrete', dec_actor=self.dec_actor,
                               share_actor=self.share_actor, **{
                'use_lambda': self.use_lambda,
                'task_emb_dim': self.task_emb_dim,
                'use_mate': self.use_mate,
                'use_gcn': self.use_gcn,
                'gcn_emb_dim': self.gcn_emb_dim,
                'gcn_nodes_num': self.gcn_nodes_num,
                'cat_position': self.cat_position,
                'gcn_model': self.gcn_model,
                'input_args': self.input_args,
                'n_poi': self.n_poi,
                'n_node': self.n_node
            })
        self.all_parameters += list(self.transformer.parameters())
        self.optimizer = torch.optim.Adam(list(set(self.all_parameters)),
                                          lr=self.lr, eps=self.opti_eps,
                                          weight_decay=self.weight_decay)

        if self.lr_scheduler in ['cos']:
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5000)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=25000)
        elif self.lr_scheduler in ['linear']:
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.9)

        if self.use_mate:
            if agent_args.mate_type == 'mix':
                from source_code.algorithms.model.mix_mate import MixedMATE as MATE
                config_path = '/home/liuchi/fangchen/AirDropMCS/source_code/algorithms/config/ae_config/mix_mate.yaml'
            elif agent_args.mate_type == 'ind':
                from source_code.algorithms.model.ind_mate import IndependentMATE as MATE
                config_path = '/home/liuchi/fangchen/AirDropMCS/source_code/algorithms/config/ae_config/ind_mate.yaml'
            elif agent_args.mate_type == 'cen':
                from source_code.algorithms.model.cen_mate import CentralisedMATE as MATE
                config_path = '/home/liuchi/fangchen/AirDropMCS/source_code/algorithms/config/ae_config/cen_mate.yaml'
            else:
                raise NotImplementedError("Unsupported")

            self.mate = {}
            if self.share_mate:
                mate = MATE([Box(low=-1, high=1, shape=(self.observation_dim,)) for _ in range(self.n_agent_all)],
                            [self.action_space['uav'] for _ in range(self.n_agent_all)],
                            gcn_models=[self.gcn_model for _ in range(self.n_agent_all)], config_path=config_path,
                            config=agent_args, )
            else:
                mate = None
            for type in self.agent_type:
                if mate is None:
                    self.mate[type] = MATE(
                        [Box(low=-1, high=1, shape=(self.observation_dim,)) for _ in range(self.n_agent[type])],
                        [self.action_space[type] for _ in range(self.n_agent[type])],
                        gcn_models=[self.gcn_model for _ in range(self.n_agent[type])], config_path=config_path,
                        config=agent_args)
                else:
                    self.mate[type] = mate

            if self.restore_mate:
                for type in self.agent_type:
                    self.mate[type].restore(os.path.join(input_args.mate_path, type))



    def log_model(self):
        if self.use_hgcn:
            self.log_parameters("adj_dict/", self.adj)
        self.log_parameters(f"MAT/", self.transformer.named_parameters())
        for type in self.agent_type:
            for agent_id in range(self.n_agent[type]):
                prefix = f"{type}_{agent_id}/"
            if self.use_mate:
                if self.mate_type in ['mix', 'ind']:
                    self.log_parameters(f"{prefix}/Encoders0_", self.mate[type].encoders[0].named_parameters())
                    self.log_parameters(f"{prefix}/Encoders1_", self.mate[type].encoders[1].named_parameters())
                else:
                    self.log_parameters(f"{prefix}/Encoders", self.mate[type].encoders.named_parameters())
                self.log_parameters(f"{prefix}/Decoders_", self.mate[type].decoder.named_parameters())

    def log_parameters(self, prefix, n_params):
        log_dict = {}
        if n_params is None:
            return
        if isinstance(n_params, list):
            for p_name, param in n_params:
                p_name = prefix + p_name
                log_dict[p_name] = torch.norm(param).item()
                if param.grad is not None:
                    log_dict[p_name.replace(".weight", ".grad")] = torch.norm(param.grad).item()
        elif isinstance(n_params, dict):
            for key, value in n_params.items():
                log_dict[prefix + key] = torch.norm(value).item()
        self.logger.log(**log_dict)

    def act(self, state, task_emb=None, random_seq=False, permutation_sequence=None, deterministic=False, memory=None):
        """
        非向量环境：Requires input of [batch_size, n_agent, dim] or [n_agent, dim].
        向量环境：Requires input of [batch_size*n_thread, n_agent, dim] or [n_thread, n_agent, dim].
        其中第一维度的值在后续用-1表示
        This method is gradient-free. To get the gradient-enabled probability information, use get_logp().
        Returns a distribution with the same dimensions of input.
        """

        cent_obs = torch.as_tensor(state['State'], dtype=torch.float32, device=self.device).unsqueeze(1).repeat(1,
                                                                                                                self.n_agent_all,
                                                                                                                1)
        obs = torch.cat(
            [torch.as_tensor(state[type], dtype=torch.float32, device=self.device) for type in self.agent_type], dim=1)
        available_actions = torch.cat(
            [torch.as_tensor(state['mask_' + type], dtype=torch.float32, device=self.device) for type in
             self.agent_type], dim=1)

        cent_obs = cent_obs.reshape(-1, self.n_agent_all, self.state_dim)
        obs = obs.reshape(-1, self.n_agent_all, self.observation_dim)
        available_actions = available_actions.reshape(-1, self.n_agent_all, self.act_dim)

        cols = None
        if random_seq:
            rows, cols = self._shuffle_agent_grid(obs.size(0), self.n_agent_all, self.random_permutation)

            if self.permutation_strategy != '':
                cols = self.agent_sequence
                cols = np.expand_dims(cols, axis=0).repeat(obs.size(0), axis=0)
            if permutation_sequence is not None and self.test:
                cols = permutation_sequence
            cent_obs = cent_obs[rows, cols]
            obs = obs[rows, cols]
            available_actions = available_actions[rows, cols]
            if memory is not None:
                memory = memory[rows, cols]

        t_emb, graph, adj = None, None, None
        # if self.use_gcn:
        #     # nodes = torch.cat(
        #     #     [torch.as_tensor(state['Nodes_' + type], dtype=torch.float32, device=self.device) for type in
        #     #      self.agent_type], dim=1)
        #     # edges = torch.cat(
        #     #     [torch.as_tensor(state['Edges_' + type], dtype=torch.float32, device=self.device) for type in
        #     #      self.agent_type], dim=1)
        #     # graph = [nodes, edges]
        #     nodes = torch.as_tensor(state['Nodes'], dtype=torch.float32, device=self.device)
        #     edges = torch.as_tensor(state['Edges'], dtype=torch.float32, device=self.device)
        #     graph = [nodes, edges]

        # if self.use_hgcn:
        #     adj = {key: torch.as_tensor(state[key], dtype=torch.float32, device=self.device) for key in
        #            adjacent_name_list}

        # if task_emb is not None: t_emb = torch.cat(
        #     [torch.as_tensor(task_emb[type], dtype=torch.float32, device=self.device) for type in self.agent_type],
        #     dim=1)

        with torch.no_grad():
            actions, action_log_probs, _, new_memory = self.transformer.get_actions(cent_obs,
                                                                                    obs,
                                                                                    available_actions,
                                                                                    deterministic, task_emb=t_emb,
                                                                                    graph_inputs=graph,
                                                                                    hgcn_inputs=adj, memory=memory)
            #added by zf: actions的shape是[batch_size, n_agent, 1]
        if random_seq:
            actions = actions.squeeze(-1)
            action_log_probs = action_log_probs.squeeze(-1)
            original_order_rows = rows
            original_order_cols = np.argsort(cols, axis=1)

            restored_actions = np.take_along_axis(actions, original_order_rows, axis=0)
            restored_actions = np.take_along_axis(restored_actions, original_order_cols, axis=1)

            restored_logprob = np.take_along_axis(action_log_probs, original_order_rows, axis=0)
            restored_logprob = np.take_along_axis(restored_logprob, original_order_cols, axis=1)

            actions = restored_actions[:, :, None]
            action_log_probs = restored_logprob[:, :, None]

            if new_memory is not None:
                restored_memorys = []
                for i in range(self.n_agent_all):
                    restored_memorys.append(new_memory[:, cols[0, i]])
                new_memory = torch.stack(restored_memorys, axis=1)

        actions_final = {}
        log_p_final = {}
        agent_count = 0
        for type in self.agent_type:
            actions_final[type] = actions[:, agent_count:agent_count + self.n_agent[type]]
            log_p_final[type] = action_log_probs[:, agent_count:agent_count + self.n_agent[type]]
            agent_count += self.n_agent[type]
        #print("actions_final:", actions_final)
       # print("action_final shape:", actions_final['uav'].shape)
       #print("carrier action shape:", actions_final['carrier'].shape)
        return actions_final, log_p_final, cols, new_memory
        #shape是，actions_final:{'uav': [batch_size, n_agent, 1]}, log_p_final:{'uav': [batch_size, n_agent, 1]}, cols:None, new_memory:None
        #[61 , 4 ,1 ]
    def evaluate_actions(self, cent_obs, obs, actions, available_actions=None, task_emb=None, graph=None, adj=None,
                         memory=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param actions: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        cent_obs = cent_obs.reshape(-1, self.n_agent_all, self.state_dim)
        obs = obs.reshape(-1, self.n_agent_all, self.observation_dim)
        if actions.shape[-1] == self.act_dim:
            actions = actions.reshape(-1, self.n_agent_all, self.act_dim)
            # last_dim = len(actions.shape) - 1
            # actions = torch.argmax(actions, dim=last_dim).long().unsqueeze(last_dim)
        else:
            actions = actions.reshape(-1, self.n_agent_all, 1)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.n_agent_all, self.act_dim)

        action_log_probs, values, entropy, new_memory, distri = self.transformer(cent_obs, obs, actions,
                                                                                 available_actions,
                                                                                 task_emb, graph,
                                                                                 adj, memory)
        if action_log_probs is not None:
            action_log_probs = action_log_probs.view(-1, self.n_agent_all, 1)
        values = values.view(-1, self.n_agent_all, 1)
        entropy = entropy.view(-1, self.n_agent_all, 1)
        entropy = entropy.mean()

        return values, action_log_probs, entropy, distri

    def get_values(self, cent_obs, obs, available_actions=None, task_emb=None, graph=None, adj=None, memory=None):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """

        cent_obs = cent_obs.reshape(-1, self.n_agent_all, self.state_dim)
        obs = obs.reshape(-1, self.n_agent_all, self.observation_dim)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.n_agent_all, self.act_dim)

        values = self.transformer.get_values(cent_obs, obs, available_actions, task_emb, graph, adj, memory=memory)

        values = values.view(-1, self.n_agent_all, 1)

        return values

    def updateAgent(self, trajs, clip=None):
        current_step = trajs['step']
        del trajs['step']
        time_t = time.time()
        if clip is None:
            clip = self.clip
        n_minibatch = self.n_minibatch

        names = Trajectory.names()  # ['s', 'a', 'r', 's1', 'd', 'logp']

        names.remove('h')
        if not self.use_mate:
            for n in ['ae_h', 't_emb', 'a0', 'r0']:
                if n in names: names.remove(n)
        if not self.use_lambda:
            for n in ['cost']:
                if n in names: names.remove(n)

        if not self.use_gcn:
            for n in ['nodes', 'edges']:
                if n in names: names.remove(n)

        if not self.use_hgcn:
            for n in adjacent_name_list:
                if n in names: names.remove(n)

        if not self.random_permutation:
            for n in ['cols']:
                if n in names: names.remove(n)

        if not self.use_temporal_type or not self.use_hgcn:
            for n in ['memory']:
                if n in names: names.remove(n)

        traj_type_all = {}
        for type in self.agent_type:
            traj_all = {name: [] for name in names}

            for traj in trajs[type]:  # len(trajs) == n_thread
                for name in names:
                    traj_all[name].append(traj[name])
            traj_type_all[type] = {name: torch.stack(value, dim=0) for name, value in traj_all.items()}

        traj = {}
        for name in names:
            if name in ['memory', 'nodes', 'edges'] + adjacent_name_list:
                traj[name] = traj_type_all['carrier'][name]
            else:
                traj[name] = torch.cat([traj_type_all[type][name] for type in self.agent_type], dim=2)

        share_s, s, a, r, share_s1, s1, d, a_mask, logp = traj['share_s'], traj['s'], traj['a'], traj['r'], traj[
            'share_s1'], traj['s1'], traj['d'], traj['a_mask'], traj['logp']
        share_s, s, a, r, share_s1, s1, d, a_mask, logp = [item.to(self.device) for item in
                                                           [share_s, s, a, r, share_s1, s1, d, a_mask, logp]]

        if self.use_mate:
            ae_h, t_emb, a0, r0 = [traj[key].to(self.device) for key in ['ae_h', 't_emb', 'a0', 'r0']]
        else:
            ae_h, t_emb, a0, r0 = [None for _ in range(4)]

        if self.use_gcn:
            nodes, edges = [traj[key].to(self.device) for key in ['nodes', 'edges']]
        else:
            nodes, edges = None, None

        if self.use_hgcn:
            adj = {key: traj[key] for key in adjacent_name_list}
            self.adj = adj
        else:
            adj = None

        if self.use_temporal_type and self.use_hgcn:
            memory = traj['memory'].to(self.device)
            memory = memory.view(-1, self.n_agent_all, self.n_embd, 2, self.hgcn_rollout_len)
        else:
            memory = None

        # 关键：all in shape [n_thread, T, n_agent, dim]

        value_old, returns, advantages, _ = self._process_traj(traj['share_s'], traj['s'], traj['a'], traj['r'],
                                                               traj['share_s1'], traj['s1'], traj['d'], traj['logp'],
                                                               traj.get('h', None), traj.get('ae_h', None),
                                                               traj.get('t_emb', None), traj.get('a0', None),
                                                               traj.get('r0', None), traj.get('nodes', None),
                                                               traj.get('edges', None), adj,
                                                               traj.get('memory', None),
                                                               traj.get('a_mask', None))  # 不同traj分开计算adv和return
        advantages_old = advantages
        # self.logger.log(**{"adv_avg": advantages.mean().item()})
        _, T, n, d_s = s.size()
        d_a = a.size()[-1]
        s = s.view(-1, n, d_s)
        s1 = s1.view(-1, n, d_s)
        d = d.view(-1, n, 1)
        share_s = share_s.view(-1, n, self.state_dim)
        a = a.view(-1, n, d_a)
        logp = logp.view(-1, n, d_a)
        advantages_old = advantages_old.view(-1, n, 1)
        returns = returns.view(-1, n, 1)
        value_old = value_old.view(-1, n, 1)
        a_mask = a_mask.view(-1, n, self.act_dim)
        # 关键：s, a, logp, adv, ret, v are now all in shape [-1, n_agent, dim] 因为计算完adv和return后可以揉在一起做mini_batch训练
        batch_total = logp.size()[0]
        batch_size = int(batch_total / n_minibatch)

        if self.use_mate:
            t_emb = t_emb.view(-1, n, self.task_emb_dim * 2)
            ae_h = ae_h.view(-1, n, ae_h.size()[-1])
            a0 = a0.view(-1, n)
            r0 = r0.view(-1, n)

        if self.use_gcn:
            nodes = nodes.view(-1, self.gcn_nodes_num, 3)
            edges = edges.view(-1, 2, edges.size()[-1])
            # edges = edges.view(-1,n,self.gcn_nodes_num,self.gcn_nodes_num)

        if self.use_hgcn:
            self.reshape_adjacent_mat(adj)

        if self.random_permutation:
            cols = traj['cols'].view(-1, 2 * n)[:, :n].long()

        train_info = {'value_loss': 0, 'policy_loss': 0, 'dist_entropy': 0,
                      'critic_grad_norm': 0, 'ratio': 0}

        loss_dict = {}
        for iter in range(self.n_update_pi):
            for n_mini in range(self.n_minibatch):
                batch_task_emb, batch_nodes, batch_edges, batch_memory = None, None, None, None
                if n_minibatch > 1:
                    idxs = np.random.choice(range(batch_total), size=batch_size, replace=False)
                    [batch_share_obs, batch_state, batch_action, batch_logp, batch_advantages_old, batch_state_next,
                     batch_done] = [item[idxs] for item in [share_s, s, a, logp, advantages_old, s1, d]]
                    [batch_value, batch_returns, batch_a_masks] = [item[idxs] for item in [value_old, returns, a_mask]]
                    batch_nodes, batch_edges, batch_adj, batch_cols, batch_memory = None, None, None, None, None
                    if self.use_gcn:
                        batch_nodes = nodes[idxs]
                        batch_edges = edges[idxs]

                    if self.use_hgcn:
                        batch_adj = {key: value[idxs] for key, value in adj.items()}

                    if self.use_temporal_type and self.use_hgcn:
                        batch_memory = memory[idxs]

                    if self.random_permutation:
                        batch_cols = cols[idxs]

                else:
                    [batch_share_obs, batch_state, batch_action, batch_logp, batch_advantages_old, batch_state_next,
                     batch_done] = [item for item in [share_s, s, a, logp, advantages_old, s1, d]]
                    [batch_value, batch_returns, batch_a_masks] = [item for item in [value_old, returns, a_mask]]
                    batch_nodes, batch_edges, batch_adj, batch_cols, batch_memory = None, None, None, None, None

                    if self.use_gcn:
                        batch_nodes = nodes
                        batch_edges = edges

                    if self.use_hgcn:
                        batch_adj = {key: value for key, value in adj.items()}

                    if self.use_temporal_type and self.use_hgcn:
                        batch_memory = memory

                    if self.random_permutation:
                        batch_cols = cols

                sample = (batch_share_obs, batch_state, batch_action,
                          batch_value, batch_returns, batch_logp,
                          batch_advantages_old, batch_a_masks,
                          batch_task_emb, batch_nodes, batch_edges, batch_adj, batch_memory)

                B, N, _ = batch_state.size()
                rows, tmp_cols = self._shuffle_agent_grid(B, N, self.random_permutation)
                if not self.random_permutation:
                    batch_cols = tmp_cols
                if self.permutation_strategy != '':
                    if self.act_update_seq_diff:
                        my_sequence = np.random.permutation(self.n_agent_all)
                    else:
                        my_sequence = self.agent_sequence
                    batch_cols = np.expand_dims(my_sequence, axis=0).repeat(B, axis=0)
                batch_share_obs, batch_state, batch_action, batch_value, batch_returns, batch_logp, batch_advantages_old, batch_a_masks = \
                    [item[rows, batch_cols] for item in
                     [batch_share_obs, batch_state, batch_action, batch_value, batch_returns, batch_logp,
                      batch_advantages_old, batch_a_masks]]
                if batch_memory is not None:
                    batch_memory = batch_memory[rows, batch_cols]

                if self.use_sequential_update:
                    factor = torch.ones((B, 1), dtype=torch.float64, device=batch_state.device)
                    imp_weights = []
                    for agent_idx in range(self.n_agent_all):
                        old_log_probs = batch_logp
                        sample = (batch_share_obs, batch_state, batch_action, batch_value, batch_returns, batch_logp,
                                  batch_advantages_old, batch_a_masks, batch_task_emb, batch_nodes, batch_edges,
                                  batch_adj, batch_memory, factor)

                        value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights_agent = self.ppo_update(
                            sample, index=agent_idx)

                        _, new_log_probs, _, _ = self.evaluate_actions(batch_share_obs,
                                                                       batch_state,
                                                                       batch_action,
                                                                       batch_a_masks,
                                                                       task_emb=None, graph=[batch_nodes, batch_edges],
                                                                       adj=batch_adj, memory=batch_memory)
                        with torch.no_grad():
                            factor = factor * torch.exp(new_log_probs - old_log_probs)[:, agent_idx]
                            eps = 0.05
                            factor = torch.clamp(factor, min=1 - eps, max=1 + eps)

                        imp_weights.append(imp_weights_agent)
                        train_info['value_loss'] += value_loss.item()
                        train_info['policy_loss'] += policy_loss.item()
                        train_info['dist_entropy'] += dist_entropy.item()
                        # train_info['actor_grad_norm'] += actor_grad_norm
                        train_info['critic_grad_norm'] += critic_grad_norm
                        train_info['ratio'] += imp_weights_agent.mean()

                    imp_weights = torch.stack(imp_weights, dim=1)

                else:
                    sample = (batch_share_obs, batch_state, batch_action, batch_value, batch_returns, batch_logp,
                              batch_advantages_old, batch_a_masks, batch_task_emb, batch_nodes, batch_edges, batch_adj,
                              batch_memory, None)

                    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.ppo_update(
                        sample)

                    # if self.rep_learning and iter < self.rep_iter:
                    #     loss_dict = self.transformer.update_rep(batch_state, batch_action, batch_returns, batch_state_next,
                    #                                             batch_done, graph_inputs=[batch_nodes, batch_edges])

                    train_info['value_loss'] += value_loss.item()
                    train_info['policy_loss'] += policy_loss.item()
                    train_info['dist_entropy'] += dist_entropy.item()
                    # train_info['actor_grad_norm'] += actor_grad_norm
                    train_info['critic_grad_norm'] += critic_grad_norm
                    train_info['ratio'] += imp_weights.mean()

        for k in train_info.keys():
            train_info[k] /= self.n_update_pi * self.num_mini_batch

        if self.permutation_strategy != '':
            self.agent_sequence = self.update_order(advantages, value_old, imp_weights, current_step)
        self.logger.log(**{
            f'Seq/order_{n}': self.agent_sequence[n] for n in range(self.n_agent_all)
        })

        if self.lr_scheduler in ['cos', 'linear']:
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.current_lr = current_lr
            self.logger.log({'lr': current_lr})

        train_info.update(loss_dict)
        # print(f"Final Grad:{train_info['actor_grad_norm']}")
        self.logger.log(**train_info)
        self.logger.log(agent_update_time=time.time() - time_t)
        return [r.mean().item(), train_info['dist_entropy'], train_info['ratio']]

    def update_order(self, advantages, value_old, imp_weights, current_step=-1):
        if "fixed" in self.permutation_strategy:
            return np.arange(self.n_agent_all)
        if "eps" in self.permutation_strategy:
            eps = np.random.rand()
            if self.decay_eps:
                factor = self.current_lr / self.lr
            else:
                factor = 1.0
            # print(factor)
            if eps < self.permutation_eps * factor:
                return np.random.permutation(self.n_agent_all)
        if "random" in self.permutation_strategy:
            return np.random.permutation(self.n_agent_all)
        if "cyclic" in self.permutation_strategy:
            return np.flip(np.arange(self.n_agent_all))
        if "greedy" in self.permutation_strategy or "ucb" in self.permutation_strategy:
            adv_s = advantages.reshape(-1, *advantages.shape[2:])  # (B,N,1)
            value_preds = value_old.reshape(-1, *value_old.shape[1:])  # B,N,1
        if "_r" in self.permutation_strategy:
            ratios = imp_weights.reshape(-1, *imp_weights.shape[1:])
            adv_s *= ratios
        adv_s = adv_s.detach().cpu().numpy()
        value_preds = value_preds.cpu().numpy()
        score = np.abs(adv_s / (value_preds + self.deno_eps))
        # a single update within 30 timestep?
        score = np.mean(score, axis=0)  # N,1
        score = np.sum(score, axis=score.shape[1:])  # N
        if "ucb" in self.permutation_strategy:
            extra_adv = self.confidence * np.sqrt(np.log(current_step[-1]) / self.permute_count)
            # print(score)
            score += extra_adv
            # print(extra_adv)
        id_scores = [(_i, _s) for (_i, _s) in zip(range(self.n_agent_all), score)]
        to_reverse = not ("reverse" in self.permutation_strategy)
        id_scores = sorted(id_scores, key=lambda i_s: i_s[1], reverse=to_reverse)

        if "greedy" in self.permutation_strategy:
            if "semi" in self.permutation_strategy:
                # other strategies
                order = []
                a_i = 0
                while a_i < self.n_agent_all:
                    order.append(id_scores[0][0])
                    id_scores.pop(0)
                    a_i += 1
                    if len(id_scores) > 0:
                        next_i = np.random.choice(len(id_scores))
                        order.append(id_scores[next_i][0])
                        id_scores.pop(next_i)
                        a_i += 1
                order = np.array(order)
            else:
                order = np.array([i_s[0] for i_s in id_scores])
            if "ucb" in self.permutation_strategy:
                for weight, item in zip(reversed(range(self.n_agent_all)), order):
                    self.permute_count[item] += weight
            return order

    def _shuffle_agent_grid(self, batch_size, nb_agent, random=False):
        x = batch_size
        y = nb_agent
        rows = np.indices((x, y))[0]
        if random:
            cols = np.stack([np.random.permutation(y) for _ in range(x)])
        else:
            cols = np.stack([np.arange(y) for _ in range(x)])
        return rows, cols

    def cal_order(self, pis, states, obs, max_t, graph_inputs, n_thread):
        """
        generate order by integrated gradient. agent contributes most to the gradient will be executed first.
        """
        unrolled_paths, full_step_size = self.unroll_path(pis, states, obs, graph_inputs[0], graph_inputs[1])
        flat_unrolled = []
        for item in unrolled_paths:
            flat_unrolled.append(torch.flatten(item, start_dim=0, end_dim=1))
        grad = self.get_integrated_gradients(flat_unrolled)
        grad = grad.view(full_step_size.shape) * full_step_size
        agent_grad_list = []
        for _ in range(max_t):
            agent_grad = grad.sum(1)
            agent_grad_list.append(agent_grad)
        agent_grad = torch.stack(agent_grad_list, dim=1).sum(-1)
        _, order = agent_grad.sort(descending=True, dim=-1)
        return order

    def unroll_path(self, *paths):
        def unroll_single_path(vector) -> list:
            """
            Expand vector transformation into smallet step, according to each_ig_step_num
            """
            vector_list = vector.split(1, dim=1)
            unrolled_vector_path = []
            # discretize vector transformation into steps.
            for vector_pos, vector_next_pos in zip(vector_list[:-1], vector_list[1:]):
                vector_pos, vector_next_pos = vector_pos.squeeze(1), vector_next_pos.squeeze(1)
                vector_step_sizes = (vector_next_pos - vector_pos) / self.each_ig_step_num
                each_unrolled_vector_path = [vector_step_sizes * i_step + vector_pos for i_step in
                                             range(self.each_ig_step_num)]
                unrolled_vector_path += each_unrolled_vector_path
            unrolled_vector_path.append(vector_list[-1].squeeze(1))
            unrolled_vector_path = torch.stack(unrolled_vector_path, dim=1)
            step_sizes = unrolled_vector_path[:, :-1] - unrolled_vector_path[:, 1:]
            unrolled_vector_path = unrolled_vector_path[:, 0:-1]
            return unrolled_vector_path, step_sizes

        unrolled_paths = []
        first = True
        pi_step_sizes = None
        for item in paths:
            result, step_sizes = unroll_single_path(item)
            if first:
                pi_step_sizes = step_sizes
                first = False
            unrolled_paths.append(result)

        return unrolled_paths, pi_step_sizes

    def get_integrated_gradients(self, unrolled_paths: list):
        pi_path = torch.autograd.Variable(unrolled_paths[0].detach(),
                                          requires_grad=True)
        unrolled_paths.pop(0)
        for i in range(len(unrolled_paths)):
            unrolled_paths[i] = torch.autograd.Variable(unrolled_paths[i].detach(), requires_grad=False)
        _, action_probs, _, _ = self.evaluate_actions(unrolled_paths[0], unrolled_paths[1], pi_path,
                                                graph=[unrolled_paths[2], unrolled_paths[3]])
        action_probs.mean().backward()
        return pi_path.grad

    def _process_traj(self, share_s, s, a, r, share_s1, s1, d, logp, h=None, ae_h=None, t_emb=None, a0=None, r0=None,
                      nodes=None, edges=None, adj=None, memory=None, a_mask=None):
        # 过网络得到value_old， 使用GAE计算adv和return
        """
        Input are all in shape [batch_size, T, n_agent, dim]
        """
        b, T, n, dim_s = s.shape
        s, a, r, s1, d, logp = [item.to(self.device) for item in [s, a, r, s1, d, logp]]
        if self.use_mate:
            b_e, T_e, n_e, dim_e = t_emb.shape
            t_emb = t_emb.to(self.device).view(-1, T, n_e, dim_e)
            t_emb_all = t_emb.view(-1, n_e, dim_e)
        else:
            t_emb_all = None
        if self.use_gcn:
            nodes = nodes.to(self.device).view(-1, T, self.gcn_nodes_num, 3)
            edges = edges.to(self.device).view(-1, T, 2, edges.size()[-1])
            nodes_all = nodes.view(-1, self.gcn_nodes_num, 3)
            edges_all = edges.view(-1, 2, edges.size()[-1])
            graph_all = [nodes_all, edges_all]
        else:
            graph_all = None

        if self.use_temporal_type and self.use_hgcn:
            memory = memory.to(self.device).view(-1, T, n, self.n_embd, 2, self.hgcn_rollout_len)
            memory_all = memory.view(-1, n, self.n_embd, 2, self.hgcn_rollout_len)
            memory_next = memory.select(1, T - 1)
        else:
            memory, memory_all, memory_next = None, None, None

        if self.use_hgcn:
            adj_all = copy.deepcopy(adj)
            self.reshape_adjacent_mat(adj_all)
            adj_next = {key: value.select(1, T - 1) for key, value in adj.items()}
        else:
            adj_all = None
            adj_next = None
            # edges_all = edges.to(self.device).view(-1,n,self.gcn_nodes_num,self.gcn_nodes_num)
        # 过网络前先merge前两个维度，过网络后再复原
        value = self.get_values(share_s.view(-1, n, self.state_dim), s.view(-1, n, self.observation_dim),
                                task_emb=t_emb_all, graph=graph_all, adj=adj_all, memory=memory_all)
        value = value.view(b, T, n, -1)

        returns = torch.zeros(value.size(), device=self.device)
        deltas, advantages = torch.zeros_like(returns), torch.zeros_like(returns).squeeze(-1)

        if self.use_gcn:
            graph_next = [copy.deepcopy(nodes.select(1, T - 1)).contiguous(),
                          copy.deepcopy(edges.select(1, T - 1)).contiguous()]
        else:
            graph_next = None
        # t_emb_next = {key:torch.zeros((b,n,self.task_emb_dim)) for key in all_type} if type is None else torch.zeros((b,n,self.task_emb_dim))
        if self.use_mate:
            t_emb_next, _ = self.encode(s1.select(1, T - 1), a.select(1, T - 1).squeeze(-1),
                                        r.select(1, T - 1).squeeze(-1), ae_h.select(1, T - 1), no_grads=True,
                                        graph_inputs=graph_next)
        else:
            t_emb_next = None

        d_mask = d.float()
        if self.credit_assign:
            n_thread, T, n, d_s = s.size()
            d_a = a.size()[-1]
            s_all = s.view(-1, n, d_s)
            a_all = a.view(-1, n, d_a)
            share_s_all = share_s.view(-1, n, self.state_dim)
            values, _, _, distri = self.evaluate_actions(share_s_all, s_all, a_all, a_mask.view(-1, n, self.act_dim),
                                                         task_emb=None, graph=[nodes_all, edges_all],
                                                         adj=adj_all, memory=memory_all)
            policy_prob_out = distri.probs.view(n_thread, T, n, -1)
            policy_prob_out[a_mask == 0] = 0
            policy_prob_out = policy_prob_out / policy_prob_out.sum(dim=-1, keepdim=True)
            policy_prob_out[a_mask == 0] = 0
            order = self.cal_order(policy_prob_out, share_s, s, T,
                                   graph_inputs=[nodes, edges], n_thread=n_thread)
            _, F1, _, _ = self.evaluate_actions(share_s_all, s_all, policy_prob_out, None,
                                                task_emb=None, graph=[nodes_all, edges_all],
                                                adj=adj_all, memory=memory_all)
            tmp = policy_prob_out.clone()
            one_hot_action = F.one_hot(a.squeeze(-1))
            for i in range(self.n_agent_all):
                index = order[:, :, i:i + 1].clone().unsqueeze(-1).repeat(1, 1, 1, self.action_dim[0])
                tmp = tmp.clone()
                # The action of selected agent in the order
                onehot_action_gather = torch.gather(one_hot_action, dim=2, index=index)
                tmp = tmp.scatter(dim=2, index=index, src=onehot_action_gather.float())
                # The value of taking a particular action
                _, F2, _, _ = self.evaluate_actions(share_s_all, s_all, torch.flatten(tmp, start_dim=0, end_dim=1),
                                                    a_mask.view(-1, n, self.act_dim),
                                                    task_emb=None, graph=[nodes_all, edges_all],
                                                    adj=adj_all, memory=memory_all)
                F2 = F2.view(F1.shape)
                adv_tmp = (F2 - F1).clone().detach().squeeze(-1).view(n_thread, T, -1)  # Q(s,a) - V(s)
                advantages = advantages.scatter(dim=2, index=order[:, :, i:i + 1], src=adv_tmp)
                F1 = F2.clone()
            advantages = advantages.unsqueeze(-1)
            with torch.no_grad():
                prev_return = torch.zeros((n_thread, T, 1))
                for t in reversed(range(T)):
                    if self.use_gae_returns:
                        returns[:, t, :, :] = value.select(1, t).detach() + advantages.select(1, t)
                    else:
                        returns[:, t, :, :] = r.select(1, t) + self.gamma * (1 - d_mask.select(1, t)) * prev_return
                    prev_return = returns.select(1, t)
        else:
            with torch.no_grad():
                advantages = advantages.unsqueeze(-1)
                prev_value = self.get_values(share_s1.select(1, T - 1), s1.select(1, T - 1), task_emb=t_emb_next,
                                             graph=graph_next, adj=adj_next, memory=memory_next)
                if not self.use_rtg:
                    prev_return = prev_value
                else:
                    prev_return = torch.zeros_like(prev_value)
                prev_advantage = torch.zeros_like(prev_return)

                for t in reversed(range(T)):
                    deltas[:, t, :, :] = r.select(1, t) + self.gamma * (
                            1 - d_mask.select(1, t)) * prev_value - value.select(1,
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

    def reshape_adjacent_mat(self, adj_all):
        adj_all['uav-carrier'] = adj_all['uav-carrier'].view(-1, self.n_agent['uav'], self.n_agent['carrier'])
        adj_all['uav-poi'] = adj_all['uav-poi'].view(-1, self.n_agent['uav'], self.n_poi)
        adj_all['uav-epoi'] = adj_all['uav-epoi'].view(-1, self.n_agent['uav'], self.n_poi)
        adj_all['uav-road'] = adj_all['uav-road'].view(-1, self.n_agent['uav'], self.n_node)
        adj_all['carrier-uav'] = adj_all['carrier-uav'].view(-1, self.n_agent['carrier'], self.n_agent['uav'])
        adj_all['carrier-poi'] = adj_all['carrier-poi'].view(-1, self.n_agent['carrier'], self.n_poi)
        adj_all['carrier-epoi'] = adj_all['carrier-epoi'].view(-1, self.n_agent['carrier'], self.n_poi)
        adj_all['carrier-road'] = adj_all['carrier-road'].view(-1, self.n_agent['carrier'], self.n_node)
        adj_all['poi-uav'] = adj_all['poi-uav'].view(-1, self.n_poi, self.n_agent['uav'])
        adj_all['poi-carrier'] = adj_all['poi-carrier'].view(-1, self.n_poi, self.n_agent['carrier'])
        adj_all['road-carrier'] = adj_all['road-carrier'].view(-1, self.n_node, self.n_agent['carrier'])

    def zero_grad(self):
        for type in self.agent_type:
            self.mate[type].zero_grad()
        return

    def save_nets(self, dir_name, iter=0, is_newbest=False):
        if not os.path.exists(dir_name + '/Models'):
            os.mkdir(dir_name + '/Models')
        prefix = 'best' if is_newbest else str(iter)
        torch.save(self.transformer.state_dict(), dir_name + '/Models/' + prefix + '_actor.pt')
        if self.use_mate:
            for type in self.agent_type:
                self.mate[type].save(dir_name + '/Models/' + prefix + '_mate_' + type + '.pt')
        print('RL saved successfully')

    def load_nets(self, dir_name, iter=0, best=False):
        prefix = 'best' if best else str(iter)
        self.transformer.load_state_dict(torch.load(dir_name + '/Models/' + prefix + '_actor.pt'))
        if self.use_mate:
            for type in self.agent_type:
                self.mate[type].restore(dir_name + '/Models/' + prefix + '_mate_' + type + '.pt')
        print('load networks successfully')

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        values, value_preds_batch, return_batch = [item.view(-1, 1) for item in
                                                   [values, value_preds_batch, return_batch]]
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        # if self._use_value_active_masks and not self.dec_actor:
        value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, index=None):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        :return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, actions_batch, \
            value_preds_batch, return_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, \
            task_emb, nodes, edges, adj, memory, factor = sample

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy, _ = self.evaluate_actions(share_obs_batch,
                                                                          obs_batch,
                                                                          actions_batch,
                                                                          available_actions_batch,
                                                                          task_emb=task_emb, graph=[nodes, edges],
                                                                          adj=adj,
                                                                          memory=memory)

        if index is not None:
            action_log_probs = action_log_probs[:, index]
            old_action_log_probs_batch = old_action_log_probs_batch[:, index]
            adv_targ = adv_targ[:, index]
            values = values[:, index]
            value_preds_batch = value_preds_batch[:, index]
            return_batch = return_batch[:, index]

        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if factor is not None:
            policy_loss = -torch.sum(factor * torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)

        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef
        # loss = value_loss
        self.optimizer.zero_grad()
        loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_grad_norm(self.transformer.parameters())

        self.optimizer.step()

        return value_loss, grad_norm, policy_loss, dist_entropy, grad_norm, imp_weights

    def prep_training(self):
        self.transformer.train()

    def prep_rollout(self):
        self.transformer.eval()
