import os
import os.path as osp
from datetime import datetime
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from source_code.algorithms.utils import collect, mem_report
from tqdm.std import trange
# from algorithms.algorithm import ReplayBuffer
from gym.spaces.box import Box
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import pickle
from copy import deepcopy as dp
from source_code.algorithms.models import CategoricalActor
import random
import multiprocessing as mp
# import torch.multiprocessing as mp
from torch import distributed as dist
from itertools import permutations
from source_code.algorithms.algo.buffer import MultiCollect, Trajectory, TrajectoryBuffer, adjacent_name_list
from source_code.LaunchMCS.launch_mcs import exclude_names
from source_code.algorithms.algo.agent.PoIAgent import PoIAgent

class OnPolicyRunner:
    def __init__(self, logger, agent, envs_learn, envs_test, dummy_env,
                 run_args, alg_args, input_args, **kwargs):
        self.run_args = run_args
        self.input_args = input_args
        self.alg_args = alg_args
        self.debug = self.run_args.debug
        self.logger = logger
        self.name = run_args.name
        # agent initialization
        self.agent = agent
        self.agent_type = agent.agent_type
        self.num_agent = agent.n_agent
        self.device = self.agent.device if hasattr(self.agent, "device") else "cpu"
        
        self.poi_decision_mode = input_args.poi_decision_mode
        if self.poi_decision_mode:
            self.poi_agent = PoIAgent(logger,self.agent.device,self.agent.agent_args,input_args)
            self.agent_type = agent.agent_type + self.poi_agent.agent_type
            poi_num_agent =  {'poi_'+key:value for key,value in self.num_agent.items()}
            self.num_agent = {**poi_num_agent,**self.num_agent}
        else:
            self.poi_agent = None
            
        

        if run_args.checkpoint is not None and input_args.test:  # not train from scratch
            self.agent.load_nets(run_args.checkpoint, best=True)
            # logger.log(interaction=run_args.start_step)
        if run_args.checkpoint is not None:  # not train from scratch
            self.agent.load_nets(run_args.checkpoint, iter=input_args.model_iter)
        self.start_step = run_args.start_step
        self.env_name = input_args.env
        self.algo_name = input_args.algo
        self.n_thread = input_args.n_thread

        self.best_episode_reward = float('-inf')
        self.best_test_episode_reward = float('-inf')

        # algorithm arguments
        self.n_iter = alg_args.n_iter
        self.n_inner_iter = alg_args.n_inner_iter
        self.n_warmup = alg_args.n_warmup if not self.run_args.debug else 1
        self.n_model_update = alg_args.n_model_update
        self.n_model_update_warmup = alg_args.n_model_update_warmup if not self.run_args.debug else 1
        self.n_test = alg_args.n_test
        self.test_interval = alg_args.test_interval
        self.rollout_length = alg_args.rollout_length
        self.use_stack_frame = alg_args.use_stack_frame

        # environment initialization
        self.envs_learn = envs_learn
        self.envs_test = envs_test
        self.dummy_env = dummy_env
        # 一定注意，PPO并不是在每次调用rollout时reset，一次rollout和是否reset没有直接对应关系
        _, self.episode_len = self.envs_learn.reset(), 0
        # 每个环境分别记录episode_reward
        self.episode_reward = np.zeros((self.input_args.n_thread))

        self.use_lambda = input_args.use_lambda
        self.use_mate = input_args.use_mate
        self.ae_hiddens, self.a, self.r = None, None, None
        self.random_permutation = input_args.random_permutation

        self.random_map = input_args.random_map

        setSeed(run_args.seed)

    def run(self):  # 被launcher.py调用的主循环

        if self.run_args.test:
            self.test(1)
            return

        self.routine_count = 0
        self.rr = 0
        for iter in trange(self.n_iter, desc='rollout env'):
            if iter % 100 == 0:
                self.test(iter)
            if iter % 1000 == 0:
                self.agent.save_nets(dir_name=self.run_args.output_dir, iter=iter)  # routine
                if self.poi_decision_mode: self.poi_agent.save_nets(dir_name=self.run_args.output_dir, iter=iter)
            trajs = self.rollout_env(iter)

            agentInfo = []
            for inner in trange(self.n_inner_iter, desc='inner-iter updateAgent'):
                if self.use_mate:
                    self.agent.zero_grad()
                info = self.agent.updateAgent(trajs)
                if self.poi_decision_mode:  self.poi_agent.updateAgent(trajs)
                agentInfo.append(info)   
            if self.use_mate:
                self.agent.update_encoder(trajs)

            if iter % 20 == 0:
                self.agent.log_model()
                if self.poi_decision_mode: self.poi_agent.log_model()
            self.logger.log(inner_iter=inner + 1, iter=iter)

    def test(self, iter=1):
        """
        The environment should return sth like [n_agent, dim] or [batch_size, n_agent, dim] in either numpy or torch.
        """
        returns = []
        lengths = []
        save_emb = True if self.run_args.test else False
        permutation = False
        output_strings = []
        if permutation:
            cols_all = np.array(list(permutations(list(range(sum(self.num_agent.values()))))))
            self.n_test = len(cols_all)
        uav_pois = []
        ugv_pois = []
        for i in trange(self.n_test, desc='test'):
            done, ep_ret, ep_len = False, np.zeros((1,)), 0  # ep_ret改为分threads存
            order = np.expand_dims(cols_all[i], axis=0) if permutation else None
            memory = None
            if self.use_mate:
                a = {key: np.zeros((1, self.num_agent[key])) for key in self.agent_type}
                r = {key: np.zeros((1, self.num_agent[key])) for key in self.agent_type}
                ae_hiddens = {key: np.zeros((1, self.num_agent[key], self.agent.mate[key].hidden_dims()[0])) for key in
                              self.agent_type}
            
            envs = self.envs_test
            envs.reset()
            while not done:  # 测试时限定一个episode最大为length步
                s = envs.get_obs_from_outside()
                if self.use_mate:
                    task_embs, ae_hiddens = self.agent.encode(s, a, r, ae_hiddens, no_grads=True)
                else:
                    task_embs = None
         
                if self.input_args.algo == 'CPPO':
                    a, logp, cols, memory = self.agent.act(s, task_embs, random_seq=False, deterministic=True,
                                                           memory=memory)
                else:
                    dist = self.agent.act(s, task_embs)
                    a = {}
                    logp = {}
                    for type in self.agent_type:
                        if 'poi' in type : continue
                        a_tmp = dist[type].sample()
                        logp_tmp = dist[type].log_prob(a_tmp)
                        a_tmp = a_tmp.detach().cpu().numpy()
                        a[type] = a_tmp
                        logp[type] = logp_tmp
                        
                if self.poi_decision_mode:
                    dist_poi = self.poi_agent.act(s,task_embs)
                    for type in self.poi_agent.agent_type:
                        a_tmp = dist_poi[type].sample()
                        logp_tmp = dist_poi[type].log_prob(a_tmp)
                        a_tmp = a_tmp.detach().cpu().numpy()
                        a[type] = a_tmp
                        logp[type] = logp_tmp
                    
                s1, r, done, envs_info = envs.step(a)
                uav_pois.append(envs_info[0]['uav_pois'])
                ugv_pois.append(envs_info[0]['ugv_pois'])
                done = done[0]
                ep_ret += r['uav'].sum(axis=-1) + r['carrier'].sum(axis=-1)  # 对各agent的奖励求和
                ep_len += 1
                self.logger.log(interaction=None)
            max_id = ep_ret.argmax()
            envs_info[max_id].update({"uav_pois": uav_pois, "ugv_pois": ugv_pois})
            if ep_ret.max() > self.best_test_episode_reward:
                self.best_test_episode_reward = ep_ret[max_id]
                self.write_output(envs_info[max_id], self.run_args.output_dir, tag='test')
                envs.save_trajectory(envs_info[max_id])
                if not self.run_args.test:
                    self.agent.save_nets(dir_name=self.run_args.output_dir, is_newbest=True)
            if iter % 1000 == 0:
                self.write_output(envs_info[max_id], self.run_args.output_dir, tag='test')
                envs.save_trajectory(envs_info[max_id])
                self.agent.save_nets(dir_name=self.run_args.output_dir, iter=iter, is_newbest=False)
                
            if self.run_args.test:
                output_strings.append(f"\nsequence:{order},"
                                      f"Map {envs_info[0]['map']}: "
                                      f"sensing_efficiency  {'%.3f' % envs_info[0]['a_sensing_efficiency']} "
                                      f"data_collection_ratio  {'%.3f' % envs_info[0]['a_data_collection_ratio']} "
                                      f"episodic_aoi: {'%.3f' % envs_info[0]['a_episodic_aoi']} "
                                      f"energy_consumption_ratio: {'%.3f' % envs_info[0]['a_energy_consumption_ratio']}\n"
                                      f"reward_list: {envs_info[0]['reward_list']}")
                print(output_strings[-1])
                envs.save_trajectory(envs_info[max_id])
            returns += [ep_ret.sum()]
            lengths += [ep_len]
        returns = np.stack(returns, axis=0)
        lengths = np.stack(lengths, axis=0)
        self.logger.log(test_episode_reward=returns.mean(),
                        test_episode_len=lengths.mean(), test_round=None)

        log_dict = {}
        for key in envs_info[0].keys() - exclude_names:
            log_dict['test_metric/' + key] = sum(d[key] for d in envs_info) / len(envs_info)
        self.logger.log(**log_dict)

        if self.run_args.test:
            logging_path = osp.join(self.run_args.output_dir, f'sequence_output.txt')
            with open(logging_path, 'a') as f:
                output_strings += '\n\n'
                f.writelines(output_strings)

        average_ret = returns.mean()

        return average_ret

    def rollout_env(self, iter):  # 与环境交互得到trajs
        """
        The environment should return sth like [n_agent, dim] or [batch_size, n_agent, dim] in either numpy or torch.
        """
        self.routine_count += 1

        trajBuffer = TrajectoryBuffer(self.alg_args, device=self.device)
        envs = self.envs_learn
        cols, task_embs, ae_hiddens, memory = None, None, None, None
        for t in range(int(self.rollout_length / self.input_args.n_thread)):  # 加入向量环境后，控制总训练步数不变
            s = envs.get_obs_from_outside()
            if self.use_mate:
                if self.ae_hiddens is None:
                    self.a = {key: np.zeros((self.input_args.n_thread, self.num_agent[key])) for key in self.agent_type}
                    self.r = {key: np.zeros((self.input_args.n_thread, self.num_agent[key])) for key in self.agent_type}
                    self.ae_hiddens = {key: np.zeros(
                        (self.input_args.n_thread, self.num_agent[key], self.agent.mate[key].hidden_dims()[0])) for key
                                       in self.agent_type}

                task_embs, ae_hiddens = self.agent.encode(s, self.a, self.r, self.ae_hiddens, no_grads=True)

            if self.input_args.algo == 'CPPO':
                a, logp, cols, memory = self.agent.act(s, task_embs, random_seq=True, memory=memory)
            else:
                dist = self.agent.act(s, task_embs)
                a = {}
                logp = {}
                for type in self.agent_type:
                    if 'poi' in type: continue
                    a_tmp = dist[type].sample()
                    logp_tmp = dist[type].log_prob(a_tmp)
                    a_tmp = a_tmp.detach().cpu().numpy()
                    a[type] = a_tmp
                    logp[type] = logp_tmp
                    
            if self.poi_decision_mode:
                dist_poi = self.poi_agent.act(s,task_embs)
                for type in self.poi_agent.agent_type:
                    a_tmp = dist_poi[type].sample()
                    logp_tmp = dist_poi[type].log_prob(a_tmp)
                    a_tmp = a_tmp.detach().cpu().numpy()
                    a[type] = a_tmp
                    logp[type] = logp_tmp
                    
            
            s1, r, done, env_info = envs.step(a)
            if self.use_lambda:
                cost = {key: np.array([e['cost'] for e in env_info]) for key in self.agent_type}
            else:
                cost = None
            done = sum(done)
            trajBuffer.store(s['State'], s, a, r, s1['State'], s1,
                             {key: np.full((self.n_thread, self.num_agent[key]), done) for key in self.agent_type},
                             {key: s['mask_' + key] for key in self.agent_type}, logp,
                             ae_hidden=ae_hiddens, task_embedding=task_embs, action_0=self.a, reward_0=self.r,
                             cost=cost,
                             nodes=s['Nodes'], edges=s['Edges'],
                             adj={key: s[key] for key in adjacent_name_list}
                             , cols=cols, memory=memory, step=env_info[0]['step'])

            self.ae_hiddens, self.r, self.a = ae_hiddens, r,a
            episode_r = r['uav']
            assert episode_r.ndim > 1
            episode_r = episode_r.sum(axis=-1)  # 对各agent奖励求和
            self.episode_reward += episode_r
            self.episode_len += 1
            self.logger.log(interaction=None)

            if done:
                ep_r = self.episode_reward
                #print('train episode reward:', ep_r)
                self.logger.log(mean_episode_reward=ep_r.mean(), episode_len=self.episode_len, episode=None)
                self.logger.log(max_episode_reward=ep_r.max(), episode_len=self.episode_len, episode=None)
                cols, task_embs, ae_hiddens, memory = None, None, None, None
                # 大家是一起结束的，可以不用区分
                if self.use_mate:
                    self.ae_hiddens = None
                # if ep_r.max() > self.best_episode_reward:
                #     max_id = ep_r.argmax()
                #     self.best_episode_reward = ep_r.max()
                #     self.agent.save_nets(dir_name=self.run_args.output_dir, is_newbest=True)
                #     best_train_trajs = self.envs_learn.get_saved_trajs()
                #     poi_aoi_history = self.envs_learn.get_poi_aoi_history()
                #     serves = self.envs_learn.get_serves()
                #     write_output(env_info[max_id], self.run_args.output_dir)
                #     self.dummy_env.save_trajs_2(
                #         best_train_trajs[max_id], poi_aoi_history[max_id], serves[max_id], phase='train', is_newbest=True)

                # if self.routine_count // 500 > 0:  # routinely vis (OK)
                #     max_id = ep_r.argmax()
                #     best_train_trajs = self.envs_learn.get_saved_trajs()
                #     poi_aoi_history = self.envs_learn.get_poi_aoi_history()
                #     serves = self.envs_learn.get_serves()
                #     self.dummy_env.save_trajs_2(
                #         best_train_trajs[max_id], poi_aoi_history[max_id], serves[max_id],
                #         iter=self.rr*500, phase='train')

                # self.rr += self.routine_count // 500
                # self.routine_count = self.routine_count % 500

                log_dict = {}
                for key in env_info[0].keys() - exclude_names:
                    log_dict['train_metric/' + key] = sum(d[key] for d in env_info) / len(env_info)
                    
                self.logger.log(**log_dict)
                '''执行env的reset'''
                try:
                    map_index = np.random.randint(0, 2) if self.random_map else 0
                    _, self.episode_len = self.envs_learn.reset(map_index), 0
                    self.episode_reward = np.zeros((self.input_args.n_thread))
                except Exception as e:
                    raise NotImplementedError

        return trajBuffer.retrieve()

    def write_output(self, info, output_dir, tag='train'):
        logging_path = osp.join(output_dir, f'{tag}_output.txt')
        with open(logging_path, 'a') as f:
            f.write('[' + datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S') + ']\n')
            f.write(f"Map {info['map']}: "
                    f"sensing_efficiency  {'%.3f' % info['a_sensing_efficiency']} "
                    f"data_collection_ratio  {'%.3f' % info['a_data_collection_ratio']} "
                    f"episodic_aoi: {'%.3f' % info['a_episodic_aoi']} "
                    f"energy_consumption_ratio: {'%.3f' % info['a_energy_consumption_ratio']} "
                    f"best_{tag}_reward: {'%.3f' % self.best_test_episode_reward if tag == 'test' else '%.3f' % self.best_episode_reward}"
                    + '\n'
                    )



def generate_permutations(N):
    nums = np.arange(1, N + 1)
    all_permutations = np.array(np.meshgrid(*([nums] * N))).T.reshape(-1, N)
    return all_permutations -1


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
