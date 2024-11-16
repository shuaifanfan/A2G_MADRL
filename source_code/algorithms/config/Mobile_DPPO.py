'''本文件从Catchup_DPPO魔改而来'''

import numpy as np
from gym.spaces import Box
import torch.nn
from algorithms.models import MLP
from algorithms.utils import Config


def getArgs(radius_v, radius_pi, env, input_args):

    alg_args = Config()
    # 总训练步数 = n_iter * rollout_length，默认5K * 0.6K = 3M
    alg_args.n_iter = 5000  # 25000
    alg_args.n_inner_iter = 1
    alg_args.n_warmup = 0
    alg_args.n_model_update = 5
    alg_args.n_model_update_warmup = 10
    alg_args.n_test = 1  # default=5, 意为每次test 5个episode
    alg_args.test_interval = 20
    
    alg_args.max_episode_len = input_args.max_episode_step
    
    alg_args.rollout_length = alg_args.max_episode_len*input_args.n_thread  # 也即PPO中的T_horizon，
    alg_args.model_based = False
    alg_args.load_pretrained_model = False
    alg_args.pretrained_model = None
    alg_args.model_batch_size = 128
    alg_args.model_buffer_size = 0

    agent_args = Config()


    agent_args.n_agent = env.NUM_UAV
    agent_args.gamma = 0.99
    agent_args.lamda = 0.5
    agent_args.clip = 0.2
    agent_args.target_kl = 0.01
    agent_args.v_coeff = 1.0
    agent_args.v_thres = 0.
    agent_args.entropy_coeff = 0.0
    agent_args.lr = 5e-5
    agent_args.lr_v = 5e-4
    agent_args.n_update_v = 30
    agent_args.n_update_pi = 10
    agent_args.n_minibatch = 8
    agent_args.batch_size = 256
    agent_args.use_reduced_v = True
    agent_args.use_rtg = True
    agent_args.use_gae_returns = False
    agent_args.advantage_norm = True
    # agent_args.observation_space = env.observation_space
    agent_args.observation_dim = env.observation_space['Box'].shape[1]  # 标量1715，意为每个agent的obs的向量维度
    agent_args.action_space = env.action_space
    # agent_args.adj = env.neighbor_mask
    agent_args.radius_v = radius_v
    agent_args.radius_pi = radius_pi
    
    agent_args.squeeze = False

    p_args = None
    agent_args.p_args = p_args
    
    agent_args.use_lambda = False
    agent_args.use_mate = False
    agent_args.ae_type = 'mix'
    agent_args.task_emb_dim = 20

    v_args = Config()
    v_args.activation = torch.nn.ReLU
    v_args.sizes = [-1, 128, 128, 1]
    v_args.task_emb_dim =  agent_args.task_emb_dim
    agent_args.v_args = v_args

    pi_args = Config()
    pi_args.sizes = [-1, 128, 128]  # 9是硬编码的离散动作数
    pi_args.branchs = env.action_space
    pi_args.have_last_branch = False
    pi_args.network = MLP
    pi_args.activation = torch.nn.ReLU
    pi_args.squash = False
    pi_args.type=None
    pi_args.task_emb_dim =  agent_args.task_emb_dim
    
    
    agent_args.pi_args = pi_args
    alg_args.agent_args = agent_args

    return alg_args
