# yyx: 和Mobile_DPPO的内容保持一致

import numpy as np
from gym.spaces import Box
import torch.nn
from algorithms.models import MLP
from algorithms.utils import Config


def getArgs(radius_v, radius_pi, env, input_args=None):

    alg_args = Config()
    # 总训练步数 = n_iter * rollout_length，默认25K * 0.6K = 15M
    # 改为5K * 0.6K = 6M
    alg_args.n_iter = input_args.n_iter  # 25000
    alg_args.n_inner_iter = 1
    alg_args.n_warmup = 0
    alg_args.n_model_update = 5
    alg_args.n_model_update_warmup = 10
    alg_args.n_test = 1  # default=5, 意为每次test 5个episode
    alg_args.test_interval = 20

    alg_args.max_episode_len = input_args.max_episode_step
    alg_args.rollout_length = input_args.rollout_length # 必须是环境数量8的整数倍
 
    alg_args.model_based = False
    alg_args.load_pretrained_model = False
    alg_args.pretrained_model = None
    alg_args.model_batch_size = 128
    alg_args.model_buffer_size = 0
    alg_args.poi_decision_mode = input_args.poi_decision_mode

    agent_args = Config()
    agent_args.agent_type = env.UAV_TYPE
    agent_args.n_agent = env.NUM_UAV
    agent_args.n_poi = env.POI_NUM
    agent_args.n_node = 2
    agent_args.gamma = 0.99
    agent_args.lamda = 0.95
    agent_args.clip = 0.2
    agent_args.target_kl = 0.01
    agent_args.v_coeff = 1.0
    agent_args.v_thres = 0.
    agent_args.entropy_coeff = 0.0
    agent_args.lr = input_args.lr
    agent_args.lr_v = 5e-4
    agent_args.n_update_v = 20
    agent_args.each_ig_step_num = input_args.each_ig_step_num
    agent_args.n_update_pi = input_args.n_update_pi
    agent_args.n_minibatch = 1
    agent_args.batch_size = alg_args.rollout_length
    agent_args.use_reduced_v = False  # 和dppo不同
    agent_args.use_rtg = True
    agent_args.use_gae_returns = True
    agent_args.advantage_norm = True
    # agent_args.observation_space = env.observation_space
    agent_args.observation_dim = {key: env.obs_space[key + "_obs"].shape[1] for key in agent_args.agent_type}
    agent_args.observation_dim.update({'State':env.obs_space['State'].shape[0]})
    agent_args.action_space = env.action_space
    agent_args.poi_action_space = env.poi_action_space
    # agent_args.adj = env.neighbor_mask
    agent_args.radius_v = radius_v
    agent_args.radius_pi = radius_pi

    agent_args.squeeze = False

    p_args = None
    agent_args.p_args = p_args

    agent_args.use_lambda = input_args.use_lambda
    agent_args.use_mate = input_args.use_mate
    agent_args.ae_type = input_args.ae_type
    agent_args.mate_type = input_args.mate_type
    agent_args.mate_rl_gradient = input_args.mate_rl_gradient
    agent_args.task_emb_dim = input_args.task_emb_dim
    agent_args.random_permutation = input_args.random_permutation
    agent_args.use_temporal_type = input_args.use_temporal_type
    agent_args.decay_eps = input_args.decay_eps
    agent_args.use_hgcn = input_args.use_hgcn
    agent_args.use_gcn = input_args.use_gcn
    agent_args.gcn_nodes_num = 2
    agent_args.gcn_emb_dim = 1
    agent_args.n_embd = input_args.n_embd

    v_args = Config()
    v_args.activation = torch.nn.ReLU
    v_args.sizes = [-1,512, 256, 128, 1]
    v_args.use_lambda=  agent_args.use_lambda
    v_args.task_emb_dim =  agent_args.task_emb_dim
    v_args.use_mate =  agent_args.use_mate
    
    v_args.use_gcn = agent_args.use_gcn
    v_args.gcn_emb_dim = agent_args.gcn_emb_dim
    v_args.gcn_nodes_num = agent_args.gcn_nodes_num

    pi_args = Config()
    pi_args.sizes = [-1,512, 256, 128]  # 9是硬编码的离散动作数
    pi_args.branchs = env.action_space
    pi_args.poi_branchs = env.poi_action_space
    pi_args.have_last_branch = False
    pi_args.network = MLP
    pi_args.activation = torch.nn.ReLU
    pi_args.squash = False
    pi_args.type=None
    pi_args.task_emb_dim =  agent_args.task_emb_dim
    pi_args.use_lambda =  agent_args.use_lambda
    pi_args.use_mate =  agent_args.use_mate
    
    pi_args.use_gcn = agent_args.use_gcn
    pi_args.gcn_emb_dim = agent_args.gcn_emb_dim
    pi_args.gcn_nodes_num = agent_args.gcn_nodes_num
    
    agent_args.v_args = v_args
    agent_args.pi_args = pi_args
    alg_args.agent_args = agent_args

    return alg_args
