import argparse


def parse_args(old_args=None):
    parser = argparse.ArgumentParser()
    # 已经验证这里的参数可被存入params.json

    parser.add_argument('--debug', action='store_true', default=False, )
    parser.add_argument('--test', action='store_true', default=False, )
    parser.add_argument('--user', type=str, default='zf')
    parser.add_argument('--env', type=str, default='Mobile')
    parser.add_argument('--algo', type=str, required=False, default='CPPO',
                        help="algorithm(G2ANet/IC3Net/CPPO/DPPO/IA2C/IPPO/Random) ")
    parser.add_argument('--device', type=str, required=False, default='cuda:0', help="device(cpu/cuda:0/cuda:1/...) ")
    parser.add_argument("--dataset", type=str, default='Rome', choices=['KAIST', 'Rome'])
    parser.add_argument("--tag", type=str, default='', help='每个单独实验的备注')
    parser.add_argument("--gpu", type=str, default='1', help='')
    # dirs
    parser.add_argument("--output_dir", type=str, default='runs/debug', help="which fold to save under 'runs/'")
    parser.add_argument('--group', type=str, default='zf_group',
                        help='填写我对一组实验的备注，作用与wandb的group和tb的实验保存路径')
    # system stub
    parser.add_argument('--mute_wandb', default=False, action='store_true')
    # tune agent
    parser.add_argument('--checkpoint', type=str)  # load pretrained model
    parser.add_argument('--model_iter', type=int, default=14000)  # load pretrained model
    parser.add_argument('--n_thread', type=int, default=16)
    parser.add_argument('--rollout_length', type=int, default=480)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_iter', type=int, default=15000)
    parser.add_argument('--few_shot_iter', type=int, default=2000)
    # tune algo
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_update_pi', type=int, default=10)

    parser.add_argument('--lr_v', type=float)
    parser.add_argument('--use-stack-frame', action='store_true')
    # parser.add_argument('--use_extended_value', action='store_false', help='反逻辑，仅用于DPPO')
    # parser.add_argument('--use-mlp-model', action='store_true', help='将model改为最简单的mlp，仅用于DMPO')
    # parser.add_argument('--multi-mlp', action='store_true', help='在model中分开预测obs中不同类别的信息，仅用于DMPO')

    parser.add_argument('--use_lambda', action='store_true')
    parser.add_argument('--use_mate', action='store_true')
    parser.add_argument('--ae_type', type=str, default='vae')
    parser.add_argument('--mate_type', type=str, default='mix')
    parser.add_argument('--share_mate', action='store_true', default=False)
    parser.add_argument('--mate_path', type=str, default='')
    parser.add_argument('--mate_rl_gradient', action='store_true')
    # parser.add_argument('--use_gcn',action='store_false', default=True)
    parser.add_argument('--use_gcn', action='store_false', default=False)
    parser.add_argument('--mat', type=str, default='no',
                        choices=['mat', 'mat_dec', 'mat_gru', 'mat_decoder', 'mat_encoder'])
    parser.add_argument('--cat_position', action='store_false', default=True)

    parser.add_argument('--rep_iter', type=int, default=1)
    parser.add_argument('--rep_learning', action='store_true', default=False)
    parser.add_argument('--mask_obs', action='store_true', default=False)
    parser.add_argument('--lr_scheduler', type=str, default='cos')
    parser.add_argument('--use_hgcn', action='store_true', default=False)
    parser.add_argument('--random_permutation', action='store_true', default=False)
    parser.add_argument('--centralized', action='store_true', default=False)
    parser.add_argument('--permutation_strategy', type=str, default='')
    parser.add_argument('--permutation_eps', type=float, default=0.3)
    parser.add_argument('--decay_eps', action='store_true', default=False)
    parser.add_argument('--use_sequential_update', action='store_true', default=False)
    parser.add_argument('--use_ar_policy', action='store_true', default=False)
    parser.add_argument('--use_temporal_type', action='store_true', default=False)
    parser.add_argument('--hgcn_rollout_len', type=int, default=20)
    parser.add_argument('--hgcn_layer_num', type=int, default=1)
    

    parser.add_argument('--task_emb_dim', type=int, default=20)
    parser.add_argument('--n_embd', type=int, default=32)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--use_max_grad_norm', action='store_false', default=True)
    parser.add_argument('--use_clipped_value_loss', action='store_false', default=True)
    parser.add_argument('--use_huber_loss', action='store_false', default=True)
    parser.add_argument('--use_valuenorm', action='store_true', default=False)

    #addedy bu zf，下面这两个单步reward先不用，只看一下原来的code，能否起到下面这两个metrics的效果
    parser.add_argument('--use_aoi_var', action='store_true', default=False)#在单步 reward中加入aoi的方差
    # Value of Information
    parser.add_argument('--use_voi', type=int,default=0,
                        choices=[0,1,2])#默认不用voi，1表示在单步reward中，使用voi总体代替collect_data,2表示只是用voi的第二部分代替collect_data
    parser.add_argument('--voi_beta', type=float, default=0.8)
    parser.add_argument('--voi_k', type=float, default=0.1)
    # tune env
    ## setting
    parser.add_argument('--limited_collection', action='store_true')
    parser.add_argument('--random_map', action='store_true')
    parser.add_argument('--fixed_relay', action='store_true', default=True)
    parser.add_argument('--time_slot', type=float, default=20)
    parser.add_argument('--uav_poi_dis', type=float, default=-1.0)
    parser.add_argument('--reduce_poi', action='store_true', default=False)
    parser.add_argument('--carrier_explore_reward', action='store_true', default=False)
    parser.add_argument("--credit_assign", action='store_true', default=False)
    parser.add_argument("--each_ig_step_num", type=int, default=5)

    ## MDP
    parser.add_argument('--max_episode_step', type=int, default=120)
    parser.add_argument('--map', type=int, default=0)
    parser.add_argument('--channel_num', type=int, default=5)
    parser.add_argument('--num_uav', type=int, default=2)
    parser.add_argument('--user_data_amount', type=int, default=40)
    parser.add_argument('--edge_type', default='dis', choices=['dis', 'data_rate'])
    parser.add_argument("--data_rate_thre", type=float, default=1.0)
    parser.add_argument("--reward_type", default='none', choices=['none', 'prod', 'prod_thre', 'sum', 'square'])
    parser.add_argument("--dis_bonus", action='store_true', default=False)
    parser.add_argument("--act_update_seq_diff", action='store_true', default=False)
    parser.add_argument("--mask_range", type=float, default=1.2)
    parser.add_argument("--ucb_confidence", type=float, default=2.0)
    parser.add_argument("--poi_decision_mode", default=False, action='store_true')
    parser.add_argument("--near_selection_mode", default=False, action='store_true')
    parser.add_argument("--two_stage_mode", default=False, action='store_true')
    parser.add_argument("--use_graph_feature", default=False, action='store_true')
    parser.add_argument("--rl_greedy_reward", default=False, action='store_true')
    parser.add_argument("--decoupled_actor", default=False, action='store_true')
    parser.add_argument("--poi_in_obs_num", type=int, default=-1)
    
    parser.add_argument('--data_collect_range', type=int, default=300)
    parser.add_argument('--data_poi_init_data', type=int, default=1200)
    parser.add_argument('--data_num_uav', type=int, default=3)
    
    
    
    
    if old_args is None:
        input_args = parser.parse_args()
    else:
        input_args = old_args

    # if input_args.multi_mlp:
    #     assert input_args.use_mlp_model

    if input_args.algo == 'Random':
        input_args.test = True
        input_args.debug = False

    if input_args.debug:
        input_args.group = 'debug'
    input_args.output_dir = f'../runs/{input_args.group}'

    if input_args.test:
        input_args.group = 'test'
        input_args.n_thread = 1
        input_args.output_dir = f'{input_args.checkpoint}/test'

    if input_args.algo == 'Random':
        input_args.output_dir = f'runs/random'

    if input_args.env == 'Mobile':
        env_args = {
            "max_episode_step": input_args.max_episode_step,
            "random_map": input_args.random_map,
            "limited_collection": input_args.limited_collection,
            "centralized": input_args.centralized,
            "map": input_args.map,
            "fixed_relay": input_args.fixed_relay,
            "time_slot": input_args.time_slot,
            "uav_poi_dis": input_args.uav_poi_dis,
            "use_hgcn": input_args.use_hgcn,
            "dataset": input_args.dataset,
            "edge_type": input_args.edge_type,
            "channel_num": input_args.channel_num,
            "mask_range": input_args.mask_range,
            "num_uav": {'carrier': input_args.num_uav, 'uav': input_args.num_uav},
            "user_data_amount": input_args.user_data_amount,
            'is_sub_env': False,
            "data_rate_thre": input_args.data_rate_thre,
            #'debug_mode': input_args.debug,
            'debug_mode': False,
            "reward_type": input_args.reward_type,
            "dis_bonus": input_args.dis_bonus,
            "reduce_poi": input_args.reduce_poi,
            "poi_decision_mode":input_args.poi_decision_mode,
            'two_stage_mode':input_args.two_stage_mode,
            "rl_greedy_reward":input_args.rl_greedy_reward,
            "near_selection_mode":input_args.near_selection_mode,
            
            #added by zf, value of information, 24.11.18
            "use_voi":input_args.use_voi,
            "voi_beta":input_args.voi_beta,
            "voi_k":input_args.voi_k,
        }
    else:
        env_args = {
            "max_episode_step": input_args.max_episode_step,
            "centralized": input_args.centralized,
            "use_hgcn": input_args.use_hgcn,
            "dataset": input_args.dataset,
            "num_uav": {'uav': input_args.data_num_uav},
            'is_sub_env': False,
            "data_rate_thre": input_args.data_rate_thre,
            'debug_mode': False,
            "poi_init_data":input_args.data_poi_init_data,
            "collect_range":{'uav':input_args.data_collect_range}
        }
    # if input_args.poi_num is not None:
    #     env_args["poi_num"] = input_args.poi_num
    # input_args.lr_scheduler = ''
    return input_args, env_args
