import os
import sys
from datetime import datetime
import importlib
import json

os.environ["WANDB_API_KEY"] = 'c31608fd9c00b810892209570fd0ac47da7e8acb'
#os.environ["WANDB_MODE"] = "offline"


proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("---------------------")
print(proj_dir)
sys.path.append(proj_dir)

from algorithms.utils import Config, LogClient, LogServer
from algorithms.algo.main import OnPolicyRunner
from algorithms.algo.random_runner import RandomRunner
from types import SimpleNamespace

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# for item in ['MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
#     os.environ[item] = "1"


import setproctitle


def getRunArgs(input_args):
    run_args = Config()
    run_args.device = input_args.device

    run_args.debug = input_args.debug
    run_args.test = input_args.test
    run_args.checkpoint = input_args.checkpoint
    run_args.group = input_args.group
    run_args.mute_wandb = input_args.mute_wandb

    run_args.radius_v = 1
    run_args.radius_pi = 1

    run_args.start_step = 0
    run_args.save_period = 1800  # in seconds
    run_args.log_period = int(20)
    run_args.seed = None
    return run_args


def getAlgArgs(run_args, input_args, env):
    #assert input_args.env.startswith("Mobile")
    assert input_args.algo in [
        "DPPO",
        "CPPO",
        "IPPO",
        "DMPO",
        "IC3Net",
        "IA2C",
        "G2ANet",
        "G2ANe2",
        "ConvLSTM",
        "Random",
    ]
    filename = input_args.algo

    config = importlib.import_module(f"algorithms.config.{input_args.env}_{filename}")
    alg_args = config.getArgs(
        run_args.radius_v, run_args.radius_pi, env, input_args=input_args
    )
    return alg_args


def initAgent(logger, device, agent_args, input_args):
    return AgentFn(logger, device, agent_args, input_args)


def get_name(input_args):
    timenow = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
    if input_args.tag is None:
        name = timenow
    else:
        name = "{}_{}_{}_{}".format(timenow, input_args.dataset, input_args.tag, input_args.algo)  
        
    name += f"_num_{input_args.num_uav}"
    # name += f"_mask_range_{input_args.mask_range}"
    if input_args.reduce_poi:
        name += 'easy_'
    if input_args.use_hgcn:
        name += "_hgcn_"
    if input_args.use_lambda:
        name += "_lambda_"
    # if input_args.use_mate:
    #     name += "_mate_"
    #     name += input_args.ae_type
    #     name += "_" + input_args.mate_type

    # if input_args.use_gcn:
    #     run_args.name += "_use_gcn_"

    if input_args.random_map:
        name += "_random_map_"

    if input_args.limited_collection:
        name += "_limited_collection_"

    # if input_args.centralized:
    #     run_args.name += "_centralized_"  

    if input_args.reward_type:
        name += f"_{input_args.reward_type}"

    if input_args.random_permutation:
        name += "_rand_permute_"

    # if input_args.fixed_relay:
    #     run_args.name += "_fixed_relay_"
    # if input_args.credit_assign:
    #     name += "_credit_ass"

    permutation_mapping = {
        "ucb": input_args.ucb_confidence,
        "eps": input_args.permutation_eps,
        "semi": None,
        "greedy": None,
        "fixed": None,
        "random": None
    }

    new_string = ""

    exclusive_strategies = {"fixed", "random", "greedy"}

    # Check for mutually exclusive strategies
    for strategy in exclusive_strategies:
        if strategy in input_args.permutation_strategy:
            if permutation_mapping[strategy] is not None:
                new_string += f"_{strategy}{permutation_mapping[strategy]}"
            else:
                new_string += f"_{strategy}"
            break  # Exit the loop since one of the exclusive strategies was found

    # Handle the remaining strategies
    if "greedy" in input_args.permutation_strategy:
        for strategy in (permutation_mapping.keys() - exclusive_strategies):
            value = permutation_mapping[strategy]
            if strategy in input_args.permutation_strategy:
                if value is not None:
                    new_string += f"_{strategy}{value}"
                else:
                    new_string += f"_{strategy}"

    name += new_string

    # if input_args.decay_eps:
    #     name += "_decay_eps_"

    # if input_args.share_mate:
    #     run_args.name+='_share_mate_'

    # if input_args.cat_position:
    #     run_args.name+='_cat_position_'

    if input_args.checkpoint and not input_args.test:
        name += f'_fewshot_'

    if input_args.map != 0:
        name += f'_Map_{input_args.map}'

    if input_args.use_ar_policy:
        name += "_arp_"

    if input_args.use_sequential_update:
        name += "_sequ_"

    if input_args.use_temporal_type:
        name += 'tgcn'

    if input_args.dis_bonus:
        name += "_dis_re"
        
    if input_args.two_stage_mode and input_args.near_selection_mode:
        name += "_use_hgcn"


    return name


def override(alg_args, run_args, input_args, env, name):
    if run_args.debug:
        alg_args.model_batch_size = 5  # 用于训练一次model的traj数量
        alg_args.max_ep_len = 120
        alg_args.rollout_length = input_args.rollout_length
        # 测试episode的最大步长
        alg_args.model_buffer_size = 10
        alg_args.n_model_update = 3
        alg_args.n_model_update_warmup = 3
        alg_args.n_warmup = 1
        alg_args.model_prob = 1  # 规定一定会执行从model中采样用于更新policy的经验
        # 注意: n_iter*rollout_length得比一个episode长，不然一次train episode done都不触发，train_trajs不会保存到外存
        alg_args.n_iter = 10
        alg_args.n_test = 1
        alg_args.n_traj = 4
        alg_args.n_inner_iter = 1
    if run_args.test:
        run_args.debug = True
        alg_args.n_warmup = 0
        alg_args.n_test = 1
    if run_args.seed is None:
        # 固定随机种子
        #run_args.seed = 2023
        run_args.seed = 2025
        # run_args.seed = int(time.time())

    if input_args.algo == "Random":
        alg_args.n_test = 1

    run_args.name = name
    if not input_args.test or input_args.algo == "Random":
        final = "{}/{}".format(input_args.output_dir, run_args.name)
        run_args.output_dir = final
        input_args.output_dir = final
    else:
        run_args.output_dir = input_args.output_dir

    alg_args.algo = input_args.algo
    alg_args.use_stack_frame = input_args.use_stack_frame
    alg_args.agent_type = env.UAV_TYPE
    alg_args.use_mate = input_args.use_mate
    alg_args.ae_type = input_args.ae_type
    alg_args.mate_type = input_args.mate_type
    alg_args.use_lambda = input_args.use_lambda

    return alg_args, run_args, input_args


def record_input_args(input_args, env_args, output_dir):
    params = dict()
    from LaunchMCS.util.config_3d import Config

    env_config = Config(env_args)
    params["input_args"] = vars(input_args)
    params["env_config"] = env_config.dict

    for key in shared_feature_list:
        del params["env_config"][key]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "params.json"), "w") as f:
        f.write(json.dumps(params))


from get_args import parse_args

input_args, env_args = parse_args()

if input_args.checkpoint:
    with open(os.path.join(input_args.checkpoint, "params.json"), "r") as f:
        old_args = json.load(f)["input_args"]
    old_args['test'] = input_args.test
    old_args['checkpoint'] = input_args.checkpoint
    old_args['map'] = input_args.map
    old_args['group'] += '_fewshot'
    old_args['n_iter'] = input_args.few_shot_iter
    old_args['model_iter'] = input_args.model_iter
    input_args, env_args = parse_args(SimpleNamespace(**old_args))
    env_args["test_mode"] = input_args.test
    env_args["debug_mode"] = False

if input_args.algo == "IA2C":
    from algorithms.algo.agent.IA2C import IA2C as AgentFn
elif input_args.algo == "IC3Net":
    from algorithms.algo.agent.IC3Net import IC3Net as AgentFn
elif input_args.algo == "DPPO":
    from algorithms.algo.agent.DPPO import DPPOAgent as AgentFn
elif input_args.algo == "CPPO":
    from algorithms.algo.agent.CPPO import CPPOAgent as AgentFn
elif input_args.algo == "DMPO":
    from algorithms.algo.agent.DMPO import DMPOAgent as AgentFn
elif input_args.algo == "IPPO":
    from algorithms.algo.agent.IPPO import IPPOAgent as AgentFn
elif input_args.algo == "ConvLSTM":
    from algorithms.algo.agent.ConvLSTM import ConvLSTMAgent as AgentFn
elif input_args.algo == "Random":
    from algorithms.algo.agent.Random import RandomAgent as AgentFn

if input_args.env == "Mobile":
    from LaunchMCS.launch_mcs import EnvUCS, shared_feature_list
    env_fn_train, env_fn_test = EnvUCS, EnvUCS
else:
    raise NotImplementedError

# print(input_args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = input_args.gpu

run_args = getRunArgs(input_args)
print("debug =", run_args.debug)
print("test =", run_args.test)

name = get_name(input_args)
env_args["description"] = name

print(name)
import time

start = time.time()
from env_configs.wrappers.env_wrappers import SubprocVecEnv

env_args['is_sub_env'] = False
main_env = env_fn_train(env_args)
if input_args.reduce_poi:
    main_env.config.dict['poi'] = main_env.config.dict['poi'][::2]
    main_env.config.dict['poi_num'] = len(main_env.config.dict['poi'])
alg_args = getAlgArgs(run_args, input_args, main_env)
alg_args, run_args, input_args = override(alg_args, run_args, input_args, main_env, name)
if input_args.test:
    env_args["save_path"] = run_args.output_dir
else:
    env_args["save_path"] = "../{}/{}/{}".format('runs', input_args.group, name)
share_feature = {share_name: getattr(main_env, share_name) for share_name in
                 shared_feature_list}
env_args.update(share_feature)
env_args['is_sub_env'] = True
envs = [main_env]
for _ in range(input_args.n_thread - 1):
    envs.append(env_fn_train(env_args))
envs_train = SubprocVecEnv(envs)

# envs_train = SubprocVecEnv([env_fn_train(env_args) for _ in range(input_args.n_thread)])
envs_test = SubprocVecEnv([env_fn_test(env_args)])

dummy_env = env_fn_train(env_args)

assert run_args.output_dir == env_args["save_path"]

if not run_args.test:
    record_input_args(input_args, env_args, run_args.output_dir)
setproctitle.setproctitle(run_args.name)

logger = LogServer(
    {"run_args": run_args, "algo_args": alg_args, "input_args": input_args}
)
logger = LogClient(logger)
# logger同时被传入agent类和runner类
agent = initAgent(logger, run_args.device, alg_args.agent_args, input_args)

import time

start = time.time()
if input_args.algo == "Random":
    runner_fn = RandomRunner
else:
    runner_fn = OnPolicyRunner
runner_fn(
    logger=logger,
    agent=agent,
    envs_learn=envs_train,
    envs_test=envs_test,
    dummy_env=dummy_env,
    run_args=run_args,
    alg_args=alg_args,
    input_args=input_args,
).run()
end = time.time()
print(f"OK! 用时{end - start}秒")
