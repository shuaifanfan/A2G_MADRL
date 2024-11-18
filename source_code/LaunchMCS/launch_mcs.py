import os
import sys
sys.path.append('/home/liuchi/zf/MCS_with_git/MCS_TEST')
import copy
import pickle
import warnings
import osmnx as ox
import networkx as nx
import json
import random

from typing import Dict
from LaunchMCS.util.config_3d import Config
from LaunchMCS.util.utils import IsIntersec
from LaunchMCS.util.noma_utils import *
from LaunchMCS.util.roadmap_utils import Roadmap
from LaunchMCS import shared_feature_list
from tqdm import tqdm
from gym import spaces
from collections import OrderedDict
from math import exp
from importlib import import_module

np.seterr(all="raise")
EPISODIC_AOI_NAME = 'a_episodic_aoi'
DATA_COLLECTION_RATIO_NAME = 'a_data_collection_ratio'
SENSING_EFFICIENCY_NAME = 'a_sensing_efficiency'


class EnvUCS(object):

    def __init__(self, args=None, **kwargs):
        if args != None:
            self.args = args
        self.config = Config(args) #里面有env默认参数，更高优先级的是main_DPPO.py里面的args,会覆盖这里的参数
        map_number = self.config.dict['map']
        self.REDUCE_POI = self.config('reduce_poi')
        self.load_map(map_number)
        self.DISCRIPTION = self.config('description')
        self.CENTRALIZED = self.config('centralized')
        self.SCALE = self.config("scale")
        self.INITIAL_ENERGY = self.config("initial_energy")
        self.EPSILON = self.config("epsilon")
        self.DATA_RATE_THRE = self.config("data_rate_thre")
        self.DEBUG_MODE = self.config("debug_mode")
        #self.DEBUG_MODE = True
        self.TEST_MODE = self.config("test_mode")
        self.ACTION_MODE = self.config("action_mode")
        self.COLLECT_MODE = self.config("collect_mode")
        self.MAX_EPISODE_STEP = self.config("max_episode_step")
        self._max_episode_steps = self.MAX_EPISODE_STEP
        self.TIME_SLOT = self.config("time_slot")
        self.SAVE_COUNT = self.config('seed')
        self.USE_HGCN = self.config('use_hgcn')

        self.USE_VOI = self.config('use_voi')
        self.VOI_BETA = self.config('voi_beta')
        self.VOI_K = self.config('voi_k')

        self.CONCAT_OBS = self.config("concat_obs")
        self.POI_INIT_DATA = self.config("poi_init_data")
        self.AOI_THRESHOLD = self.config("aoi_threshold")
        self.TOTAL_TIME = self.MAX_EPISODE_STEP * self.TIME_SLOT
        self.THRESHOLD_PENALTY = self.config("threshold_penalty")
        self.UAV_HEIGHT = self.config("uav_height")
   
        self.USER_DATA_AMOUNT = self.config("user_data_amount")
        self.CHANNEL_NUM = self.config("channel_num")
        self.NOMA_MODE = self.config("noma_mode")
        self.ROADMAP_MODE = self.config("roadmap_mode")
        self.edge_type = self.config('edge_type')
        self.reward_type = self.config('reward_type')
        #self.dis_bonus = self.config('dis_bonus')
        self.dis_bonus = False
        self.UAV_TYPE = ['carrier', 'uav']
        self.NUM_UAV = OrderedDict(self.config("num_uav"))
        self.UAV_SPEED = OrderedDict(self.config("uav_speed"))
        self.RATE_THRESHOLD = OrderedDict(self.config("rate_threshold"))
        self.UPDATE_NUM = OrderedDict(self.config("update_num"))
        self.COLLECT_RANGE = OrderedDict(self.config("collect_range"))
        self.LIMIT_RANGE = OrderedDict(self.config("limit_range"))
        self.POI_DECISION_MODE = self.config('poi_decision_mode')
        self.TWO_STAGE_MODE = self.config('two_stage_mode')
        self.NEAR_SELECTION_MODE = self.config('near_selection_mode')
        self.RL_GREEDY_REWARD = self.config('rl_greedy_reward')
        self.POI_IN_OBS_NUM = min(self.config('poi_in_obs_num'),self.config('poi_num'))  #-1
        self.POI_SELECTION_NUM = 15
        if not self.TWO_STAGE_MODE and self.NEAR_SELECTION_MODE and self.POI_DECISION_MODE:
            self.POI_IN_OBS_NUM = self.POI_SELECTION_NUM

        self.ACTION_ROOT = self.config("action_root")
        self.UAV_ACTION_ROOT = 9 
        self.n_agents = sum(self.NUM_UAV.values())
        self.episode_limit = self._max_episode_steps
        self.n_actions = 1 if self.ACTION_MODE else self.ACTION_ROOT
        self.agent_field = OrderedDict(self.config("agent_field"))
        self.reset_count = 0
        
        if self.config('dataset') == 'KAIST':
            self.broke_threshold = 100
            self.normal_threshold = 50
        elif self.config('dataset') == 'Rome':
            #self.broke_threshold = 500
            #self.normal_threshold = 200
            self.broke_threshold = 100
            self.normal_threshold = 50
        self.MAP_X = self.config("map_x")
        self.MAP_Y = self.config("map_y")
        self.POI_NUM = self.config("poi_num")
        self.OBSTACLE = self.config('obstacle')

        self._poi_position = np.array(self.config('poi'))
        self.mask_range = self.config("mask_range")
        self._poi_position[:, 0] *= self.MAP_X
        self._poi_position[:, 1] *= self.MAP_Y
        self.is_sub_env = self.config('is_sub_env')

        self._uav_energy = {this_uav: [self.config("initial_energy")[this_uav] for i in range(self.NUM_UAV[this_uav])]
                            for this_uav in
                            self.UAV_TYPE}
        self._uav_position = \
            {this_uav: [[self.config("init_position")[this_uav][i][0], self.config("init_position")[this_uav][i][1]] for
                        i in
                        range(self.NUM_UAV[this_uav])] for this_uav in self.UAV_TYPE}
        self.map = str(self.config("map"))
        # print(f'selected map:{self.map}')

        if self.ROADMAP_MODE:#True
            self.rm = Roadmap(self.config("dataset"))
            if self.is_sub_env:
                for item in shared_feature_list:
                    setattr(self, item, self.config(item))
                    # setattr(self, item + '_mem', shared_memory.SharedMemory(self.config(item)))
                    # buffer = getattr(self, item + '_mem').buf
                    # setattr(self, item, json.loads(bytes(buffer[:]).decode('utf-8')))
            else:
                with open(f"/home/liuchi/zf/MCS_with_git/MCS_TEST/source_code/LaunchMCS/util/"
                          f"{self.config('dataset')}/road_map.json", 'r') as f:
                    self.ROAD_MAP = json.load(f)
                    self.ROAD_MAP = {key: set(value) for key, value in self.ROAD_MAP.items()}  # 要remove的edge

                if self.config('dataset') == 'KAIST':
                    pair = -1
                elif self.config('dataset') == 'Rome':
                    pair = 3
                else:
                    raise NotImplementedError
                
                if pair == 0:
                    dis_path = '/home/liuchi/zf/MCS_TEST/source_code/LaunchMCS/util/Rome/(12.4523,41.865)_(12.5264,41.919)_drive_service_Dict_node_857.json'
                    self.normal_threshold = 400
                    self.UAV_SPEED['carrier'] = 40
                elif pair == 1:
                    dis_path = '/home/liuchi/zf/MCS_TEST/source_code/LaunchMCS/util/Rome/(12.4523,41.865)_(12.5264,41.919)_drive_service_Dict_node_572.json'
                    self.normal_threshold = 500
                    self.UAV_SPEED['carrier'] = 40
                elif pair == 2:
                    dis_path = '/home/liuchi/zf/MCS_TEST/source_code/LaunchMCS/util/Rome/(12.4523,41.865)_(12.5264,41.919)_drive_service_Dict_node_406.json'
                    self.normal_threshold = 600
                    self.UAV_SPEED['carrier'] = 40
                elif pair == 3: #zf新加的roma地图
                    dis_path = '/home/liuchi/zf/MCS_with_git/MCS_TEST/source_code/LaunchMCS/util/Rome/(12.4994,41.8822)_(12.5264,41.9018)_drive_service_ZF_Dict_100.json'
                    self.normal_threshold = 50
                    self.UAV_SPEED['carrier'] = 20
                else:
                    dis_path = f"/home/liuchi/zf/MCS_with_git/MCS_TEST/source_code/LaunchMCS/util/{self.config('dataset')}/pair_dis_dict_0.json"
                    
                    
                with open(dis_path, 'r') as f:
                    pairs_info = json.load(f)
                    self.PAIR_DIS_DICT = pairs_info
                    self.valid_nodes = set([int(item) for item in pairs_info['0'].keys()])   
                    self.valid_edges = {key: set() for key in self.ROAD_MAP.keys()}
                    for key in tqdm(['0',], desc='Constructing Edges'):
                        for i in self.valid_nodes:
                            for j in self.valid_nodes:
                                if i == j:
                                    continue
                                dis = pairs_info[key][str(i)][str(j)]
                                if key == '0' and dis <= self.normal_threshold:
                                    self.valid_edges[key].add((i, j))
                                elif key != '0' and dis <= self.broke_threshold:
                                    self.valid_edges[key].add((i, j))
                if self.config('dataset') == 'KAIST':
                    self.ALL_G = ox.load_graphml(
                        f"/home/liuchi/zf/MCS_with_git/MCS_TEST/source_code/LaunchMCS/util/{self.config('dataset')}/map_0.graphml").to_undirected()
                elif self.config('dataset') == 'Rome':
                    self.ALL_G = ox.load_graphml(
                   "/home/liuchi/zf/MCS_with_git/MCS_TEST/source_code/LaunchMCS/util/Rome/(12.4994,41.8822)_(12.5264,41.9018)_map.graphml").to_undirected()
                else:
                    raise NotImplementedError
                    
                
                self.node_map = {}
                for i, node in enumerate(self.ALL_G.nodes):
                    self.node_map[str(node)] = i

                self.NX_G = {}
                for map_num, nodes_to_remove in self.ROAD_MAP.items():
                    if(map_num != '0'):
                        break
                    new_graph = nx.MultiDiGraph()
                    new_graph.graph = self.ALL_G.graph
                    new_graph.add_nodes_from(self.valid_nodes)
                    for node in self.valid_nodes:
                        new_graph.nodes[node].update(self.ALL_G.nodes[node])
                    new_graph.add_edges_from(self.valid_edges[map_num])
                    new_graph = nx.convert_node_labels_to_integers(self.get_sub_graph(new_graph, nodes_to_remove),
                                                                   first_label=0,
                                                                   label_attribute='old_label')
                    self.NX_G[map_num] = new_graph

                   # print(
                   #     f"dataset:{self.config('dataset')},map:{map_num}, number of nodes:{len(new_graph.nodes())}, number of edges: {len(new_graph.edges())}")
                    # print(len(new_graph.edges))
                self.OSMNX_TO_NX = {data['old_label']: node for node, data in
                                    self.NX_G[self.map].nodes(data=True)}

                all_keys = ['0',]
                for key in all_keys:
                    for node, data in self.NX_G[key].nodes(data=True):
                        x, y = self.rm.lonlat2pygamexy(data['x'], data['y'])
                        self.NX_G[key].nodes[node]['py_x'] = x
                        self.NX_G[key].nodes[node]['py_y'] = y
                        self.NX_G[key].nodes[node]['data'] = 0  # 每个node的data在check_arrival中会被更新，含义是对应的距离最近的poi的数据剩余量，get_obs中会用到node的data作为obs
                self.EDGE_FEATURES = {key: np.array(list(self.NX_G[key].edges())).T for key in all_keys}
                #edge_features的格式是：ndarray，shape=（2，edge_num),比如有（1，2）（3，4）（5，6）三条边
                #edge——features:[[1,3,5],
                #                 [2,4,6]]
                # Warning: edges and edges() output different things.
                # if not self.is_sub_env:

                poi_map = {key: self.get_node_poi_map(key, self.NX_G[key]) for key in all_keys}
                self.POI_NEAREST_DIS = {key: poi_map[key][1] for key in all_keys}# poi_index:poi_dis，这样的map
                self.NODE_TO_POI = {key: poi_map[key][0] for key in all_keys}# poi_index:node_index，这样的map

            self.ignore_node = []

        self.RATE_MAX = self._get_data_rate((0, 0), (0, 0))
        self.RATE_MIN = self._get_data_rate((0, 0), (self.MAP_X, self.MAP_Y))
        self.distance_normalization = np.sqrt(self.MAP_X**2+self.MAP_Y**2)
        self.energy_penalty = 5e-7

        self.Power_flying = {}
        self.Power_hovering = {}
        self._get_energy_coefficient()

        self.noma_config = {
            'noise0_density': 5e-20,
            'bandwidth_subchannel': 40e6 / self.CHANNEL_NUM,
            'p_uav': 5,  # w, 也即34.7dbm
            'p_poi': 0.1,
            'aA': 2,
            'aG': 4,
            'nLoS': 0,  # dB, 也即1w
            'nNLoS': -20,  # dB, 也即0.01w
            'uav_init_height': self.UAV_HEIGHT,
            'psi': 9.6,
            'beta': 0.16,
        }
        if self.ACTION_MODE == 1:
            self.action_space = spaces.Box(min=-1, max=1, shape=(2,))
        elif self.ACTION_MODE == 0:#这一个
            action_space_move = {'uav':spaces.Discrete(self.ACTION_ROOT),'carrier':spaces.Discrete(self.ACTION_ROOT)}
            #太屎山了，这里uav就得是self.uav_action_root,uav这个key压根就没有用过
            if self.NEAR_SELECTION_MODE:
                if not self.TWO_STAGE_MODE:
                    action_space_collect = {'poi_'+key: spaces.MultiDiscrete([self.POI_SELECTION_NUM+1 for _ in range(self.CHANNEL_NUM)]) for key in self.UAV_TYPE}
                else:
                    action_space_collect = {'poi_'+key: spaces.MultiDiscrete([self.POI_SELECTION_NUM for _ in range(self.CHANNEL_NUM)]) for key in self.UAV_TYPE}
            else:
                action_space_collect = {'poi_'+key: spaces.MultiDiscrete([self.POI_NUM+1 for _ in range(self.CHANNEL_NUM)]) for key in self.UAV_TYPE}
                #action_space_collect = {'poi_carrier': spaces.MultiDiscrete([self.POI_NUM for _ in range(self.CHANNEL_NUM)])}
            
            self.action_space = action_space_move#uav:15,carrier:15,
            self.poi_action_space = action_space_collect #poi_uav:[15,15,15,15,15],poi_carrier:[15,15,15,15,15], 每个频道的channel选一个poi
        else:
            self.action_space = spaces.Discrete(1)

        if self.COLLECT_MODE == 0:#这个
            self._poi_value = [self.POI_INIT_DATA for _ in range(self.POI_NUM)]
        elif self.COLLECT_MODE == 1:
            self._poi_value = [0 for _ in range(self.POI_NUM)]
        elif self.COLLECT_MODE == 2:
            self._poi_value = [0 for _ in range(self.POI_NUM)]

        self.poi_property_num = 2 + 1 + 2 #2是pi的2d坐标，1是信息，2是（特定uav到所有poi距离+uav对应carrier到所有poi距离）
        self.agent_property_num = 2 + 1 + 1 #2是agent的2d坐标，1是one-hot编码，1是高度坐标
        info = self.get_env_info()   #info是一个字典，包含了state_shape,obs_shape,poi_shape



        obs_dict = {
            'poi_state':spaces.Box(low=-1, high=1, shape=(1,)), #实际下面poi_state更复杂
            'State': spaces.Box(low=-1, high=1, shape=(info['state_shape'],)), # 长度是 poi_num*3 + (车+飞机)*3
            'available_actions': spaces.Box(low=0, high=1, shape=(self.n_agents, self.ACTION_ROOT)), #[（车+飞机），15],车没有mask，uav会mask，只选前9个action
        }
        for type in self.UAV_TYPE:
            obs_dict[type + "_obs"] = spaces.Box(low=-1, high=1, shape=(self.n_agents, info['obs_shape'][type]))
             #{uav_obs:shape=((车+飞机), （车+飞机）*4+2+兴趣点数*5））
             #uav_obs:shape=((车+飞机), （车+飞机）*4+2+兴趣点数*5））
     
        self.obs_space = spaces.Dict(obs_dict)
        #生成5个space.dict空间，分别是poi_state 、 State（num_poi*3+all_agent*3)、 available_actions(all_agent,15)、 uav_obs(all_agent,obs_dim)、 carrier_obs(all_anget,obs_dim)
        self.observation_space = self.obs_space
        self.reset()
        
        if self.NEAR_SELECTION_MODE and self.TWO_STAGE_MODE:
            self.poi_reset()
            obs = self.poi_get_obs()
            # obs是一个dict，包括
            #id:      shape:(1) 含义：当前agent的id
            #state:   shape:(num_of_agent*(4+4*channel_num)+15*5) 含义：agent的信息，poi的信息。agent的信息包括位置xy，收集时间，是否是当前agent，收集历史(4*channel)，15是当前agent的可选poi数量，5是poi信息（位置xy，数据量，是否被分配，到当前agent的距禿，到当前agent的relay的距离）
            #mask:    shape：(1+poi_num),含义：当前agent的当前channel的可选poi的mask，开头的1表示全都不可选，开头是0表示存在可选的poi
            #carrier: shape: (num_of_carrier, 4+4*channel_num),每行代表一个agent的信息,包括位置xy，收集时间，是否是当前agent，收集历史(4*channel)
            #uav:     shape: (num_of_uav, 4+4*channel_num),每行代表一个agent的信息,包括位置xy，收集时间，是否是当前agent，收集历史(4*channel)
            #poi:     shape:(可选poi数 or poi_num,5), 5代表了所有poi相对于当前agent当前channel而言的信息
            #key-key邻接矩阵：

            obs_dict['poi_state'] = spaces.Box(low=-1, high=1, shape=((obs['state'].shape[0],))) 
        
        self.obs_space = spaces.Dict(obs_dict)
        self.observation_space = self.obs_space#5个space.dict空间，分别是poi_state (num_of_agent*(4+4*channel_num)+15*5) and so on

   


    def load_map(self, map_number):

        if self.config.dict['dataset'] == "beijing":
            from source_code.LaunchMCS.util.Beijing.env_config import dataset_config
        elif self.config.dict['dataset'] == "sf":
            from source_code.LaunchMCS.util.Sanfrancisco.env_config import dataset_config
        elif self.config.dict['dataset'] == "KAIST":
            from source_code.LaunchMCS.util.KAIST.env_config import dataset_config
        elif self.config.dict['dataset'] == 'Rome':
            from source_code.LaunchMCS.util.Rome.env_config import dataset_config
        elif self.config.dict['dataset'] == "multienv":
            setting = import_module('env_setting.env_setting_{}'.format(map_number))
            setting = setting.Setting(None)
            dataset_config = {
                "map_x": 80,
                "map_y": 80,
                "origin_obstacle": setting.V['OBSTACLE'],
                'poi_position': setting.V['DATA'],
                'init_position': {
                    'carrier': [[setting.V['INIT_POSITION'][1] * 5, setting.V['INIT_POSITION'][2] * 5] for _ in
                                range(10)],
                    'uav': [[setting.V['INIT_POSITION'][1] * 5, setting.V['INIT_POSITION'][2] * 5] for _ in range(10)]}
            }
            new_o_all = []
            for o in dataset_config['origin_obstacle']:
                x_min = o[0] * 5
                y_min = o[1] * 5
                x_max = o[0] * 5 + o[2] * 5
                y_max = o[1] * 5 + o[3] * 5
                new_o = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
                new_o_all.append(new_o)

            dataset_config['obstacle'] = new_o_all
            dataset_config['poi_num'] = len(dataset_config['poi_position'])
            dataset_config['poi'] = np.array(dataset_config['poi_position'])[:, 0:2],
            dataset_config['poi'] = np.array(dataset_config['poi'][0]).tolist()
        else:
            raise NotImplementedError
        self.config.dict = {
            **self.config.dict,
            **dataset_config
        }

    def reset(self, map_index=0):
        # if map_index != self.map:
        # print('change map!!!',map_index)
        self.reload_map(str(map_index))

        self.reset_count += 1
        self.uav_trace = {key: [[] for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        self.uav_state = {key: [[] for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        self.uav_energy_consuming_list = {key: [[] for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        self.uav_data_collect = {key: [[] for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        self.uav_voi_collect = {key: [[] for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        self.uav_voi_decline = {key: [[] for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        self.greedy_data_rate = {key: [[1e-5] for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        self.rl_data_rate = {key: [[1e-5] for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}

        self.dead_uav_list = {key: [False for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        self.collect_list = {key: [[] for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        self.step_count = 0
        self.last_collect = [{key: np.zeros((self.NUM_UAV[key], self.POI_NUM)) for key in self.UAV_TYPE} for _ in
                             range(self.MAX_EPISODE_STEP)]

        self.uav_energy = copy.deepcopy(self._uav_energy)
        self.poi_value = copy.deepcopy(self._poi_value)
        self.agent_position = copy.deepcopy(self._uav_position)
        self.poi_position = copy.deepcopy(self._poi_position)

        if self.ROADMAP_MODE:
            self.carrier_node = []
            for i in range(self.NUM_UAV['carrier']):
                raw_node = ox.distance.nearest_nodes(self.nx_g, *self.rm.pygamexy2lonlat(
                    self.agent_position['carrier'][i][0] * self.SCALE,
                    self.agent_position['carrier'][i][1] * self.SCALE))
                self.carrier_node.append(self.nx_g.nodes(data=True)[raw_node]['old_label'])
            self.carrier_node_history = [copy.deepcopy(self.carrier_node)]
            self.wipe_last_things()

        # for render
        self.poi_history = []
        self.aoi_history = [0]
        self.emergency_history = []
        self.episodic_reward_list = {key: [] for key in self.UAV_TYPE}
        self.single_uav_reward_list = {key: [] for key in ['poi_'+t for t in self.UAV_TYPE]+self.UAV_TYPE}
        self.distance = {key: [0 for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        #distance[type][index]表示当前时间不，特定agent需要移动的距离
        return self.get_obs()

    def get_sub_graph(self, graph: nx.Graph, sub_g: list):
        if len(sub_g) == 0:
            return graph
        remove_list = []
        for u, v, data in graph.edges(keys=False, data=True):
            if v in sub_g or u in sub_g:
                remove_list.append([u, v])
        graph.remove_edges_from(remove_list)
        return graph

    def wipe_last_things(self):
        self.last_dst_node = np.zeros((self.NUM_UAV['carrier'], self.action_space['carrier'].n))
        self.last_length = np.zeros((self.NUM_UAV['carrier'], self.action_space['carrier'].n))
        self.last_dst_lonlat = np.zeros((self.NUM_UAV['carrier'], self.action_space['carrier'].n, 2))

    def reload_map(self, map_index: str):
        # if self.config("dataset") == 'multitask':
        #     if self.map == map_index:
        #         return

        #     self.map = map_index
        #     self.config.load_map(map_index)
        # else:
        if self.config("random_map"):
            self.map = map_index
        else:
            self.map = str(self.config("map"))

        if self.ROADMAP_MODE:
            self.nx_g = self.NX_G[self.map]
            self.node_to_poi = self.NODE_TO_POI[self.map]
            self.edge_features = self.EDGE_FEATURES[self.map]
            for node in self.node_to_poi.values():
                self.nx_g.nodes[node]['data'] = 0

            self.visited_nodes_count = {}
            self.poi_nearest_dis = self.POI_NEAREST_DIS[self.map]

            self.reward_co = {}
            if self.config("uav_poi_dis") > 0:
                self.reward_co['uav'] = [1 if dis > self.config("uav_poi_dis") else self.config("colla_co") for dis in
                                         self.poi_nearest_dis]
                self.reward_co['carrier'] = [self.config("colla_co") if dis > self.config("uav_poi_dis") else 1 for dis
                                             in self.poi_nearest_dis]
                print(
                    f"无人机负责的poi大于路网{self.config('uav_poi_dis')}米的 共有{np.mean([1 if dis > self.config('uav_poi_dis') else 0 for dis in self.poi_nearest_dis])}")
            else:
                self.reward_co['uav'] = [1 for _ in range(self.POI_NUM)]
                self.reward_co['carrier'] = [1 for _ in range(self.POI_NUM)]

            # print(f"加载地图{self.map}, 共有顶点{len(self.nx_g.nodes())}, 边{len(self.nx_g.edges())}")

        return

    def  record_info_for_agent(self, main_info: Dict,
                              data_dict: Dict[str, np.ndarray],
                              metric_name: str):
        """
        Convert Dictionary with heterogeneous agents into wandb format.
        """
        for key, value in data_dict.items():
            for i, item in enumerate(value):
                main_info[f'{metric_name}/{key}{i}'] = item
        return main_info
    
    def step(self, action):
        uav_reward = {key: np.zeros([self.NUM_UAV[key]]) for key in self.UAV_TYPE}
        uav_penalty = {key: np.zeros([self.NUM_UAV[key]]) for key in self.UAV_TYPE}
        # uav_data_collect = {key: np.zeros([self.NUM_UAV[key]]) for key in self.UAV_TYPE}

        energy_consumption_all = 0
        all_uav_pois, all_ugv_pois = [], []
        distance = {key: [0 for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}  
        for type in self.UAV_TYPE:
            for uav_index in range(self.NUM_UAV[type]):
                a = action[type][uav_index]
                if self.ROADMAP_MODE and type == 'carrier':

                    self.carrier_node[uav_index] = int(self.last_dst_node[uav_index][a])
                    self.visited_nodes_count[self.carrier_node[uav_index]] = self.visited_nodes_count.get(
                        self.carrier_node[uav_index], 0) + 1
                    assert int(self.carrier_node[uav_index]) != 0
                    new_x, new_y = self.rm.lonlat2pygamexy(self.last_dst_lonlat[uav_index][a][0],
                                                           self.last_dst_lonlat[uav_index][a][1])
                    my_position = (new_x / self.SCALE, new_y / self.SCALE)
                    dis = self.last_length[uav_index][a]
                    energy_consuming = self._cal_energy_consuming(dis, type)
                    #energy_consumption_all += energy_consuming

                    if uav_index == self.NUM_UAV[type] - 1:
                        self.carrier_node_history.append(copy.deepcopy(self.carrier_node))
                else:
                    new_x, new_y, dis, energy_consuming = self._cal_uav_next_pos(uav_index, a,
                                                                                 type)
                    Flag = self._judge_obstacle(my_position, (new_x, new_y))
                    if not Flag:
                        my_position = (new_x, new_y)
                    else:
                        my_position = self.agent_position[type][uav_index]

                my_trace = self.uav_trace[type][uav_index]
                # pos_history_size = len(my_trace)
                # if pos_history_size > 0:
                #     # Nearest Neighbor Intrinsic Reward
                #     average_dis = np.average(np.linalg.norm(np.asarray(my_trace)
                #                                             -
                #                                             np.tile(my_position, (pos_history_size, 1)),
                #                                             axis=1))
                #     uav_reward[type][uav_index] += 1e-3 * average_dis
                my_trace.append(my_position)
                self.agent_position[type][uav_index] = my_position
                self._use_energy(type, uav_index, energy_consuming)
                energy_consumption_all += energy_consuming
                uav_reward[type][uav_index] -= energy_consuming * self.energy_penalty
                #uav_reward[type][uav_index] -= energy_consuming * 1e-6
                distance[type][uav_index] += dis
          
        if self.NOMA_MODE:#true
            relay_dict = self._relay_association()
            if self.POI_DECISION_MODE:
                # a = action['poi_carrier'][0]
                # sorted_access = self._access_determin(self.CHANNEL_NUM,a)
                sorted_access = {'uav': [], 'carrier': []}
                for type in self.UAV_TYPE:
                    for uav_index in range(self.NUM_UAV[type]):
                        if self.NEAR_SELECTION_MODE:
                            poi_list = self._cal_distance(np.array(self._uav_position[type][uav_index]), self.poi_position, type)
                            order = poi_list.argsort()
                            a = action['poi_'+type][uav_index] # [number_channel]
                            assert len(a) == self.CHANNEL_NUM
                            select_poi = order[a]
                            sorted_access[type].append(select_poi.tolist())
                        else:
                            a = action['poi_'+type][uav_index] # [number_channel]
                            assert len(a) == self.CHANNEL_NUM
                            sorted_access[type].append(a)
                            
            elif self.TWO_STAGE_MODE:  #这里直接返回是没有问题的，因为two_stage_mode，在collect的MDP结束后，会调用continue_step函数，再次返回s，reward，done，info等信息
                self.distance = distance
                self.poi_reset()
                self.poi_step_status = True
                #print("i returned nothing because i dont use parameter poi_decision_mode")
                return {},{},{},{}
            else:
                sorted_access = self._access_determin(self.CHANNEL_NUM)
                #sorted_access:dict,{uav: 二维list, carrier: 二维list}，二维list的shape：(num_uav,channel_num),含义是每个agent的每个channel对应的poi的index
                #sorted_access的每个agent的每个channel优先选最近的poi，没有可用的poi就用-1填充
        
        # if distance['carrier'][0] == 0:
        #     print(distance)
        all_collected = {my_type: np.zeros(self.NUM_UAV[my_type]) for my_type in self.UAV_TYPE}
        all_data_rate = {my_type: np.zeros(self.NUM_UAV[my_type]) for my_type in self.UAV_TYPE}
        # all_ratio = {my_type: np.zeros(self.NUM_UAV[my_type]) for my_type in self.UAV_TYPE}
        for type in self.UAV_TYPE:
            for uav_index in range(self.NUM_UAV[type]):
                collect_time = max(0, self.TIME_SLOT - distance[type][uav_index] / self.UAV_SPEED[type])

                if self.NOMA_MODE:
                    r, collected_data, data_rate, uav_poi, ugv_poi, collected_list, temp_poi_aoi_list = (
                        self._collect_data_by_noma(type, uav_index, relay_dict, sorted_access,
                                                   collect_time))
                    all_uav_pois.extend(uav_poi)
                    all_ugv_pois.extend(ugv_poi)
                    all_collected[type][uav_index] = collected_data
                    all_data_rate[type][uav_index] = data_rate
                    self.greedy_data_rate[type][uav_index].append(data_rate)
                    # if type == 'uav':
                    #     all_ratio[type][uav_index] = g2a_rate / (relay_rate + 1e-10)
                else:
                    r, collected_data = self._collect_data_from_poi(type, uav_index, collect_time)

                self.uav_data_collect[type][uav_index].append(collected_data)
                #下面求VOI的部分，added by zf，24.11.18
                total_voi_decline = 0
                for channel_index in range(self.CHANNEL_NUM):
                    collected_data_this_channel = collected_list[channel_index]
                    aoi_this_channel = temp_poi_aoi_list[channel_index]
                    voi_lambda = min(collected_data_this_channel/self.USER_DATA_AMOUNT, aoi_this_channel)
                    Decline = (exp(self.VOI_K * (aoi_this_channel - voi_lambda) ) - exp(self.VOI_K*aoi_this_channel))/self.VOI_K
                    assert voi_lambda >= Decline
                    voi_decline = (1-self.VOI_BETA)*self.USER_DATA_AMOUNT*(voi_lambda-Decline)
                    total_voi_decline += voi_decline
                
                self.uav_voi_collect[type][uav_index].append(collected_data - total_voi_decline)
                self.uav_voi_decline[type][uav_index].append(total_voi_decline)
                #VOI部分结束,added by zf,24.11.18

                uav_reward[type][uav_index] += r * (10 ** -3)  # * (2**-4)
                # print( uav_reward[type][uav_index])

                if type == 'uav':
                    # dis_reward =  self._cal_distance(self.agent_position['carrier'][relay_dict[uav_index]],
                    # self.agent_position['uav'][uav_index])*0.0001
                    # uav_reward[type][uav_index] +
                    uav_reward['carrier'][relay_dict[uav_index]] += r * (10 ** -3) / 5

                if type == 'carrier' and self.config("carrier_explore_reward"):
                    # print(uav_reward[type][uav_index])
                    uav_reward[type][uav_index] -= math.log(
                        self.visited_nodes_count[self.carrier_node[uav_index]] + 1) * 0.1

                    # print(-math.log(self.visited_nodes_count[self.carrier_node[uav_index]]+1) * 0.05)

        if self.COLLECT_MODE == 1 or self.COLLECT_MODE == 2:
            self.check_arrival()
            self.aoi_history.append(np.mean(np.asarray(self.poi_value) / self.USER_DATA_AMOUNT))
        
            self.emergency_history.append(
                    np.mean([1 if aoi / self.USER_DATA_AMOUNT >= self.AOI_THRESHOLD else 0 for aoi in self.poi_value]))
            aoi_reward = self.aoi_history[-2] - self.aoi_history[-1]
            # aoi_reward -= self.emergency_history[-1] * self.THRESHOLD_PENALTY
            #
            # for type in self.UAV_TYPE:
            #     for uav_index in range(self.NUM_UAV[type]):
            #         uav_reward[type][uav_index] -= self.emergency_history[-1] * self.THRESHOLD_PENALTY

            for type in self.UAV_TYPE:
                type_sum = sum(uav_reward[type])
                for uav_index in range(self.NUM_UAV[type]):
                    if self.CENTRALIZED:
                        if self.reward_type == 'prod':
                            aux = 1e-6 * all_collected[type][uav_index] * all_data_rate[type][uav_index]
                        elif self.reward_type == 'square':
                            aux = 5e-7 * all_collected[type][uav_index] ** 2
                        elif self.reward_type == 'prod_thre':
                            aux = (1e-6 * all_collected[type][uav_index] *
                                   max(0, all_data_rate[type][uav_index] - self.DATA_RATE_THRE))
                        elif self.reward_type == 'sum':
                            aux = 1e-3 * (all_collected[type][uav_index] + all_data_rate[type][uav_index])
                        else:
                            aux = 0

                        uav_reward[type][uav_index] += aoi_reward  + aux
                        #uav_reward[type][uav_index] = aoi_reward + aux
                        if self.dis_bonus:
                            if type == 'carrier':
                                dis = 0
                                for other_uav in range(self.NUM_UAV[type]):
                                    dis += self._cal_distance(self.agent_position[type][uav_index],
                                                              self.agent_position[type][other_uav], type=type)
                                dis /= (self.NUM_UAV[type] - 1)
                                uav_reward[type][uav_index] += 5e-5 * min(dis, 500)
        # print("step_count:",self.step_count)
        # print("if_done?",self._is_episode_done())
        done = self._is_episode_done()
        self.step_count += 1

        self.poi_history.append({
            'pos': copy.deepcopy(self.poi_position).reshape(-1, 2),
            'val': copy.deepcopy(self.poi_value)})

        info = {}
        info_old = {}

        if self.NOMA_MODE:
            self.record_info_for_agent(info, all_collected, "collected_data")
            self.record_info_for_agent(info, all_data_rate, "data_rate")
            # self.record_info_for_agent(info, all_ratio, "g2a_relay_ratio")
            # data rate hinge reward
            # uav_reward['uav'][uav_id] += max(self.DATA_RATE_THRE - data_rate, 0)

        info_old.update({"uav_pois": all_uav_pois, "ugv_pois": all_ugv_pois})

        if done:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                info = self.summary_info(info)
                info_old = copy.deepcopy(info)
                # updating info_old means the poi visit history is not necessary in every trajectory during training
                info_old.update({"uav_pois": all_uav_pois, "ugv_pois": all_ugv_pois})
                info = self.save_trajectory(info)

        global_reward = {}
        for type in self.UAV_TYPE:
            global_reward[type] = np.mean(uav_reward[type]) + np.mean(uav_penalty[type])
            self.episodic_reward_list[type].append(global_reward[type])
            self.single_uav_reward_list[type].append((uav_reward[type] + uav_penalty[type]).tolist())
        obs = self.get_obs()
        info_old['step'] = self.step_count
        
        if self.POI_DECISION_MODE:
            collect_reward = {'poi_'+key: np.zeros([self.NUM_UAV[key]]) for key in self.UAV_TYPE}
            for type in self.UAV_TYPE:
                for uav_index in range(self.NUM_UAV[type]):
                    collect_reward['poi_'+type][uav_index]+= all_data_rate[type][uav_index]*1e-3
                self.single_uav_reward_list['poi_'+type].append(collect_reward['poi_'+type].tolist())
        else:
            collect_reward = {}
        return obs, {**uav_reward,**collect_reward}, done, info_old

    def summary_info(self, info):
        if self.COLLECT_MODE == 1:
            total_data_generated = self.POI_INIT_DATA * self.POI_NUM
            data_collection_ratio = 1 - np.sum(np.sum(self.poi_value)) / (total_data_generated)
            sep_collect = {self.map + 'f_data_collection_ratio_' + type: np.sum(
                [sum(self.uav_data_collect[type][uav_index]) for uav_index in range(self.NUM_UAV[type])]) / (
                total_data_generated)
                           for type in self.UAV_TYPE}
        else:
            total_data_generated = self.step_count * self.USER_DATA_AMOUNT * self.POI_NUM
            data_collection_ratio = 1 - np.sum(np.sum(self.poi_value)) / total_data_generated
            sep_collect = {'f_data_collection_ratio_' + type: np.sum(
                [sum(self.uav_data_collect[type][uav_index]) for
                 uav_index in range(self.NUM_UAV[type])]) / total_data_generated
                           for type in self.UAV_TYPE}
        total_data_collected = data_collection_ratio*total_data_generated
        info['Metric/Data_Throughput'] = total_data_collected/(self.step_count*self.n_agents*self.MAX_EPISODE_STEP)
         
        info['Metric/'+DATA_COLLECTION_RATIO_NAME] = data_collection_ratio.item()
        info['Metric/'+EPISODIC_AOI_NAME] = np.mean(self.aoi_history)

        info.update(sep_collect)

        info[f"Map: {self.map}/a_data_collection_ratio"] = data_collection_ratio.item()
        info[f"Map: {self.map}/a_episodic_aoi"] = np.mean(self.aoi_history)
        info['map'] = int(self.map)

        t_all = 0
        uav_tall = 0
        data_collect_all = 0
        aoi = np.mean(self.aoi_history)
        greedy_rate_all = 0
        rl_rate_all = 0
        for type in self.UAV_TYPE:
            t_e = np.sum(np.sum(self.uav_energy_consuming_list[type]))
            scaled_t_e = t_e / 1000
            t_all += scaled_t_e
            if type =='uav': uav_tall += scaled_t_e
            data_e = np.sum(
                [sum(self.uav_data_collect[type][uav_index]) for uav_index in range(self.NUM_UAV[type])])
            data_collect_all += data_e
            info['f_total_energy_consuming_' + type] = t_e.item()
            info['f_energy_consuming_ratio_' + type] = t_e / (self.NUM_UAV[type] * self.INITIAL_ENERGY[type])
            info['f_sensing_efficiency_' + type] = data_e.item() / (aoi * scaled_t_e)
            
            info['RL/f_greedy_data_rate_' + type] =   sum([sum(self.greedy_data_rate[type][uav_index])/len(self.greedy_data_rate[type][uav_index]) for uav_index in range(self.NUM_UAV[type])])/self.NUM_UAV[type]
            info['RL/f_rl_data_rate_' + type] =   sum([sum(self.rl_data_rate[type][uav_index])/len(self.rl_data_rate[type][uav_index]) for uav_index in range(self.NUM_UAV[type])])/(self.NUM_UAV[type])
            
            greedy_rate_all += sum([sum(self.greedy_data_rate[type][uav_index]) for uav_index in range(self.NUM_UAV[type])])
            rl_rate_all += sum([sum(self.rl_data_rate[type][uav_index]) for uav_index in range(self.NUM_UAV[type])])

        
        info['RL/f_greedy_data_rate'] =  greedy_rate_all/(self.step_count*sum([self.NUM_UAV[type] for type in self.UAV_TYPE]))
        info['RL/f_rl_data_rate'] =   rl_rate_all/(self.step_count*sum([self.NUM_UAV[type] for type in self.UAV_TYPE]))
        
        
        info['Metric/'+SENSING_EFFICIENCY_NAME] = data_collect_all.item() / (aoi * t_all)
        info['Metric/a_sensing_efficiency_only_uav_energy'] = data_collect_all.item() / (aoi * uav_tall)
        info['Metric/a_sensing_efficiency(Mbps_kwh)'] = 3600000/1000 * data_collect_all.item() / (self.POI_NUM*self.step_count*aoi * t_all)
      
      
        #new metircs by zf
        #AOI归一化阈值，n_all_agent*n_channel是一个collect能照顾到的最大的poi数量，用总共poi数量除以这个值，得到k个step才能逛完所有的poi，因此用6*k作为poi能容忍的最大时间
        aoi_threshold = 6 * self.POI_NUM / (self.n_agents * self.CHANNEL_NUM)
        aoi_norm = aoi/aoi_threshold
        
        data_collect_norm = data_collect_all.item() / total_data_generated
        all_energy_consuming = 0
        for type in self.UAV_TYPE:
            all_energy_consuming += np.sum(np.sum(self.uav_energy_consuming_list[type]))
        energy_consuming_norm = all_energy_consuming / (self.INITIAL_ENERGY['uav'] * self.NUM_UAV['uav'] + self.INITIAL_ENERGY['carrier'] * self.NUM_UAV['carrier'])
        info['Metric/zf/normalized_efficiency'] = data_collect_norm/(aoi_norm*energy_consuming_norm)

        voi_decline_all = sum([sum(self.uav_voi_decline[type][uav_index]) for type in self.UAV_TYPE for uav_index in range(self.NUM_UAV[type])])
        voi_all = sum([sum(self.uav_voi_collect[type][uav_index]) for type in self.UAV_TYPE for uav_index in range(self.NUM_UAV[type])])
        voi_all_norm = voi_all/total_data_generated
        void_decline_norm = voi_decline_all/total_data_generated
        info['Metric/zf/voi_norm'] = voi_all_norm
        info['Metric/zf/voi_decline_norm'] = void_decline_norm
        #计算AOI方差,需要平衡这些数值的大小，方便组合，在单步reward和总体metrics，都要考虑到这个问题
        aoi_var_norm = np.var(self.aoi_history)/1000
        info['Metric/zf/aoi_var_norm'] = aoi_var_norm
        info['Metric/zf/energy_consuming_norm'] = energy_consuming_norm
        info['Metric/zf/aoi_norm'] = aoi_norm
        info['Metric/zf/data_collection_ratio_norm_from_PoI'] = data_collection_ratio
        info['Metric/zf/data_collection_ratio_norm_from_Agent'] = data_collect_norm
        #从agent收集的角度计算的收集率，和从poi收集的角度计算的收集率应该是一样的
        assert (data_collect_norm / data_collection_ratio) > 0.99 and (data_collect_norm / data_collection_ratio) < 1.01
        assert data_collect_norm < 1 and data_collection_ratio < 1
        info['Metric/zf/normalized_efficiency_with_aoivar'] = data_collect_norm/(aoi_norm*energy_consuming_norm*aoi_var_norm)
        info['Metric/zf/normalized_efficiency_w_aoivar_wo_energy'] = data_collect_norm/(aoi_norm*aoi_var_norm)
        info['Metric/zf/noramlied_efficiency_with_aoi'] = voi_all_norm/(aoi_norm*energy_consuming_norm*aoi_var_norm)
        info['Metric/zf/noramlied_efficiency_with_aoi_wo_energy'] = voi_all_norm/(aoi_norm*aoi_var_norm)
        info['Metric/zf/noramlied_efficiency_with_aoi_decline'] = data_collect_norm/(energy_consuming_norm*aoi_var_norm*void_decline_norm*aoi_norm)
        info['Metric/zf/noramlied_efficiency_with_aoi_decline_wo_energy'] = data_collect_norm/(aoi_var_norm*void_decline_norm*aoi_norm)

        ##单独看每个type的metircs情况
        for type in self.UAV_TYPE:
            voi_decline_all_temp = sum([sum(self.uav_voi_decline[type][uav_index]) for uav_index in range(self.NUM_UAV[type])])
            voi_all_temp = sum([sum(self.uav_voi_collect[type][uav_index]) for uav_index in range(self.NUM_UAV[type])])
            voi_all_norm_temp = voi_all_temp/total_data_generated
            void_decline_norm_temp = voi_decline_all_temp/total_data_generated
            info[f'Metric/zf/voi_norm_{type}'] = voi_all_norm_temp
            info[f'Metric/zf/voi_decline_norm_{type}'] = void_decline_norm_temp
            energy_temp = np.sum(np.sum(self.uav_energy_consuming_list[type]))
            energy_consuming_norm_temp = energy_temp / (self.INITIAL_ENERGY[type] * self.NUM_UAV[type])
            info[f'Metric/zf/energy_consuming_norm_{type}'] = energy_consuming_norm_temp
            data_collected_temp = np.sum([sum(self.uav_data_collect[type][uav_index]) for uav_index in range(self.NUM_UAV[type])])
            data_collect_norm_temp = data_collected_temp / total_data_generated
            info[f'Metric/zf/data_collection_ratio_norm_from_Agent_{type}'] = data_collect_norm_temp
         #end new metrics by zf



        #info['Metric/Data_Throughput'] = data_collect_all.item() / (aoi)
        # if self.TWO_STAGE_MODE:
        #     info['Metric/Data_Throughput_old'] = rl_rate_all/(self.step_count*sum([self.NUM_UAV[type] for type in self.UAV_TYPE]))
        # else:
        #     info['Metric/Data_Throughput_old'] = greedy_rate_all/(self.step_count*sum([self.NUM_UAV[type] for type in self.UAV_TYPE]))
        
        info['a_energy_consumption_ratio'] = t_all * 1000 / sum(
            [self.NUM_UAV[type] * self.INITIAL_ENERGY[type] for type in self.UAV_TYPE])
        info['Metric/a_energy_consumption_ratio'] = info['a_energy_consumption_ratio']
        reward_per_agent = {}
        for type in self.UAV_TYPE:
            move_reward_list = []
            collect_reward_list = []
            for agent_id in range(self.NUM_UAV[type]):
                move_reward = sum([r[agent_id] for r in self.single_uav_reward_list[type]])
                collect_reward =  sum([r[agent_id] for r in self.single_uav_reward_list['poi_'+type]])
                move_reward_list.append(move_reward)
                collect_reward_list.append(collect_reward)
            reward_per_agent['poi_'+type] = collect_reward_list
            reward_per_agent[type] = move_reward_list
        info['reward_list'] = reward_per_agent
        
        for type in ['uav','carrier','poi_uav','poi_carrier']:
            info[f'g_{type}_reward_sum'] = np.array(self.single_uav_reward_list[type]).sum()
        return info

    def save_trajectory(self, info):
        if self.TEST_MODE:
            for type in self.UAV_TYPE:
                temp_info = {}
                temp_info['uav_trace'] = self.uav_trace[type]
                max_len = max((len(l) for l in self.uav_data_collect[type]))
                new_matrix = list(
                    map(lambda l: l + [0] * (max_len - len(l)), self.uav_data_collect[type]))
                temp_info['uav_collect'] = np.sum(new_matrix, axis=0).tolist()
                temp_info['reward_history'] = self.episodic_reward_list[type]
                temp_info['uav_reward'] = self.single_uav_reward_list[type]

                if self.ROADMAP_MODE and type == 'carrier':
                    temp_info['carrier_node_history'] = self.carrier_node_history

                info[type] = temp_info

            info['map'] = self.map
            info['config'] = self.config.dict
            info['poi_history'] = self.poi_history
            path = self.args['save_path'] + '/map_{}_count_{}.txt'.format(self.map, self.SAVE_COUNT)
            self.save_variable(info, path)
            info = {}

        return info

    def save_trajectory_from_outside(self, info):
        for type in self.UAV_TYPE:
            temp_info = {}
            temp_info['uav_trace'] = self.uav_trace[type]
            max_len = max((len(l) for l in self.uav_data_collect[type]))
            new_matrix = list(
                map(lambda l: l + [0] * (max_len - len(l)), self.uav_data_collect[type]))
            temp_info['uav_collect'] = np.sum(new_matrix, axis=0).tolist()
            temp_info['reward_history'] = self.episodic_reward_list[type]
            temp_info['uav_reward'] = self.single_uav_reward_list[type]

            if self.ROADMAP_MODE and type == 'carrier':
                temp_info['carrier_node_history'] = self.carrier_node_history

            info[type] = temp_info

        info['map'] = self.map
        info['config'] = self.config.dict
        info['poi_history'] = self.poi_history
        path = self.args['save_path'] + \
               '/map_{}_{}.txt'.format(self.map, self.reset_count)
        for key in shared_feature_list:
            # if key == 'rm':
            #     continue
            try:
                del info['config'][key]
            except KeyError:
                continue
        self.save_variable(info, path)

    def save_variable(self, v, filename):
        # print('save variable to {}'.format(filename))
        f = open(filename, 'wb')
        pickle.dump(v, f)
        f.close()
        return filename

    def p_seed(self, seed=None):
        pass

    def _cal_distance(self, pos1, pos2, type):
        """
        pos1: [x,y],或者二维array，表示n个点的坐标
        pos2: [x,y]
        return：如果pos1和pos2都是单个点，直接返回距离，如果是两个array，返回n个点的距离的ndarray
        """

        height = self.UAV_HEIGHT if type == 'uav' else 0

        if isinstance(pos1, np.ndarray) and isinstance(pos2, np.ndarray):
            # UAV必须在后面，poi在前面
            while pos1.ndim < 2:
                pos1 = np.expand_dims(pos1, axis=0)
            while pos2.ndim < 2:
                pos2 = np.expand_dims(pos2, axis=0)
            # expanded to 3dim
            pos1_all = np.concatenate([pos1 * self.SCALE, np.zeros((pos1.shape[0], 1))], axis=1)
            pos2_all = np.concatenate([pos2 * self.SCALE, np.ones((pos2.shape[0], 1)) * height], axis=1)
            distance = np.linalg.norm(pos1_all - pos2_all, axis=1)
        else:
            assert len(pos1) == len(
                pos2) == 2, 'cal_distance function only for 2d vector'
            distance = np.sqrt(
                np.power(pos1[0] * self.SCALE - pos2[0] * self.SCALE, 2) + np.power(pos1[1] * self.SCALE
                                                                                    - pos2[1] * self.SCALE,
                                                                                    2) + np.power(
                    height, 2))
        return distance

    def _cal_theta(self, pos1, pos2, height=None):
        if len(pos1) == len(pos2) and len(pos2) == 2:
            r = np.sqrt(np.power(pos1[0] * self.SCALE - pos2[0] * self.SCALE, 2) + np.power(
                pos1[1] * self.SCALE - pos2[1] * self.SCALE, 2))
            h = self.UAV_HEIGHT if height is None else height
            theta = math.atan2(h, r)
        elif len(pos1) == 2:
            repeated_pos1 = np.tile(pos1, len(pos2)).reshape(-1, 2)
            r = self._cal_distance(repeated_pos1, pos2, type='carrier')
            h = self.UAV_HEIGHT if height is None else height
            theta = np.arctan2(h, r)
        return theta

    def _cal_energy_consuming(self, move_distance, type):
        moving_time = move_distance / self.UAV_SPEED[type]
        hover_time = self.TIME_SLOT - moving_time
        if type == 'carrier':
            moving_time = min(20, move_distance / 15)
            return self.Power_flying[type] * moving_time + self.Power_hovering[type] * hover_time
        else:
            return self.Power_flying[type] * moving_time + self.Power_hovering[type] * hover_time

    def _cal_uav_next_pos(self, uav_index, action, type):
        if self.ACTION_MODE == 1:
            dx, dy = self._get_vector_by_theta(action)
        else: # discrete action
            dx, dy = self._get_vector_by_action(int(action))
            # dx, dy = self._get_vector_by_smart_action_(uav_index,int(action))
        distance = np.sqrt(np.power(dx * self.SCALE, 2) +
                           np.power(dy * self.SCALE, 2))
        energy_consume = self._cal_energy_consuming(distance, type)
        if self.uav_energy[type][uav_index] >= energy_consume:
            new_x, new_y = self.agent_position[type][uav_index][0] + dx, self.agent_position[type][uav_index][1] + dy
        else:
            new_x, new_y = self.agent_position[type][uav_index]

        return new_x, new_y, distance, min(self.uav_energy[type][uav_index], energy_consume)

    def _relay_association(self):
        '''
        每个无人机就近选择relay的无人车
        :return: relay_dict, 形如 {0: 1, 1: 1, 2: 1, 3: 1}
        '''
        # if self.config("fixed_relay"):
        #     return {i:i for i in range(self.NUM_UAV['uav'])}

        relay_dict = {uav_index:uav_index for uav_index in range(self.NUM_UAV['uav'])}
        #available_car = [1 for _ in range(self.NUM_UAV['carrier'])]
        #for uav_index in range(self.NUM_UAV['uav']):
            #relay_dict[uav_index] = uav_index
            # dis_mat = [self._cal_distance(self.agent_position['uav'][uav_index], car_pos, 'uav') for car_pos in
            #            self.agent_position['carrier']]
            # for index in range(self.NUM_UAV['carrier']):
            #     if available_car[index] == 0 and self.config("fixed_relay"):
            #         dis_mat[index] = 999999999

            # car_index = np.argmin(dis_mat)
            # relay_dict[uav_index] = car_index
            # available_car[car_index] = 0

        return relay_dict

    
    def _access_determin(self, CHANNELS,carrier1_actions=None):
        '''
        limited_collection: 限制采集范围,在carrier和uav中不同，是9999还是500？
        :param CHANNELS: 信道数量
        :return: sorted_access, 形如 {'uav': [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 'carrier': [[0, 0, 0]]}
        '''

        carrier_index = []
        sorted_access = {'uav': [], 'carrier': []}
        collect_range = {type:self.LIMIT_RANGE[type]*self.mask_range for type in self.UAV_TYPE} if self.config("limited_collection") else {key: 99999 for key in
                                                                                      self.UAV_TYPE}
        for type in ['carrier']:
            for index,uav_pos in enumerate(self.agent_position[type]):
                cur_poi_value = np.asarray(self.poi_value)
                distances = self._cal_distance(self.poi_position,np.array(uav_pos), type)
                distance_mask = np.logical_and(distances <= self.COLLECT_RANGE[type],
                                               cur_poi_value > 0)
                _, max_intake = compute_capacity_G2G(self.noma_config, distances)
                max_intake = max_intake / 1e6 * self.TIME_SLOT
                dis_list = -np.where(cur_poi_value < max_intake, cur_poi_value, max_intake) * distance_mask
                pois = self.get_valid_pois(CHANNELS, dis_list, distances, self.COLLECT_RANGE[type], type)
                # rate_list = []
                # for index,poi_pos in enumerate(self.poi_position):
                #     dis =  self._cal_distance(uav_pos, poi_pos,type)
                #     if dis <=  self.COLLECT_RANGE[type] and self.poi_value[index]>self.USER_DATA_AMOUNT*5:
                #         _, capacity_i = compute_capacity_G2G(self.noma_config,dis)
                #         rate_list.append(-min(self.poi_value[index],capacity_i/1e6*self.TIME_SLOT)) 
                #     else:
                #         rate_list.append(0)
                # #dis_list = np.array([max(1, self._cal_distance(uav_pos, poi_pos,0)) for poi_pos in self.poi_position])
                # dis_list = np.array(rate_list)
                # pois =  np.argsort(dis_list)[:CHANNELS]
                # #pois = [x for x in pois if self.poi_value[x]>self.USER_DATA_AMOUNT*2 and  self._cal_distance(uav_pos, self.poi_position[x])< self.COLLECT_RANGE[type]] #车一定做限制
                # pois = [x for x in pois if  self._cal_distance(uav_pos, self.poi_position[x],type) < self.COLLECT_RANGE[type]]
                # while len(pois) < CHANNELS: pois.append(-1)
                if index == 0 and carrier1_actions is not None: pois = carrier1_actions
                sorted_access[type].append(pois)
                carrier_index.extend(pois)

        for type in ['uav']:
            for uav_pos in self.agent_position[type]:
                distances = np.clip(self._cal_distance(self.poi_position, np.array(uav_pos), type), a_min=1,
                                    a_max=np.inf)
                #np.put(distances, carrier_index, 9999999)
                pois = self.get_valid_pois(CHANNELS, distances, distances, collect_range[type], type)
                # dis_list = np.array([max(1, self._cal_distance(uav_pos, poi_pos, type)) if index not in carrier_index else 9999999 for index,poi_pos in enumerate(self.poi_position)])
                # pois =  np.argsort(dis_list)[:CHANNELS]
                # pois = [x for x in pois if self._cal_distance(uav_pos, self.poi_position[x], type)< collect_range[type]]
                # while len(pois) < CHANNELS: pois.append(-1)
                sorted_access[type].append(pois)

        return sorted_access

    def get_valid_pois(self, CHANNELS, dis_list, distances, threshold, type):
        pois = np.argsort(dis_list)[:CHANNELS]
        # print("type",type)
        # print("threshold",threshold)
        # print("distatances",distances)
        # print("pois:前channel个吞吐量较大的poi的索引",pois)
        # print("distances[pois]:前channel个吞吐量较大的poi的距离",distances[pois])
        pois = pois[distances[pois] < threshold]
        if type == 'carrier':
            pois = pois[dis_list[pois] < 0]
        size = CHANNELS - len(pois)
        if size > 0:
            pois = np.concatenate([pois, np.full(size, fill_value=-1)])
        return pois.tolist()

    def _collect_data_by_noma(self, type, uav_index, relay_dict, sorted_access, collect_time=0):

        reward_list = []
        collect_list = []
        data_rate_list = []
        relay_rate_list = []
        g2a_rate_list = []
        uav_pois = []
        ugv_pois = []
        temp_poi_aoi_list = []
        if self.DEBUG_MODE:
            print(f'====设备类型={type}，ID={uav_index}====')
        if collect_time <= 0:
            return (0, 0, 0 , uav_pois, ugv_pois, [0]*self.CHANNEL_NUM, [0]*self.CHANNEL_NUM)
        for channel in range(self.CHANNEL_NUM):
            poi_i_index = sorted_access[type][uav_index][channel]  # 确定从哪个poi收集数据
            if poi_i_index == -1:
                data_rate_list.append(0)
                collect_list.append(0)
                temp_poi_aoi_list.append(0)
                continue
            poi_i_pos = self.poi_position[poi_i_index]
            # 机和车以不同的方式计算capacity，遵循noma模型
            if type == 'carrier':
                # car从poi_i收集数据
                ugv_pois.append(tuple(poi_i_pos))
                car_pos = self.agent_position[type][uav_index]
                sinr_i, capacity_i = compute_capacity_G2G(self.noma_config,
                                                          self._cal_distance(poi_i_pos, car_pos, type)
                                                          )
                capacity_i /= 1e6
                if self.DEBUG_MODE:
                    poi_i_val = self.USER_DATA_AMOUNT
                    print(
                        f'在第{channel}个信道中，从ID={poi_i_index} poi收集数据，'
                        f'capacity={capacity_i}, collect_time ={collect_time}, '
                        f'车PoI的距离是{self._cal_distance(poi_i_pos, car_pos, 0)},'
                        f'收集量={min(capacity_i * collect_time, self.poi_value[poi_i_index]) / poi_i_val}')
            else:
                assert type == 'uav'
                uav_pois.append(tuple(poi_i_pos))
                relay_car_index = relay_dict[uav_index]  # 确定当前uav转发给哪个car
                uav_pos, relay_car_pos = self.agent_position[type][uav_index], self.agent_position['carrier'][
                    relay_car_index]
                # uav从poi_i收集数据，但poi_j会造成干扰
                poi_j_index = sorted_access['carrier'][relay_car_index][channel]

                if self._cal_distance(uav_pos, relay_car_pos, type) >= self.LIMIT_RANGE[type]*self.mask_range and self.config(
                        "limited_collection"):
                    R_G2A = R_RE = 1
                    if self.DEBUG_MODE: print(
                        f"车机距离：{self._cal_distance(uav_pos, relay_car_pos, type)}, 无人机与poi i的距离{self._cal_distance(poi_i_pos, uav_pos, type)}")
                else:
                    if poi_j_index != -1:
                        poi_j_pos = self.poi_position[poi_j_index]

                        sinr_G2A, R_G2A = compute_capacity_G2A(self.noma_config,
                                                               self._cal_distance(poi_i_pos, uav_pos, type),
                                                               self._cal_distance(poi_j_pos, uav_pos, type),
                                                               )
                        sinr_RE, R_RE = compute_capacity_RE(self.noma_config,
                                                            self._cal_distance(uav_pos, relay_car_pos, type),
                                                            self._cal_distance(poi_i_pos, relay_car_pos, 'carrier'),
                                                            self._cal_distance(poi_j_pos, relay_car_pos, 'carrier'),
                                                            )
                    else:
                        uav_pos, relay_car_pos = self.agent_position[type][uav_index], self.agent_position['carrier'][
                            relay_car_index]
                        poi_j_pos = (99999, 999999)
                        sinr_G2A, R_G2A = compute_capacity_G2A(self.noma_config,
                                                               self._cal_distance(poi_i_pos, uav_pos, type),
                                                               -1,
                                                               )
                        sinr_RE, R_RE = compute_capacity_RE(self.noma_config,
                                                            self._cal_distance(uav_pos, relay_car_pos, type),
                                                            self._cal_distance(poi_i_pos, relay_car_pos, 'carrier'),
                                                            -1,
                                                            )
                    if self.DEBUG_MODE:
                        print(f"车机距离：{self._cal_distance(uav_pos, relay_car_pos, type)}, "
                              f"无人机与poi i的距离{self._cal_distance(poi_i_pos, uav_pos, type)},"
                              f"无人机与poi_j的距离{self._cal_distance(poi_j_pos, uav_pos, type)}")

                #  在小地图中（如KAIST），由于poi的功率远小于UAV的功率，
                #  所以更多情况下R_G2A < R_RE，前者是瓶颈。
                #  在大地图中，由于UAV和UGV距离很远，中继信号（后者）大部分时间是瓶颈。
                capacity_i = min(R_G2A, R_RE) / 1e6  # 取两段信道的较小值
                g2a_rate_list.append(R_G2A / 1e6)
                relay_rate_list.append(R_RE / 1e6)
                if self.DEBUG_MODE:
                    print(
                        f"在第{channel}个信道中，R_G2A={R_G2A / 1e6}，"
                        f"R_RE={R_RE / 1e6}，前者是后者的{'%.3f' % (R_G2A / R_RE * 100)}%")
                if self.DEBUG_MODE:
                    poi_i_val = self.USER_DATA_AMOUNT
                    print(
                        f"在第{channel}个信道中，"
                        f"从ID={poi_i_index} poi收集数据，"
                        f"并转发给ID={relay_car_index} carrier，"
                        f"受到ID={poi_j_index} poi的干扰，"
                        f"capacity={'%.3f' % capacity_i}, collect_time ={collect_time}, "
                        f"收集量={capacity_i * collect_time / poi_i_val}"
                    )
            # 根据capacity进行数据收集
            collected_data = min(capacity_i * collect_time, self.poi_value[poi_i_index])

            if self.COLLECT_MODE == 0:
                self.poi_value[poi_i_index] -= collected_data
                reward_list.append(collected_data * self.reward_co[type][poi_i_index])
                collect_list.append(collected_data)

            elif self.COLLECT_MODE == 1:
                reward_list.append(self.poi_value[poi_i_index] * self.reward_co[type][poi_i_index])
                collect_list.append(self.poi_value[poi_i_index])
                self.poi_value[poi_i_index] = 0

            elif self.COLLECT_MODE == 2:
                temp_poi_aoi_list.append(self.poi_value[poi_i_index]/self.USER_DATA_AMOUNT)
                self.poi_value[poi_i_index] -= collected_data
                reward_list.append(collected_data * self.reward_co[type][poi_i_index])
                collect_list.append(collected_data)
            data_rate_list.append(capacity_i)

            self.last_collect[self.step_count][type][uav_index][poi_i_index] = 1

        return (sum(reward_list), sum(collect_list),
                sum(data_rate_list), uav_pois, ugv_pois, collect_list, temp_poi_aoi_list)
        # sum(g2a_rate_list) / max(1, len(g2a_rate_list)),
        # sum(relay_rate_list) / max(1, len(relay_rate_list)))

    def _collect_data_from_poi(self, type, uav_index, collect_time=0):
        raise AssertionError
        reward_list = []
        if type == 'uav':
            position_list = []
            if collect_time >= 0:
                for poi_index, (poi_position, poi_value) in enumerate(zip(self.poi_position, self.poi_value)):
                    d = self._cal_distance(poi_position, self.agent_position[type][uav_index], type)
                    if d < self.COLLECT_RANGE[type] and poi_value > 0:
                        position_list.append((poi_index, d))
                position_list = sorted(position_list, key=lambda x: x[1])

                update_num = min(len(position_list), self.UPDATE_NUM[type])

                for i in range(update_num):
                    poi_index = position_list[i][0]
                    rate = self._get_data_rate(self.agent_position[type][uav_index], self.poi_position[poi_index])
                    if rate <= self.RATE_THRESHOLD[type]:
                        break
                    if self.COLLECT_MODE == 0:
                        collected_data = min(rate * collect_time / update_num, self.poi_value[poi_index])
                        self.poi_value[poi_index] -= collected_data
                        reward_list.append(collected_data)
                    elif self.COLLECT_MODE == 1:
                        reward_list.append(self.poi_value[poi_index])
                        self.poi_value[poi_index] = 0
                    elif self.COLLECT_MODE == 2:
                        collected_data = min(rate * collect_time, self.poi_value[poi_index])
                        self.poi_value[poi_index] -= collected_data
                        reward_list.append(collected_data)

        elif type == 'carrier':
            pass
        return sum(reward_list), len(reward_list)

    def _get_vector_by_theta(self, action):
        theta = action[0] * np.pi
        l = action[1] + 1
        dx = l * np.cos(theta)
        dy = l * np.sin(theta)
        return dx, dy

    def _get_vector_by_action(self, action, type='uav'):
        single = 3
        base = single / math.sqrt(2)

        action_table = [
            [0, 0],
            [-base, base],
            [0, single],
            [base, base],
            [-single, 0],
            [single, 0],
            [-base, -base],
            [0, -single],
            [base, -base],
            # 额外添加3个动作，向无人车靠近，还有向剩余poi最多的靠近
            [2 * single, 0],
            [0, 2 * single],
            [-2 * single, 0],
            [0, -2 * single]
        ]
        return action_table[action]

    def _get_vector_by_smart_action_(self, uav_index, action, type='uav'):
        single = 3
        base = single / math.sqrt(2)
        if action < 9:
            action_table = [
                [0, 0],
                [-base, base],
                [0, single],
                [base, base],
                [-single, 0],
                [single, 0],
                [-base, -base],
                [0, -single],
                [base, -base],

                # 额外添加3个动作，向无人车靠近，还有向剩余poi最多的靠近
                # [2*single,0],
                # [0,2*single],
                # [-2*single,0],
                # [0,-2*single]
            ]
            return action_table[action]
        elif action == 9:  # 向剩余数据量最大的poi 移动
            data_list = []
            for i in range(self.POI_NUM):
                dis = self._cal_distance(self.agent_position['uav'][uav_index], self.poi_position[i], type)
                data = self.poi_value[i] if dis < self.COLLECT_RANGE['uav'] else 0
                data_list.append(data)
            if len(data_list) > 0:
                move_target = np.argmax(data_list)
                dx = self.poi_position[move_target][0] - self.agent_position['uav'][uav_index][0]
                dy = self.poi_position[move_target][1] - self.agent_position['uav'][uav_index][1]
            else:
                dx = dy = 0
            return [dx, dy]
        else:  # 向无人车移动
            action = action - 10
            target_position = self.agent_position['carrier'][action]
            dx = target_position[0] - self.agent_position['uav'][uav_index][0]
            dy = target_position[1] - self.agent_position['uav'][uav_index][1]
            if math.sqrt(dx ** 2 + dy ** 2) > 4:
                dx = min(3, np.abs(dx)) * math.copysign(1, dx)
                dy = min(3, np.abs(dy)) * math.copysign(1, dy)
            return [dx, dy]

    def _is_uav_out_of_energy(self, uav_index, type):
        return self.uav_energy[type][uav_index] < self.EPSILON

    def _is_episode_done(self):
        if (self.step_count + 1) >= self.MAX_EPISODE_STEP:
            return True
        else:
            for type in self.UAV_TYPE:
                if type == 'carrier':
                    continue
                for i in range(self.NUM_UAV[type]):
                    if self._judge_obstacle(None, self.agent_position[type][i]):
                        print('cross the border!')
                        return True
            # return np.bool(np.all(self.dead_uav_list))
        return False

    def _judge_obstacle(self, cur_pos, next_pos):
        if self.ACTION_MODE == 2 or self.ACTION_MODE == 3: return False
        if cur_pos is not None:
            for o in self.OBSTACLE:
                vec = [[o[0], o[1]],
                       [o[2], o[3]],
                       [o[4], o[5]],
                       [o[6], o[7]]]
                if IsIntersec(cur_pos, next_pos, vec[0], vec[1]):
                    return True
                if IsIntersec(cur_pos, next_pos, vec[1], vec[2]):
                    return True
                if IsIntersec(cur_pos, next_pos, vec[2], vec[3]):
                    return True
                if IsIntersec(cur_pos, next_pos, vec[3], vec[0]):
                    return True

        if (0 <= next_pos[0] <= self.MAP_X) and (0 <= next_pos[1] <= self.MAP_Y):
            return False
        else:
            return True

    def _use_energy(self, type, uav_index, energy_consuming):
        self.uav_energy_consuming_list[type][uav_index].append(
            min(energy_consuming, self.uav_energy[type][uav_index]))
        self.uav_energy[type][uav_index] = max(
            self.uav_energy[type][uav_index] - energy_consuming, 0)

        if self._is_uav_out_of_energy(uav_index, type):
            if self.DEBUG_MODE:
                print("Energy should not run out!")
            self.dead_uav_list[type][uav_index] = True
            self.uav_state[type][uav_index].append(0)
        else:
            self.uav_state[type][uav_index].append(1)

    def _get_energy_coefficient(self):

        P0 = 58.06  # blade profile power, W
        P1 = 79.76  # derived power, W
        U_tips = 120  # tip speed of the rotor blade of the UAV,m/s
        v0 = 4.03  # the mean rotor induced velocity in the hovering state,m/s
        d0 = 0.2  # fuselage drag ratio
        rho = 1.225  # density of air,kg/m^3
        s0 = 0.05  # the rotor solidity
        A = 0.503  # the area of the rotor disk, m^2

        for type in self.UAV_TYPE:
            Vt = self.config("uav_speed")[type]  # velocity of the UAV,m/s ???
            if type == 'uav':
                self.Power_flying[type] = P0 * (1 + 3 * Vt ** 2 / U_tips ** 2) + \
                                          P1 * np.sqrt(
                    (np.sqrt(1 + Vt ** 4 / (4 * v0 ** 4)) - Vt ** 2 / (2 * v0 ** 2))) + \
                                          0.5 * d0 * rho * s0 * A * Vt ** 3

                self.Power_hovering[type] = P0 + P1
            elif type == 'carrier':
                self.Power_flying[type] = 17.49 + 7.4 * 15
                self.Power_hovering[type] = 17.49

    def _get_data_rate(self, uav_position, poi_position):
        eta = 2
        alpha = 4.88
        beta = 0.43
        distance = self._cal_distance(poi_position,uav_position, 'uav')
        theta = self._cal_theta(uav_position, poi_position)
        path_loss = (54.05 + 10 * eta * np.log10(distance) + (-19.9)
                     / (1 + alpha * np.exp(-beta * (theta - alpha))))
        w_tx = 20
        w_noise = -104
        w_s_t = w_tx - path_loss - w_noise
        w_w_s_t = np.power(10, (w_s_t - 30) / 10)
        bandwidth = 20e6
        data_rate = bandwidth * np.log2(1 + w_w_s_t)
        return data_rate / 1e6

    def get_adjacent_and_obs(self):
        """
        return_dict

        是一个字典，包含了以下键值对：

        1. **邻接矩阵**：
        - 键的格式为 `"key1-key2"`，表示从   key1   类型的智能体到 key2 类型的智能体的邻接关系。
        - 值是一个二维 numpy 数组，表示邻接矩阵。

        2. **观察信息**：
        - 键为 `"carrier"` 和 `"uav"`，表示 carrier 和 UAV 的观察信息。
        - 值是一个二维 numpy 数组，表示智能体的观察信息。

        ### 邻接矩阵的形状和含义

        邻接矩阵的键值对如下：

        - `"uav-carrier"`：表示 UAV 和 carrier 之间的邻接关系。
        - 形状：`(num_of_uav, num_of_carrier)`
        - 含义：每个 UAV 与每个 carrier 之间的邻接关系。

        - `"uav-poi"`：表示 UAV 和 POI 之间的邻接关系。
        - 形状：`(num_of_uav, self.POI_NUM)`
        - 含义：每个 UAV 与每个 POI 之间的邻接关系。

        - `"uav-epoi"`：表示 UAV 和 ePOI（通过 carrier 间接连接的 POI）之间的邻接关系。
        - 形状：`(num_of_uav, self.POI_NUM)`
        - 含义：每个 UAV 与每个 ePOI 之间的邻接关系。

        - `"uav-road"`：表示 UAV 和 road 之间的邻接关系。
        - 形状：`(num_of_uav, len(self.nx_g))`
        - 含义：每个 UAV 与每个 road 节点之间的邻接关系。

        - `"carrier-uav"`：表示 carrier 和 UAV 之间的邻接关系。
        - 形状：`(num_of_carrier, num_of_uav)`
        - 含义：每个 carrier 与每个 UAV 之间的邻接关系。

        - `"carrier-poi"`：表示 carrier 和 POI 之间的邻接关系。
        - 形状：`(num_of_carrier, self.POI_NUM)`
        - 含义：每个 carrier 与每个 POI 之间的邻接关系。

        - `"carrier-road"`：表示 carrier 和 road 之间的邻接关系。
        - 形状：`(num_of_carrier, len(self.nx_g))`
        - 含义：每个 carrier 与每个 road 节点之间的邻接关系。

        - `"carrier-epoi"`：表示 carrier 和 ePOI 之间的邻接关系。
        - 形状：`(num_of_carrier, self.POI_NUM)`
        - 含义：每个 carrier 与每个 ePOI 之间的邻接关系。

        - `"poi-uav"`：表示 POI 和 UAV 之间的邻接关系。
        - 形状：`(self.POI_NUM, num_of_uav)`
        - 含义：每个 POI 与每个 UAV 之间的邻接关系。

        - `"poi-carrier"`：表示 POI 和 carrier 之间的邻接关系。
        - 形状：`(self.POI_NUM, num_of_carrier)`
        - 含义：每个 POI 与每个 carrier 之间的邻接关系。

        - `"road-carrier"`：表示 road 和 carrier 之间的邻接关系。
        - 形状：`(len(self.nx_g), num_of_carrier)`
        - 含义：每个 road 节点与每个 carrier 之间的邻接关系。

        ### 观察信息的形状和含义

        观察信息的键值对如下：

        - `"carrier"`：表示 carrier 的观察信息。
        - 形状：`(num_of_carrier, observation_dim)`
        - 含义：每个 carrier 的观察信息，其中 `observation_dim` 是观察信息的维度。

        - `"uav"`：表示 UAV 的观察信息。
        - 形状：`(num_of_uav, observation_dim)`
        - 含义：每个 UAV 的观察信息，其中 `observation_dim` 是观察信息的维度。

        uav-carrier: (num_of_uav, num_of_carrier)
        uav-poi: (num_of_uav, self.POI_NUM)
        uav-epoi: (num_of_uav, self.POI_NUM)
        uav-road: (num_of_uav, len(self.nx_g))
        carrier-uav: (num_of_carrier, num_of_uav)
        carrier-poi: (num_of_carrier, self.POI_NUM)
        carrier-road: (num_of_carrier, len(self.nx_g))
        carrier-epoi: (num_of_carrier, self.POI_NUM)
        poi-uav: (self.POI_NUM, num_of_uav)
        poi-carrier: (self.POI_NUM, num_of_carrier)
        road-carrier: (len(self.nx_g), num_of_carrier)
        carrier: (num_of_carrier, observation_dim)
        uav: (num_of_uav, observation_dim)
        """
        adj_dict = {
            'uav': {key: None for key in ['carrier', 'poi', 'road', 'epoi']},  # n x n, n x poi
            'carrier': {key: None for key in ['uav', 'poi', 'road', 'epoi']},  # n x n, n x poi, n x node
            'poi': {key: None for key in ['uav', 'carrier']},  # poi x n, poi x n
            'road': {key: None for key in ['carrier']}  # node x n
        }
        if not self.USE_HGCN:
            return_dict = {}
            for key1, s_dict in adj_dict.items():
                for key2, adj in s_dict.items():
                    return_dict[f"{key1}-{key2}"] = adj
            return return_dict

        uav_field = self.agent_field['uav']
        carrier_field = self.agent_field['carrier']
        # uav-carrier
        num_of_carrier = self.NUM_UAV['carrier']
        num_of_uav = self.NUM_UAV['uav']
        uav_carrier = np.zeros((num_of_uav, num_of_carrier))
        # relay_dict = self._relay_association()
        # for k, v in relay_dict.items():
        #    uav_carrier[k][v] = 1

        for uav_id in range(num_of_uav):
            for carrier_id in range(num_of_carrier):
                uav_carrier[uav_id][carrier_id] = self._cal_distance(self.agent_position['uav'][uav_id],
                                                                     self.agent_position['carrier'][carrier_id],
                                                                     type='uav') / (self.SCALE * self.MAP_X * 1.414) \
                                                  < uav_field

        # for uav_id in range(self.NUM_UAV['uav']):
        #     for carrier_id in range(self.NUM_UAV['carrier']):
        #         uav_carrier[uav_id][carrier_id] = self._get_data_rate(self.agent_position['uav'][uav_id],
        #                                                              self.agent_position['carrier'][carrier_id])

        record_times = 10  # 记录过去5步收集的poi index
        start_t = max(0, self.step_count - record_times)
        uav_collect = np.zeros((num_of_uav, self.POI_NUM))
        carrier_collect = np.zeros((num_of_carrier, self.POI_NUM))
        for t in range(start_t, self.step_count):
            uav_collect = np.logical_or(uav_collect, self.last_collect[t]['uav'])
            carrier_collect = np.logical_or(carrier_collect, self.last_collect[t]['carrier'])
        uav_collect = uav_collect.astype(np.float32)
        carrier_collect = carrier_collect.astype(np.float32)

        poi_visible = np.zeros(((self.n_agents, self.POI_NUM)))
        # uav-poi
        uav_poi = np.zeros(((num_of_uav, self.POI_NUM)))
        for uav_id in range(num_of_uav):
            uav_poi[uav_id] = uav_collect[uav_id]
            if self.edge_type == 'dis':#这个
                poi_v = self._cal_distance(self.poi_position,np.array(self.agent_position['uav'][uav_id]), 
                                           type='uav') < uav_field
                poi_visible[num_of_carrier + uav_id, :] = poi_v
            elif self.edge_type == 'data_rate':
                poi_v = self._get_data_rate(np.array(self.agent_position['uav'][uav_id]), self.poi_position)
                poi_visible[num_of_carrier + uav_id, :] = poi_v / np.linalg.norm(poi_v)
            else:
                raise NotImplementedError
       

        # carrier-poi
        carrier_poi = np.zeros(((num_of_carrier, self.POI_NUM)))
        for carrier_id in range(num_of_carrier):
            carrier_poi[carrier_id] = carrier_collect[carrier_id]
            if self.edge_type == 'dis':
                carrier_v = self._cal_distance(self.poi_position,np.array(self.agent_position['carrier'][carrier_id]), 
                                               type='carrier') < carrier_field
                poi_visible[carrier_id, :] = carrier_v
            elif self.edge_type == 'data_rate':
                carrier_v = self._get_data_rate(np.array(self.agent_position['carrier'][carrier_id]), self.poi_position)
                poi_visible[carrier_id, :] = carrier_v / np.linalg.norm(carrier_v)
            else:
                raise NotImplementedError
         

        # carrier-roadmap
        carrier_road = np.zeros((num_of_carrier, len(self.nx_g)))
        num_action = self.action_space['carrier'].n
        for carrier_id in range(num_of_carrier):
            for i in range(num_action):
                last_node = self.OSMNX_TO_NX[self.last_dst_node[carrier_id][i]]
                carrier_road[carrier_id][last_node] = 1


        adj_dict['uav']['carrier'] = row_normalize(uav_carrier) #表示第i个uav与第j个carrier之间是否可见，1表示可见，0表示不可见，做行归一化
        adj_dict['uav']['poi'] = row_normalize(uav_poi)
        adj_dict['uav']['epoi'] = row_normalize(np.dot(uav_carrier, carrier_poi)) #relay（noma通信）的模式下，uav能看到car，car能看到poi，则uav能看到poi
        adj_dict['uav']['road'] = row_normalize(np.dot(uav_carrier, carrier_road))# 同理，uav能看到car，car能看到road，则uav能看到road

        adj_dict['carrier']['uav'] = row_normalize(uav_carrier.T)
        adj_dict['carrier']['poi'] = row_normalize(carrier_poi)
        adj_dict['carrier']['road'] = row_normalize(carrier_road)
        adj_dict['carrier']['epoi'] = row_normalize(np.dot(uav_carrier.T, uav_poi))#反过来，car能看到uav，uav能看到poi，则car能看到poi

        adj_dict['poi']['uav'] = row_normalize(uav_poi.T)
        adj_dict['poi']['carrier'] = row_normalize(carrier_poi.T)

        adj_dict['road']['carrier'] = row_normalize(carrier_road.T)

        return_dict = {}
        for key1, s_dict in adj_dict.items():
            for key2, adj in s_dict.items():
                return_dict[f"{key1}-{key2}"] = adj

        # -----------------------------------------------------------------
        # obs部分
        one_hot = np.eye(self.n_agents)
        agent_pos = []
        for t in self.NUM_UAV:
            for i in range(self.NUM_UAV[t]):
                agent_pos.append(self.agent_position[t][i][0] / self.MAP_X)
                agent_pos.append(self.agent_position[t][i][1] / self.MAP_Y)
        agent_pos = np.array([agent_pos for _ in range(self.n_agents)])

        poi_value = np.array([self.poi_value for _ in range(self.n_agents)])
        poi_position = np.array([self.poi_position for _ in range(self.n_agents)])
        
        dividend = self.get_poi_dividend()
        visible_poi_states = np.concatenate([poi_position[:, :, 0] * poi_visible / self.MAP_X,
                                             poi_position[:, :, 1] * poi_visible / self.MAP_Y,
                                             poi_value * poi_visible / dividend], axis=1)

        obs = np.concatenate([one_hot, agent_pos, visible_poi_states], axis=1)
        #shape：（n_agents, n_agents+2*n_agent+3*POI_NUM）,每一行代表一个agent的观察信息，前n_agents列是one-hot编码，接下来的2*n_agent列是所有agent的坐标，接下来的3*POI_NUM列是POI的坐标和数据量
        # -----------------------------------------------------------------
        return_dict.update({
            'carrier': obs[:num_of_carrier, :],
            'uav': obs[num_of_carrier:, :],
        })
        return return_dict

    def get_obs(self, aoi_now=None, aoi_next=None):
        """
        state: (n_agents, n_agents+2*n_agent+3*POI_NUM),每一行代表一个agent的观察信息，前n_agents列是one-hot编码(state全是1），接下来的2*n_agent列是所有agent的坐标，接下来的3*POI_NUM列是POI的坐标和数据量
        uav(state): (n_agents, n_agents+2+3*n_agent+5*POI_NUM),每一行代表一个agent的观察信息，前n_agents列是one-hot编码，接下来2是agentType[0,1]/[1,0],接下来的3*n_agent列是该agent的3D坐标，接下来的5*POI_NUM列是POI的2d坐标和agent-poi距离和ralyCar-poi距离和数据量
        carrier(state): (n_agents, n_agents+2+3*n_agent+5*POI_NUM),每一行代表一个agent的观察信息，前n_agents列是one-hot编码，接下来2是agentType[0,1]/[1,0],接下来的3*n_agent列是该agent的3D坐标，接下来的5*POI_NUM列是POI的2d坐标和agent-poi距离和全0（car没有relay）和数据量
        Nodes: (total_node_num,3),每个道路节点的x,y,data属性
        Edges: (2, total_edge_num),每一列代表一条边的两个端点,做了转置
        mask_uav:(num_uav,ACTION_ROOT),default(2,15),值取0/1，每一行表示uav能否取该动作的mask
        mask_carrier:(num_carrier,ACTIOn_ROOT),default(2,15),值取0/1，每一行表示uav能否取该动作的mask
        mask_poi_uav:（num_uav,1+num_poi),每一行表示uav是否在观测范围内能收集到poi信息（uav观测范围默认300）
        mask_poi_carrier:(num_carrier,1+num_poi),每一行表示carrier能否在观测范围内收集到poi信息（carrier观测范围默认250）
        key-key邻接矩阵&agent_obs是否有，取决于use_hgcn
        """

        if self.USE_HGCN:
            agents_obs = {key: np.vstack([self.get_obs_agent(i, visit_num=self.POI_IN_OBS_NUM, type=key) for i in range(self.NUM_UAV[key])]) for key in
                          self.UAV_TYPE}
        else:
            agents_obs = {key: np.vstack([self.get_obs_agent(i, visit_num=self.POI_IN_OBS_NUM, type=key) for i in range(self.NUM_UAV[key])]) for key in
                          self.UAV_TYPE}
        if self.POI_DECISION_MODE:
            poi_obs = {'poi_'+key: np.vstack([agents_obs[key][i] for i in range(self.NUM_UAV[key])]) for key in
                          self.UAV_TYPE}
            agents_obs = {**agents_obs,**poi_obs}
                        
        move_mask = self.get_avail_actions()
        poi_mask = self.get_poi_mask()
        move_mask = {'mask_' + key: move_mask[key] for key in self.UAV_TYPE}
        poi_mask = {'mask_poi_' + key: poi_mask[key] for key in self.UAV_TYPE}
        action_mask = {**move_mask,**poi_mask}
        
        # node_features = np.array([[data['py_x'] / self.MAP_X, data['py_y'] / self.MAP_Y,
        #                            data['data'] / (self.USER_DATA_AMOUNT * self.MAX_EPISODE_STEP)] for _, data in
        #                           self.nx_g.nodes(data=True)])
        # edge_features = np.array(self.nx_g.edges())
        # edge_features = self.edge_features
        obs_dict = {
            'State': self.get_state(), #这里面包含了所有agent的位置信息，以及所有poi的数据量信息，没有visual_mask
            # 'State':[],
            **{
                'Nodes': self.get_node_agents(-1, type=-1, global_view=True), #shape&meaning:(total_node_num,3),每个道路节点的x,y,data属性
                'Edges': self.edge_features,
            },
            # **{'Nodes_' + key: np.vstack([[self.get_node_agents(0, type=key)] for _ in range(self.NUM_UAV[key])]) for
            #    key in self.UAV_TYPE},
            # **{'Edges_' + key: np.vstack([[self.edge_features] for i in range(self.NUM_UAV[key])]) for key in
            #    self.UAV_TYPE},
            **action_mask,#包含了可行动作的mask，以及poi的mask
            **agents_obs,# 包含了所有agent的观察信息,对每个agent,都mask观测范围之外的poi，但所有agent的位置信息都是全局的
            **self.get_adjacent_and_obs() #邻接矩阵和观察信息
        }

        return obs_dict

    def get_node_agents(self, agent_id, type, global_view=True):
        total_nodes_num = len(self.nx_g)
        collected_datas = np.zeros((total_nodes_num, 3))
        for node in self.node_to_poi.values(): #values是，每个poi最近的node
            data = self.nx_g.nodes(data=True)[node]
            collected_datas[node, :] = data['py_x'], data['py_y'], data['data']
        collected_datas[:, 0] /= self.SCALE
        collected_datas[:, 1] /= self.SCALE

        if not global_view:
            mask = self._cal_distance([collected_datas[:, 0], collected_datas[:, 1]],
                                      self.agent_position[type][agent_id], 'carrier') \
                   <= self.agent_field[type]
        for i, diviend in zip([0, 1], [self.MAP_X, self.MAP_Y]):
            if not global_view:
                collected_datas[:, i] *= mask
            collected_datas[:, i] /= diviend
        # 100 times, 0.025s for i in range(self.POI_NUM):     temp = datas[self.node_to_poi[i], 2] / total_data[i]
        # 100 times, 0.023s for node, data in zip(self.node_to_poi.values(), total_data):     temp = datas[node, 2] / data
        max_data = self.get_poi_dividend()
        for node in self.node_to_poi.values():
            try:
                collected_datas[node, 2] = collected_datas[node, 2] / max_data if collected_datas[node, 2] > 0 else 0
            except FloatingPointError:
                collected_datas[node, 2] = 0
        return collected_datas #shape: (total_nodes_num, 3)，每行代表一个节点的坐标和数据量

    def get_obs_agent(self, agent_id, global_view=False, visit_num=None, type=None):

        if self.USE_HGCN:
            return np.zeros(self.obs_space['State'].shape, dtype=np.float32)

        if visit_num is None or visit_num == -1:
            visit_num = self.POI_NUM

        if global_view:
            distance_limit = 1e10
        else:
            distance_limit = self.agent_field[type]
        target_dis = self.agent_position[type][agent_id]

        
        active_agent_pos = np.array(self.agent_position[type][agent_id])
        relay_agent_pos = np.array(self.agent_position['carrier'][agent_id])
        active_poi_dis = self._cal_distance(self.poi_position,active_agent_pos,type=type)
        relay_poi_dis = self._cal_distance(self.poi_position,relay_agent_pos,type=type) if type =='uav' else 0
        
        agent_pos = []
        for t in self.UAV_TYPE:
            for i in range(self.NUM_UAV[t]):
                agent_pos.append(self.agent_position[t][i][0] / self.MAP_X)
                agent_pos.append(self.agent_position[t][i][1] / self.MAP_Y)
                
                if t == 'uav':
                    agent_pos.append(self._cal_distance(relay_agent_pos,active_agent_pos,type='carrier')[0]/(self.MAP_X))
                else:
                    agent_pos.append(0)

        distances = self._cal_distance(self.poi_position, np.array(target_dis), type)
        order = distances.argsort()
        is_visible = distances < distance_limit
    
        
        dividend = self.get_poi_dividend()
        # rel_vec = self.poi_position - np.tile(np.asarray(
        #     self.agent_position[type][agent_id]), (len(self.poi_position), 1))
        visible_poi_states = np.stack([self.poi_position[:, 0] * is_visible / self.MAP_X,
                                       self.poi_position[:, 1] * is_visible / self.MAP_Y,
                                       active_poi_dis*is_visible / self.distance_normalization,
                                       relay_poi_dis*is_visible / self.distance_normalization,
                                       self.poi_value * is_visible / dividend], axis=1)

        id = agent_id if type == 'carrier' else self.NUM_UAV['carrier'] + agent_id
        type_obs = [1,0] if type == 'carrier' else [0,1]
        one_hot = np.eye(self.n_agents)[id]
        #agent_pos = np.zeros_like(agent_pos)
        visible_poi_states = visible_poi_states[order[:visit_num]]
        return np.concatenate([one_hot,type_obs, np.array(agent_pos), visible_poi_states.flatten()])

    def get_poi_dividend(self):
        if self.COLLECT_MODE:
            dividend = self.USER_DATA_AMOUNT * self.MAX_EPISODE_STEP
        else:
            dividend = self.POI_INIT_DATA
        return dividend

    def get_obs_size(self, visit_num=None):
        obs_size = {}
        num = self.n_agents* self.agent_property_num + 2 # 额外的type类型,[0,1]或[1,0]

        for type in self.UAV_TYPE:
            size = num
            if visit_num is None or visit_num == -1:
                size += self.POI_NUM * self.poi_property_num
            else:
                size += visit_num * self.poi_property_num
            obs_size[type] = size
        return obs_size

    def get_state(self):

        if not self.USE_HGCN:
            return np.zeros(self.obs_space['State'].shape, dtype=np.float32)

        obs = []
        for t in self.NUM_UAV:
            for i in range(self.NUM_UAV[t]):
                obs.append(self.agent_position[t][i][0] / self.MAP_X)
                obs.append(self.agent_position[t][i][1] / self.MAP_Y)

        dividend = self.get_poi_dividend()
        poi_value = np.array(self.poi_value)
        visible_poi_states = np.stack([self.poi_position[:, 0] / self.MAP_X,
                                       self.poi_position[:, 1] / self.MAP_Y,
                                       poi_value / dividend], axis=1)

        return np.concatenate([np.ones((self.n_agents,)), obs, visible_poi_states.flatten()])

    def get_concat_obs(self, agent_obs):
        state_all = {}
        for key in self.UAV_TYPE:
            state = np.zeros_like(agent_obs[key][0])
            for i in range(self.NUM_UAV[key]):
                mask = agent_obs[key][i] != 0
                np.place(state, mask, agent_obs[key][i][mask])
            state_all[key] = state
        return state_all

    def get_state_size(self):
        size = self.POI_NUM * 3 + self.n_agents * 3
        return size

    def get_avail_actions(self):
        avail_actions_all = {}
        for type in self.UAV_TYPE:
            avail_actions = []
            for agent_id in range(self.NUM_UAV[type]):
                avail_agent = self.get_avail_agent_actions(agent_id, type)
                avail_actions.append(avail_agent)
            avail_actions_all[type] = np.vstack(avail_actions)
        return avail_actions_all

    def get_poi_mask(self):
        """
        返回一个字典，包含了以下键值对：
        uav: (num_of_uav, 1+self.POI_NUM)
        carrier: (num_of_carrier, 1+self.POI_NUM)
        含义是每个agent是否可以采集每个poi，1表示可以，0表示不可以
        """
        avail_actions_all = {}
        for type in self.UAV_TYPE:
            avail_actions = []
            for agent_id in range(self.NUM_UAV[type]):
                pos = self.agent_position[type][agent_id]
                distances = self._cal_distance( self.poi_position,np.array(pos), type)     
                if self.NEAR_SELECTION_MODE:   
                    order = distances.argsort()
                    select_dis = distances[order[:self.POI_SELECTION_NUM]]
                    distance_mask = (select_dis <= self.COLLECT_RANGE[type]).astype(np.int32).tolist()
                else:
                    distance_mask = (distances <= self.COLLECT_RANGE[type]).astype(np.int32).tolist()
                #distance_mask = [1 for i in range(self.POI_SELECTION_NUM)]                  
                avail_actions.append([1]+distance_mask)
            avail_actions_all[type] = np.vstack(avail_actions)
        return avail_actions_all
    
    def get_avail_agent_actions(self, agent_id, type):
        relay_dict = self._relay_association()
        if type == 'uav':
            avail_actions = []
            count = 0
            temp_x, temp_y = self.agent_position[type][agent_id]
            distances = np.zeros(self.UAV_ACTION_ROOT)
            for i in range(self.UAV_ACTION_ROOT):
                # dx, dy = self._get_vector_by_smart_action_(agent_id,i,type)
                dx, dy = self._get_vector_by_action(i, type)
                new_pos = (dx + temp_x, dy + temp_y)
                no_obstacle = not self._judge_obstacle((temp_x, temp_y), new_pos)
                dis = self._cal_distance(new_pos,
                                         self.agent_position['carrier'][relay_dict[agent_id]],
                                         type='carrier')
                # print(dis)
                if self.config("limited_collection"):
                    not_too_far =  dis < self.LIMIT_RANGE[type] * self.mask_range
                else:
                    not_too_far = dis < self.agent_field[type] *8 
     
                distances[i] = dis
                if no_obstacle and not_too_far:
                    count += 1
                    avail_actions.append(1)
                else:
                    avail_actions.append(0)
            if count == 0:
                selected_action = np.argmin(distances)
                # print(selected_action)
                avail_actions[selected_action] = 1
            # print(f"Step {self.step_count}: {count}")
            if len(avail_actions) != self.ACTION_ROOT:
                avail_actions.extend([0]*(self.ACTION_ROOT-len(avail_actions)))
            return np.array(avail_actions)

        elif type == 'carrier':
            num_action = self.action_space['carrier'].n + 20 
            avail_actions = np.zeros((self.action_space['carrier'].n,))

            def sort_near_set_by_angle(src_node, near_set):
                thetas = []
                src_pos = self.rm.lonlat2pygamexy(self.nx_g.nodes[src_node]['x'],
                                                  self.nx_g.nodes[src_node]['y'])
                for item in near_set:
                    dst_node = self.OSMNX_TO_NX[int(item[0])]
                    if dst_node in self.ROAD_MAP[self.map]:
                        thetas.append(99999)
                        continue
                    dst_pos = self.rm.lonlat2pygamexy(self.nx_g.nodes[dst_node]['x'], self.nx_g.nodes[dst_node]['y'])
                    theta = compute_theta(dst_pos[0] - src_pos[0], dst_pos[1] - src_pos[1], 0)
                    thetas.append(theta)
                near_set = sorted(near_set, key=lambda x: thetas[near_set.index(x)])
                return near_set

            # step2. 对每辆车，根据pair_dis_dict得到sorted的10个点，如果可达则mask=1
            id = agent_id
            # 路网栅格化之后 这里可能会返回一个不在栅格化后集合中的点 因此这里直接记录一下 车现在处于哪个index
            src_node = self.carrier_node[id]
            # 验证车一定在路网的点上
            neighbor_nodes = []
            # for neighbor in self.nx_g.neighbors(src_node):
            #     distance = nx.shortest_path_length(self.nx_g, src_node, neighbor, weight='length')
            #     neighbor_nodes.append((neighbor,distance))
            distances = self.PAIR_DIS_DICT[self.map][str(src_node)]
            distances = {key:value for key,value in distances.items() if int(key) in self.valid_nodes}
            pairs = list(zip(distances.keys(), distances.values()))
            near_set = sorted(pairs, key=lambda x: x[1])[:max(0, num_action - len(neighbor_nodes))]
            near_set = neighbor_nodes + near_set
            near_set = sort_near_set_by_angle(self.OSMNX_TO_NX[src_node], near_set)
            
            near_set = [item for item in near_set if item[1] < 600]
            
            step = len(near_set) / self.action_space['carrier'].n if len(near_set) > self.action_space['carrier'].n else 1
            # 等间距选取20个元素
            if len(near_set) > self.action_space['carrier'].n:
                near_set = [near_set[int(i * step)] for i in range(self.action_space['carrier'].n)]
            
            valid_nodes = set([data['old_label'] for _, data in self.nx_g.nodes(data=True)])
            action_index = 0 
            for act in range(num_action):
                if act == len(near_set): break
                dst_node, length = near_set[act]
                if dst_node in self.ignore_node or int(dst_node) not in valid_nodes:
                    continue
                if length != np.inf:
                    converted = self.OSMNX_TO_NX[int(dst_node)]
                    new_pos = (self.nx_g.nodes[converted]['x'], self.nx_g.nodes[converted]['y'])
                    avail_actions[action_index] = 1
                    self.last_dst_node[id][action_index] = dst_node
                    self.last_length[id][action_index] = length
                    self.last_dst_lonlat[id][action_index][0], self.last_dst_lonlat[id][action_index][1] = (
                        new_pos[0], new_pos[1]
                    )
                    action_index += 1
                if action_index == self.action_space['carrier'].n:
                    break
            return avail_actions

    def get_total_actions(self):
        return self.n_actions

    def get_num_of_agents(self):
        return self.n_agents

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(visit_num=self.POI_IN_OBS_NUM),# {uav:(车+飞机)*4+2+poi数量*5}
                                                                                   #carrier：(车+飞机)*4+2+poi数量*5，一样的         
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit} #120最大step

        return env_info

    def get_obs_from_outside(self):
        return self.get_obs()

    def check_arrival(self):
        """
        每次step，都会调用check_arrival,每个poi都不断产生新的数据， 新数据的大小是每20s30帧1080p摄像机拍摄的数据量
        假设距离每个poi最近的道路node，组成了poi的node最近邻集合
        在每次step中，poi的node最近邻集合，每个node的data都会更新成当前距离最近的poi的数据剩余量
        """
        for i in range(self.POI_NUM):
            self.nx_g.nodes[self.node_to_poi[i]]['data'] = 0

        for i in range(self.POI_NUM):
            if self.COLLECT_MODE == 1:
                self.poi_value[i] += 1
            elif self.COLLECT_MODE == 2:
                self.poi_value[i] += self.USER_DATA_AMOUNT  #不断产生新的数据，user_data_amount是1080p*20s*30fps，
            self.nx_g.nodes[self.node_to_poi[i]]['data'] += self.poi_value[i]

    def get_node_poi_map(self, map, nx_g):
        result_dict = {}
        node_positions = []
        remove_map = [self.OSMNX_TO_NX[x] for x in self.ROAD_MAP[map] if x in self.valid_nodes]
        for node, data in nx_g.nodes(data=True):
            if node not in remove_map:
                node_positions.append([data['py_x'] / self.MAP_X, data['py_y'] / self.MAP_Y])
            else:
                node_positions.append([99999999, 999999])

        node_positions = np.array(node_positions)
        poi_nearest_distance = [0 for _ in range(self.POI_NUM)]
        for poi_index in range(self.POI_NUM):
            poi_position = self._poi_position[poi_index]
            distances = np.linalg.norm(poi_position - node_positions, axis=1)
            nearest_index = np.argmin(distances)
            result_dict[poi_index] = int(nearest_index)
            poi_nearest_distance[poi_index] = float(distances[nearest_index] * self.SCALE)

        return result_dict, poi_nearest_distance
      # shape： result_dict: {poi_index:node_index} poi_nearest_distance: [distance1,distance2,...]
      #含义：result_dict记录了每个poi对应的最近的node的index，poi_nearest_distance记录了每个poi到最近node的距离
    
        
    def poi_reset(self):
        self.poi_step_status = False
        self.poi_step_count = 0
        self.poi_step_all = sum([self.NUM_UAV[key] for key in self.UAV_TYPE]) * self.CHANNEL_NUM
        #decided_sorted_access记录了每个agent的每个channel的决策，-1表示不采集，其他数字表示采集的poi的index,
        #decided_relay_dict[agent_type][agent_index]表示在当前collect时间步，特定agent的poi收集顺序，这是一个list，总长度是channel
        self.decided_sorted_access = {type:[[] for _ in range(self.NUM_UAV[type])] for type in self.UAV_TYPE}
        self.decided_relay_dict = self._relay_association()
        self.fake_poi_value = copy.deepcopy(self.poi_value)
        self.decided_collect_time = {type:[max(0, self.TIME_SLOT - self.distance[type][uav_index] / self.UAV_SPEED[type]) for uav_index in range(self.NUM_UAV[type])] for type in self.UAV_TYPE}
        #collect时间
        self.poi_allocated_list = np.zeros([self.POI_NUM])
        
        #邻接矩阵中，car-uav和uav-car之间的邻接矩阵是对称的，1表示uav和car是配对的relay关系，0表示否
        #car-poi和uav-poi之间的邻接矩阵是不对称的，0表示agent采样不到poi，0-1之间的数表示agent到poi的归一化距离
        #poi-car和poi-uav是上面的转置
        #在后续选点处理中，每当一个agent选定一个poi，就会将agent（uav/car）-poi和poi-agent（uav/car）的邻接矩阵中对应的值置为0，
        self.init_poi_adj_dict = {
            'carrier':{'uav': np.diag([1 for _ in range(self.NUM_UAV['carrier'])]),
                        'poi': np.zeros(((self.NUM_UAV['carrier'], self.POI_NUM)))}, 
            'uav': {'carrier': np.diag([1 for _ in range(self.NUM_UAV['uav'])]),
                    'poi': np.zeros(((self.NUM_UAV['uav'], self.POI_NUM)))}, 
       
            'poi': {'carrier': np.zeros(((self.POI_NUM,self.NUM_UAV['carrier']))),
                    'uav': np.zeros(((self.POI_NUM,self.NUM_UAV['uav'])))},
        }
        
                           
        if self.NEAR_SELECTION_MODE:
            self.init_poi_adj_dict = {
                'carrier':{'uav': np.diag([1 for _ in range(self.NUM_UAV['carrier'])]),
                            'poi': np.zeros(((self.NUM_UAV['carrier'], self.POI_SELECTION_NUM-1)))}, 
                'uav': {'carrier': np.diag([1 for _ in range(self.NUM_UAV['uav'])]),
                        'poi': np.zeros(((self.NUM_UAV['uav'], self.POI_SELECTION_NUM-1)))}, 
        
                'poi': {'carrier': np.zeros(((self.POI_SELECTION_NUM-1,self.NUM_UAV['carrier']))),
                        'uav': np.zeros(((self.POI_SELECTION_NUM-1,self.NUM_UAV['uav'])))},
            }
            
            #action_to_poi_index[type][agent_id]是一个dict，key是action，value是对应的poi的index，表示一个agent的一个channel的0-14个action对应的poi的index
            self.action_to_poi_index = {type:[{} for _ in range(self.NUM_UAV[type])] for type in self.UAV_TYPE}
            self.near_action_mask = {type:[[] for _ in range(self.NUM_UAV[type])] for type in self.UAV_TYPE}#near_action_mask[type][agent_id]是一个list，长度是14，表示一个agent的一个channel的前14个poi是否在观测范围内
            for type in self.UAV_TYPE:
                for agent_id in range(self.NUM_UAV[type]):
                    pos = self.agent_position[type][agent_id]
                    distances = self._cal_distance( self.poi_position,np.array(pos), type)  
                    order = distances.argsort()
                    for i in range(self.POI_SELECTION_NUM-1):#i从0到13？
                        self.action_to_poi_index[type][agent_id][i] = order[i]
                    select_dis = distances[order[:self.POI_SELECTION_NUM-1]] #前14个poi的距离
                    mask = (select_dis <= self.COLLECT_RANGE[type]).astype(np.int32).tolist()#前14个poi是否在观测范围内，1表示在，0表示不在，mask的长度是14
                    self.near_action_mask[type][agent_id].extend(mask)
                    
                    normal_dis = select_dis / self.COLLECT_RANGE[type]
                    normal_dis[normal_dis>=1] = 0
                    self.init_poi_adj_dict['poi'][type][:,agent_id] = normal_dis
                    self.init_poi_adj_dict[type]['poi'][agent_id,:] = normal_dis
        
        self.reset_poi_adj_dict()   

    
    def reset_poi_adj_dict(self):
        self.poi_adj_dict = self.init_poi_adj_dict
    
        
    def poi_step(self, policy_action):
        assert self.poi_step_status == True
        assert self.decided_collect_time is not None
        agent_type,agent_index,channel = self.get_type_index_channel_from_step(self.poi_step_count)
        
        if policy_action == 0:
            self.decided_sorted_access[agent_type][agent_index].append(-1)
            self.rl_data_rate[agent_type][agent_index].append(0)
            reward = 0
        else:
            if self.NEAR_SELECTION_MODE:
                action = self.action_to_poi_index[agent_type][agent_index][policy_action-1]
                self.poi_adj_dict[agent_type]['poi'][agent_index][policy_action-1] = 0
                self.poi_adj_dict['poi'][agent_type][policy_action-1][agent_index] = 0

            else:
                action = policy_action -1 
                
            self.poi_allocated_list[action] = 1
            self.decided_sorted_access[agent_type][agent_index].append(action)
            
            poi_i_index = action
            poi_i_pos = self.poi_position[poi_i_index]
            if agent_type == 'carrier':
                car_pos = self.agent_position[agent_type][agent_index]
                sinr_i, capacity_i = compute_capacity_G2G(self.noma_config,
                                                            self._cal_distance(poi_i_pos, car_pos, type)
                                                            )
                capacity_i /= 1e6

            else:
                assert agent_type == 'uav'
                relay_car_index = self.decided_relay_dict[agent_index] # 确定当前uav转发给哪个car
                uav_pos, relay_car_pos = self.agent_position[agent_type][agent_index], self.agent_position['carrier'][relay_car_index]
                # uav从poi_i收集数据，但poi_j会造成干扰
                poi_j_index = self.decided_sorted_access['carrier'][relay_car_index][channel]

                if self._cal_distance(uav_pos, relay_car_pos, agent_type) >= self.COLLECT_RANGE[agent_type] and self.config(
                        "limited_collection"):
                    R_G2A = R_RE = 1
                    if self.DEBUG_MODE: print(
                        f"fake test 车机距离：{self._cal_distance(uav_pos, relay_car_pos, agent_type)}, 无人机与poi i的距离{self._cal_distance(poi_i_pos, uav_pos, agent_type)}")
                else:
                    if poi_j_index != -1:
                        poi_j_pos = self.poi_position[poi_j_index]

                        sinr_G2A, R_G2A = compute_capacity_G2A(self.noma_config,
                                                                self._cal_distance(poi_i_pos, uav_pos, agent_type),
                                                                self._cal_distance(poi_j_pos, uav_pos, agent_type),
                                                                )
                        sinr_RE, R_RE = compute_capacity_RE(self.noma_config,
                                                            self._cal_distance(uav_pos, relay_car_pos, agent_type),
                                                            self._cal_distance(poi_i_pos, relay_car_pos, 'carrier'),
                                                            self._cal_distance(poi_j_pos, relay_car_pos, 'carrier'),
                                                            )
                    else:
                        uav_pos, relay_car_pos = self.agent_position[agent_type][agent_index], self.agent_position['carrier'][relay_car_index]
                        poi_j_pos = (99999, 999999)
                        sinr_G2A, R_G2A = compute_capacity_G2A(self.noma_config,
                                                                self._cal_distance(poi_i_pos, uav_pos, agent_type),
                                                                -1,
                                                                )
                        sinr_RE, R_RE = compute_capacity_RE(self.noma_config,
                                                            self._cal_distance(uav_pos, relay_car_pos, agent_type),
                                                            self._cal_distance(poi_i_pos, relay_car_pos, 'carrier'),
                                                            -1,
                                                            )
                    if self.DEBUG_MODE:
                        print(f"fake test 车机距离：{self._cal_distance(uav_pos, relay_car_pos, agent_type)}, "
                                f"无人机与poi i的距离{self._cal_distance(poi_i_pos, uav_pos, agent_type)},"
                                f"无人机与poi_j的距离{self._cal_distance(poi_j_pos, uav_pos, agent_type)}")

                capacity_i = min(R_G2A, R_RE) / 1e6  # 取两段信道的较小值

            self.rl_data_rate[agent_type][agent_index].append(capacity_i)
            collected_data = min(capacity_i * self.decided_collect_time[agent_type][agent_index], self.fake_poi_value[poi_i_index])
            self.fake_poi_value[poi_i_index] -= collected_data

            reward = collected_data / (self.USER_DATA_AMOUNT*10)
        
        self.poi_step_count += 1
        
        if self.poi_step_count == self.poi_step_all:
            done = True
            self.poi_step_status = False
        else:
            done = False    
            
        info = None
        obs = self.poi_get_obs()
        
        return  obs, reward, done, info
    
    def poi_get_obs(self):
        agent_type,agent_index,channel = self.get_type_index_channel_from_step(self.poi_step_count)
        #self.reset_poi_adj_dict()
        #第几次获取poi，得到的agent_type, agent的序号，agent的第几个channel
    
        agent_obs = []
        #active_agent_obs = []
        for t in self.UAV_TYPE:
            for i in range(self.NUM_UAV[t]):
                agent_obs.append(self.agent_position[t][i][0] / self.MAP_X)
                agent_obs.append(self.agent_position[t][i][1] / self.MAP_Y)
                agent_obs.append(self.decided_collect_time[t][i]/self.TIME_SLOT)
                if t == agent_type and i == agent_index:
                    agent_obs.append(1)
                else:
                    agent_obs.append(0)
                    
                agent_obs.extend(self.get_obs_from_access_list(self.decided_sorted_access[agent_type][agent_index]))
                
                if agent_type =='uav':
                    agent_obs.extend(self.get_obs_from_access_list(self.decided_sorted_access['carrier'][agent_index]))
                else:
                    agent_obs.extend([0 for _ in range(2*self.CHANNEL_NUM)])

        #agent_obs: [agent_pos_x,agent_pos_y,collect_time,1/0,agent_collect_history, uav_relay_collect_history]*all_agent
        # 格式是所有agent的信息，每个agent是（该agent的x，y，收集时间，是否是当前agent，该agent的收集历史（长度为channel_num*2，每个channel从poi（x,y）收集),uav_relay_collect_history(uav对应的carrier的收集历史)
        # 总长度是all_agent*（4+4*channel_num）                
        agent_obs = np.array(agent_obs)
        poi_value = np.array(self.fake_poi_value)
        poi_position = np.array(self.poi_position)
        
        active_agent_pos = np.array(self.agent_position[agent_type][agent_index])
        relay_agent_pos = np.array(self.agent_position['carrier'][agent_index])
        active_poi_dis = self._cal_distance(poi_position,active_agent_pos,type=agent_type)
        relay_poi_dis = self._cal_distance(poi_position,relay_agent_pos,type=agent_type) if agent_type =='uav' else np.zeros_like(active_poi_dis)
        
        poi_obs = np.stack([poi_position[:, 0]  / self.MAP_X,
                                             poi_position[:, 1]  / self.MAP_Y,
                                             poi_value / self.get_poi_dividend(), 
                                             self.poi_allocated_list.reshape(-1),
                                             active_poi_dis.reshape(-1) / (self.SCALE*self.MAP_X*np.sqrt(2)),
                                             relay_poi_dis.reshape(-1) / (self.SCALE*self.MAP_X*np.sqrt(2)),
                                             ], axis=1)
        #poi_obs: shape: (poi_num, 5),每行代表一个poi的信息，包括位置xy，数据量，是否被分配(之前被采集过），到当前agent的距离，到当前agent的relay的距离
        agent_dim = 4+2*self.CHANNEL_NUM*2
        carrier_obs = agent_obs[:agent_dim*self.NUM_UAV['carrier']].reshape(self.NUM_UAV['carrier'],agent_dim)
        uav_obs = agent_obs[agent_dim*self.NUM_UAV['carrier']:].reshape(self.NUM_UAV['uav'],agent_dim)
        
        poi_avail_list = np.array(list(self.action_to_poi_index[agent_type][agent_index].values()))
        #当前agent的当前channel的可选poi的index，作为切片的索引
        agent_poi_obs = poi_obs[poi_avail_list]#可选的poi的信息，切片后的poi_obs，shape: (可选poi数量，5)
        
        
        mask = self.get_poi_step_mask()
        #poi_obs = np.zeros_like(poi_obs)
        #poi_obs[np.array(mask[1:])==0] = 0 
        state = np.concatenate([agent_obs,agent_poi_obs.flatten()])
        #state = np.concatenate([active_agent_obs,poi_obs.flatten()])
        
        obs = {}
        for key1, s_dict in self.poi_adj_dict.items():
            for key2, adj in s_dict.items():
                obs[f"{key1}-{key2}"] = adj
        
        total_id = [agent_index] if agent_type=='carrier' else [agent_index + self.NUM_UAV['carrier']]
        obs.update({
            'id':total_id,#shape:(1) 含义：当前agent的id
            'state': state,#shape:(num_of_agent*(4+4*channel_num)+15*6) 含义：agent的信息，poi的信息。agent的信息包括位置xy，收集时间，是否是当前agent，收集历史(4*channel)，15是当前agent的可选poi数量，6是poi信息（位置xy，数据量，是否被分配，到当前agent的距禿，到当前agent的relay的距离）
            'mask':mask,#shape：(1+poi_num),含义：当前agent的当前channel的可选poi的mask，开头的1表示全都不可选，开头是0表示存在可选的poi
            'carrier':carrier_obs,#shape: (num_of_carrier, 4+4*channel_num),每行代表一个agent的信息,包括位置xy，收集时间，是否是当前agent，收集历史(4*channel)
            'uav':uav_obs,#shape: (num_of_uav, 4+4*channel_num),每行代表一个agent的信息,包括位置xy，收集时间，是否是当前agent，收集历史(4*channel)
            'poi':poi_obs[poi_avail_list] if self.NEAR_SELECTION_MODE else poi_obs#shape:(可选poi数 or poi_num,6), 6代表了所有poi相对于当前agent当前channel而言的信息
        })
        
        return obs 
    
    def get_obs_from_access_list(self,access_list):
        obs = []
        for p_id in access_list:
            if p_id != -1:
                obs.append(self.poi_position[p_id][0] / self.MAP_X)
                obs.append(self.poi_position[p_id][1] / self.MAP_Y)
            else:
                obs.extend([0,0])
        for _ in range(self.CHANNEL_NUM-len(access_list)):
            obs.extend([0,0])
        
        assert len(obs) == self.CHANNEL_NUM *2 
        return obs
      
    def get_poi_step_mask(self):
        if self.poi_step_count  >= self.poi_step_all:
            return [1 for _ in range(self.POI_NUM+1)] if not self.NEAR_SELECTION_MODE else [1 for _ in range(self.POI_SELECTION_NUM)]
        agent_type,agent_index,channel = self.get_type_index_channel_from_step(self.poi_step_count )
        
        pos = self.agent_position[agent_type][agent_index]
        distances = self._cal_distance(self.poi_position, np.array(pos), agent_type)     
        if self.NEAR_SELECTION_MODE:   
            mask = self.near_action_mask[agent_type][agent_index]
        else:
            mask = (distances <= self.COLLECT_RANGE[agent_type]).astype(np.int32).tolist()
        
        if sum(mask) == 0:
            return [1] + mask
        else:
            return [0] + mask

    
    def get_type_index_channel_from_step(self,step):
        agent_type = 'carrier' if step < self.NUM_UAV['carrier']*self.CHANNEL_NUM else 'uav'
        agent_index =  (step // (self.CHANNEL_NUM)) % self.NUM_UAV['carrier']
        channel = step % self.CHANNEL_NUM
        assert 0<=agent_index<self.NUM_UAV[agent_type]
                
        return agent_type,agent_index,channel

    def continue_step(self):
        
        if self.RL_GREEDY_REWARD:
            relay_dict, sorted_access = self._relay_association(), self._access_determin(self.CHANNEL_NUM)
        else:
            relay_dict, sorted_access = self.decided_relay_dict,self.decided_sorted_access
        
        uav_reward = {key: np.zeros([self.NUM_UAV[key]]) for key in self.UAV_TYPE}
        uav_penalty = {key: np.zeros([self.NUM_UAV[key]]) for key in self.UAV_TYPE}
        # uav_data_collect = {key: np.zeros([self.NUM_UAV[key]]) for key in self.UAV_TYPE}

        energy_consumption_all = 0
        all_uav_pois, all_ugv_pois = [], []
        distance = self.distance
        
        all_collected = {my_type: np.zeros(self.NUM_UAV[my_type]) for my_type in self.UAV_TYPE}
        all_data_rate = {my_type: np.zeros(self.NUM_UAV[my_type]) for my_type in self.UAV_TYPE}
        # all_ratio = {my_type: np.zeros(self.NUM_UAV[my_type]) for my_type in self.UAV_TYPE}
        for type in self.UAV_TYPE:
            for uav_index in range(self.NUM_UAV[type]):
                collect_time = max(0, self.TIME_SLOT - distance[type][uav_index] / self.UAV_SPEED[type])

                if self.NOMA_MODE:
                    r, collected_data, data_rate, uav_poi, ugv_poi, collected_list, temp_poi_aoi_list = (
                        self._collect_data_by_noma(type, uav_index, relay_dict, sorted_access,
                                                   collect_time))
                    all_uav_pois.extend(uav_poi)
                    all_ugv_pois.extend(ugv_poi)
                    all_collected[type][uav_index] = collected_data
                    all_data_rate[type][uav_index] = data_rate
                    self.greedy_data_rate[type][uav_index].append(data_rate)
                    # if type == 'uav':
                    #     all_ratio[type][uav_index] = g2a_rate / (relay_rate + 1e-10)
                else:
                    r, collected_data = self._collect_data_from_poi(type, uav_index, collect_time)

                energy_consuming = self._cal_energy_consuming(self.distance[type][uav_index], type)
                energy_consumption_all += energy_consuming
                uav_reward[type][uav_index] -= energy_consuming * self.energy_penalty
                self.uav_data_collect[type][uav_index].append(collected_data)

                 #下面求VOI的部分，added by zf，24.11.18
                total_voi_decline = 0
                for channel_index in range(self.CHANNEL_NUM):
                    collected_data_this_channel = collected_list[channel_index]
                    aoi_this_channel = temp_poi_aoi_list[channel_index]
                    voi_lambda = min(collected_data_this_channel/self.USER_DATA_AMOUNT, aoi_this_channel)
                    Decline = (exp(self.VOI_K * (aoi_this_channel - voi_lambda) ) - exp(self.VOI_K*aoi_this_channel))/self.VOI_K
                    assert voi_lambda >= Decline
                    voi_decline = (1-self.VOI_BETA)*self.USER_DATA_AMOUNT*(voi_lambda-Decline)
                    total_voi_decline += voi_decline
                
                self.uav_voi_collect[type][uav_index].append(collected_data - total_voi_decline)
                self.uav_voi_decline[type][uav_index].append(total_voi_decline)
                #上面求VOI的部分，added by zf，24.11.18

                uav_reward[type][uav_index] += r * (10 ** -3)  # * (2**-4)
                # print( uav_reward[type][uav_index])

                if type == 'uav':
                    # dis_reward =  self._cal_distance(self.agent_position['carrier'][relay_dict[uav_index]],
                    # self.agent_position['uav'][uav_index])*0.0001
                    # uav_reward[type][uav_index] +
                    uav_reward['carrier'][self.decided_relay_dict[uav_index]] += r * (10 ** -3) / 5

                if type == 'carrier' and self.config("carrier_explore_reward"):
                    # print(uav_reward[type][uav_index])
                    uav_reward[type][uav_index] -= math.log(
                        self.visited_nodes_count[self.carrier_node[uav_index]] + 1) * 0.1

                    # print(-math.log(self.visited_nodes_count[self.carrier_node[uav_index]]+1) * 0.05)

        if self.COLLECT_MODE == 1 or self.COLLECT_MODE == 2:
            self.check_arrival()
            self.aoi_history.append(np.mean(np.asarray(self.poi_value) / self.USER_DATA_AMOUNT))
        
            self.emergency_history.append(
                    np.mean([1 if aoi / self.USER_DATA_AMOUNT >= self.AOI_THRESHOLD else 0 for aoi in self.poi_value]))
            aoi_reward = self.aoi_history[-2] - self.aoi_history[-1]
            # aoi_reward -= self.emergency_history[-1] * self.THRESHOLD_PENALTY
            #
            # for type in self.UAV_TYPE:
            #     for uav_index in range(self.NUM_UAV[type]):
            #         uav_reward[type][uav_index] -= self.emergency_history[-1] * self.THRESHOLD_PENALTY

            for type in self.UAV_TYPE:
                type_sum = sum(uav_reward[type])
                for uav_index in range(self.NUM_UAV[type]):
                    if self.CENTRALIZED:
                        if self.reward_type == 'prod':
                            aux = 1e-6 * all_collected[type][uav_index] * all_data_rate[type][uav_index]
                        elif self.reward_type == 'square':
                            aux = 5e-7 * all_collected[type][uav_index] ** 2
                        elif self.reward_type == 'prod_thre':
                            aux = (1e-6 * all_collected[type][uav_index] *
                                   max(0, all_data_rate[type][uav_index] - self.DATA_RATE_THRE))
                        elif self.reward_type == 'sum':
                            aux = 1e-3 * (all_collected[type][uav_index] + all_data_rate[type][uav_index])
                        else:
                            aux = 0

                        #uav_reward[type][uav_index] = aoi_reward - energy_consumption_all * 1e-6 + aux
                       
                        
                        uav_reward[type][uav_index] += aoi_reward  + aux
                        if self.dis_bonus:
                            if type == 'carrier':
                                dis = 0
                                for other_uav in range(self.NUM_UAV[type]):
                                    dis += self._cal_distance(self.agent_position[type][uav_index],
                                                              self.agent_position[type][other_uav], type=type)
                                dis /= (self.NUM_UAV[type] - 1)
                                uav_reward[type][uav_index] += 5e-5 * min(dis, 500)

        done = self._is_episode_done()
        self.step_count += 1

        self.poi_history.append({
            'pos': copy.deepcopy(self.poi_position).reshape(-1, 2),
            'val': copy.deepcopy(self.poi_value)})

        info = {}
        info_old = {}

        if self.NOMA_MODE:
            self.record_info_for_agent(info, all_collected, "collected_data")
            self.record_info_for_agent(info, all_data_rate, "data_rate")
            # self.record_info_for_agent(info, all_ratio, "g2a_relay_ratio")
            # data rate hinge reward
            # uav_reward['uav'][uav_id] += max(self.DATA_RATE_THRE - data_rate, 0)

        info_old.update({"uav_pois": all_uav_pois, "ugv_pois": all_ugv_pois})

        if done:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                info = self.summary_info(info)
                info_old = copy.deepcopy(info)
                # updating info_old means the poi visit history is not necessary in every trajectory during training
                info_old.update({"uav_pois": all_uav_pois, "ugv_pois": all_ugv_pois})
                info = self.save_trajectory(info)

        global_reward = {}
        for type in self.UAV_TYPE:
            global_reward[type] = np.mean(uav_reward[type]) + np.mean(uav_penalty[type])
            self.episodic_reward_list[type].append(global_reward[type])
            self.single_uav_reward_list[type].append((uav_reward[type] + uav_penalty[type]).tolist())
        obs = self.get_obs()
        info_old['step'] = self.step_count
        
        if self.POI_DECISION_MODE:
            collect_reward = {'poi_'+key: np.zeros([self.NUM_UAV[key]]) for key in self.UAV_TYPE}
            for type in self.UAV_TYPE:
                for uav_index in range(self.NUM_UAV[type]):
                    collect_reward['poi_'+type][uav_index]+= all_data_rate[type][uav_index]*1e-3
                self.single_uav_reward_list['poi_'+type].append(collect_reward['poi_'+type].tolist())
        else:
            collect_reward = {}
        return obs, {**uav_reward,**collect_reward}, done, info_old
    

def myfloor(x):
    
    a = x.astype(np.int)
    return a


def compute_theta(dpx, dpy, dpz):
    '''弧度制'''
    # 法一 无法达到
    # dpx>0 dpy>0时，theta在第一象限
    # dpx<0 dpy>0时，theta在第二象限
    # dpx<0 dpy<0时，theta在第三象限
    # dpx>0 dpy<0时，theta在第四象限
    theta = math.atan(dpy / (dpx + 1e-8))

    # 法二 2022/1/10 可以达到 不过y轴是反的 但无伤大雅~
    x1, y1 = 0, 0
    x2, y2 = dpx, dpy
    ang1 = np.arctan2(y1, x1)
    ang2 = np.arctan2(y2, x2)
    # theta = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    theta = (ang1 - ang2) % (2 * np.pi)  # theta in [0, 2*pi]

    return theta


def row_normalize(matrix):
    # 计算每行的和
    row_sums = matrix.sum(axis=1)[:, np.newaxis]
    # 避免除以0
    row_sums[row_sums == 0] = 1
    # 将每行除以其和
    normalized = matrix / row_sums
    return normalized


exclude_names = {"uav_pois", "ugv_pois","reward_list"}
