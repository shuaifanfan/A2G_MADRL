import math
import numpy as np
import torch
import torch.nn as nn
from source_code.algorithms.mat.utils.transformer_act import discrete_autoregreesive_act, discrete_parallel_act, \
    continuous_autoregreesive_act, continuous_parallel_act
from source_code.algorithms.mat.utils.util import check, init
from source_code.algorithms.model.hgcn import HeteGCNLayer
from torch.nn import functional as F


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False, not_condition=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
        #                      .view(1, 1, n_agent + 1, n_agent + 1))
       
        if not_condition:
            self.register_buffer("mask", torch.tril(torch.eye(n_agent + 1, n_agent + 1))
                                .view(1, 1, n_agent + 1, n_agent + 1))
        else:
            self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                                .view(1, 1, n_agent + 1, n_agent + 1))

        # self.m_mask = torch.tril(torch.ones(n_agent + 1, n_agent + 1)).view(1, 1, n_agent + 1, n_agent + 1)
        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L].to(key.device) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent, masked):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=masked)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent, not_condition=False):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True, not_condition=not_condition)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True, not_condition=not_condition)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc):
        #print("changed transfomer----------------------------------------------------------------------------------")
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, mask_obs, use_hgcn):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        self.use_hgcn = use_hgcn
        self.seperate_embedding = False
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        HIDDEN_STATE = 256

        if self.seperate_embedding:
            self.agent_dim = n_agent + 2 + n_agent*3 
            self.poi_dim = obs_dim - self.agent_dim 
            self.agent_encoder = nn.Sequential(
                                         init_(nn.Linear(self.agent_dim, 128), activate=True), nn.GELU(),
                                         init_(nn.Linear(128, n_embd), activate=True), nn.GELU())
            self.poi_encoder = nn.Sequential(
                                         init_(nn.Linear(self.poi_dim, 256), activate=True), nn.GELU(),
                                         init_(nn.Linear(256, 2*n_embd), activate=True), nn.GELU())
            self.other_dim = obs_dim - self.agent_dim - self.poi_dim
            self.merge_encode = nn.Sequential(
                                         init_(nn.Linear(n_embd*3, 256), activate=True), nn.GELU(),
                                         init_(nn.Linear(256, n_embd), activate=True), nn.GELU())
            
            
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, HIDDEN_STATE), activate=True), nn.GELU(),
                                         init_(nn.Linear(HIDDEN_STATE, n_embd), activate=True), nn.GELU())
        
        

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent, mask_obs) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))
        if self.use_hgcn:
            self.hgcn_embedding = nn.Sequential(
                init_(nn.Linear(n_embd * 2, n_embd), activate=True), nn.GELU(),
            )

    def forward(self, state, obs, hgcn_results=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:#不用state
            if not self.use_hgcn:
                if self.seperate_embedding:
                    agent_emb = self.agent_encoder(obs[:,:,:self.agent_dim])
                    poi_emb = self.poi_encoder(obs[:,:,self.agent_dim:])
                    obs_embeddings = self.merge_encode(torch.cat([agent_emb,poi_emb],dim=2))
                else:
                    obs_embeddings = self.obs_encoder(obs)
            else:
                obs_embeddings = self.obs_encoder(obs)
                obs_embeddings = torch.cat([obs_embeddings, hgcn_results], dim=2)
                obs_embeddings = self.hgcn_embedding(obs_embeddings)
            x = obs_embeddings
        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)

        return v_loc, rep


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent, cat_position=False,
                 action_type='Discrete', dec_actor=False, share_actor=False, device='cpu', ar_policy=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type
        self.device = device
        self.n_agent = n_agent

        self.cat_position = cat_position
        extra_features = 0 if not self.cat_position else 2
        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

        if self.dec_actor:
            if self.share_actor:
                print("mac_dec!!!!!")
                self.mlp = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(),
                                         nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),
                                         nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, action_dim)))
            else: 
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(obs_dim),
                                          init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(),
                                          nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),
                                          nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, action_dim)))
                    self.mlp.append(actor)
        else:
            # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))
            if action_type == 'Discrete': #这个
                self.action_encoder = nn.Sequential(
                    init_(nn.Linear(action_dim + 1 + extra_features, n_embd, bias=False), activate=True),
                    nn.GELU())
            else:
                self.action_encoder = nn.Sequential(
                    init_(nn.Linear(action_dim + extra_features, n_embd), activate=True), nn.GELU())
            HIDDEN_STATE = 256
            self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                             init_(nn.Linear(obs_dim, HIDDEN_STATE), activate=True), nn.GELU(),
                                             init_(nn.Linear(HIDDEN_STATE, n_embd), activate=True), nn.GELU())
            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.Sequential(
                *[DecodeBlock(n_embd, n_head, n_agent, not_condition=not ar_policy) for _ in range(n_block)])
            self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, action_dim)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else: #用这个
            if self.cat_position: #用这个
                id = torch.argmax(obs[:, :, :self.n_agent], dim=2) * 3
                index = torch.zeros(obs.shape[0], obs.shape[1], 2, dtype=torch.long, device=obs.device)
                index[:, :, 0] = self.n_agent + 2 + id
                index[:, :, 1] = self.n_agent + 2 + id + 1
                position = torch.gather(obs, 2, index)
                # position = torch.zeros(B, N, 2).to(self.device)
                # for i in range(N):
                #     start_idx = self.n_agent + i * 2
                #     end_idx = self.n_agent+ (i + 1) * 2
                #     position[:, i, :] = obs[:, i, start_idx:end_idx]
                mask = torch.max(action, dim=2)[0].unsqueeze(-1).repeat(1, 1, 2)
                position = mask * position
                action = torch.cat([action, position], dim=2)
            action_embeddings = self.action_encoder(action)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, obs_rep)
            logit = self.head(x)
           # print("decoder ouput logits dim:",logit.shape)
           #shape torch.Size([16,4 ,15]),16是batch_size,4是agent数，15是action数
        return logit


class MultiAgentTransformer(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False, **kwargs):
        super(MultiAgentTransformer, self).__init__()

        self.use_lambda = kwargs['use_lambda']
        self.task_emb_dim = kwargs['task_emb_dim']
        self.use_mate = kwargs['use_mate']
        self.use_gcn = kwargs['use_gcn']
        self.gcn_emb_dim = kwargs['gcn_emb_dim']
        self.gcn_nodes_num = kwargs['gcn_nodes_num']
        self.cat_position = kwargs['cat_position']
        self.input_args = kwargs['input_args']
        self.n_poi = kwargs['n_poi']
        self.n_node = kwargs['n_node']
        mask_obs = self.input_args.mask_obs
        self.use_hgcn = self.input_args.use_hgcn
        self.ar_policy = self.input_args.use_ar_policy

        extra_features = 0
        if self.use_lambda:
            extra_features += 1
        if self.use_mate:
            extra_features += self.task_emb_dim * 2
        if self.use_gcn and not self.use_hgcn:
            extra_features += n_embd

        # if self.use_gcn:
        #     self.gcn_model = kwargs['gcn_model']

        # if self.use_hgcn:
        #     type_att_size = n_embd
        #     type_fusion = 'mean'
        #     net_schema = {
        #         'uav': ['carrier', 'poi', 'road', 'epoi'],  # n x n, n x poi
        #         'carrier': ['uav', 'poi', 'road', 'epoi'],  # n x n, n x poi, n x node
        #         'poi': ['uav', 'carrier'],  # poi x n, poi x n
        #         'road': ['carrier']  # node x n
        #     }
        #     layer_shape = [
        #         {'uav': 2, 'carrier': 2, 'poi': 3, 'road': 1},
        #         # {'uav':32,'carrier':32,'poi':32,'road':32},
        #         {'uav': n_embd, 'carrier': n_embd, 'poi': n_embd, 'road': n_embd}
        #     ]
        #     # layer_shape = [
        #     #     {'uav': 2, 'carrier': 2, 'poi': 3, 'road': 1},
        #     #     {'uav': 16, 'carrier': 16, 'poi': 4, 'road': 4},
        #     #     {'uav': 32, 'carrier': 32, 'poi': 4, 'road': 4}
        #     # ]
        #     self.hgcn1 = HeteGCNLayer(net_schema, layer_shape[0], layer_shape[1], type_fusion, type_att_size,
        #                               self.input_args)
        #     # self.hgcn2 = HeteGCNLayer(net_schema, layer_shape[1], layer_shape[2], type_fusion, type_att_size)
        #     self.poi_embedding = nn.Sequential(
        #         init_(nn.Linear(self.n_poi * 4, 64), activate=True), nn.GELU(),
        #     )
        #     self.node_embedding = nn.Sequential(
        #         init_(nn.Linear(self.n_node * 4, 64), activate=True), nn.GELU(),
        #     )
        #     # self.gcn1 = GCNConv(3,3)
        #     # self.gcn2 = GCNConv(3,self.gcn_emb_dim)
        #     # self.gcn3 = nn.Linear(self.gcn_emb_dim*self.gcn_nodes_num,n_embd)
        self.extra_features = extra_features
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self.obs_dim = obs_dim
        self.rep_learning = self.input_args.rep_learning
        self.random_permutation = self.input_args.random_permutation

        # state unused
        self.state_dim = state_dim

        # if self.use_hgcn:
        #     self.encoder = Encoder(state_dim, obs_dim + n_embd, n_block, n_embd, n_head, n_agent, encode_state, mask_obs,
        #                            self.use_hgcn)
        # else:
        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state,
                               mask_obs, self.use_hgcn)

        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               self.cat_position,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor, device=device,
                               ar_policy=self.ar_policy)

        self.save_emb = False
        if self.input_args.test:
            self.emb_list = []
            self.save_emb = True
        if self.rep_learning:
            self.vae_encoder = nn.Sequential(
                nn.GELU(),
                init_(nn.Linear(n_embd, self.task_emb_dim * 2), activate=True),
            )

            HIDDEN_STATE = 256
            self.vae_decoder = nn.Sequential(
                init_(nn.Linear(self.task_emb_dim + obs_dim + action_dim + extra_features, HIDDEN_STATE),
                      activate=True),
                nn.GELU(),
                init_(nn.Linear(HIDDEN_STATE, obs_dim + 1), activate=True)
            )

            self.params = list(self.encoder.parameters()) + list(self.encoder.parameters()) + list(
                self.vae_encoder.parameters()) + list(self.vae_decoder.parameters())
            self.optimizer = torch.optim.Adam(self.params, 1e-5)
            self.obs_loss_coef = 1.0
            self.rew_loss_coef = 1.0
            self.kl_loss_coef = 0.1
            self.max_grad_norm = 0.5

        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None, task_emb=None,
                graph_inputs=None, hgcn_inputs=None, memory=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        ori_shape = np.shape(state)
        # state = np.zeros((*ori_shape[:-1], self.state_dim), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        # if self.use_hgcn:
        #     hgcn_results, new_memory = self.hgcn_forward(state, obs, graph_inputs, hgcn_inputs, memory)
        #     obs = torch.cat([obs, hgcn_results], dim=2)
        # else:
        hgcn_results, new_memory = None, None
        # if self.use_gcn and not self.use_hgcn:
        #     tot_step, agent_num = obs.shape[0], obs.shape[1]
        #     after = []
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(state, obs, hgcn_results)


        if self.action_type == 'Discrete':
            if action.shape[-1] == 1:
                action = action.long()
            action_log, entropy, distri = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)
            distri = None

        return action_log, v_loc, entropy, new_memory, distri

    def get_actions(self, state, obs, available_actions=None, deterministic=False, task_emb=None, graph_inputs=None,
                    hgcn_inputs=None, memory=None):
        # state unused
        # ori_shape = np.shape(obs)
        # state = np.zeros((*ori_shape[:-1], self.state_dim), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self.use_hgcn:
            hgcn_results, new_memory = self.hgcn_forward(state, obs, graph_inputs, hgcn_inputs, memory)
            obs = torch.cat([obs, hgcn_results], dim=2)
        else:
            hgcn_results, new_memory = None, None


        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(state, obs, hgcn_results)
        if self.save_emb:
            self.emb_list.append(obs_rep)


        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic)
        return output_action, output_action_log, v_loc, new_memory

    def get_values(self, state, obs, available_actions=None, task_emb=None, graph_inputs=None, hgcn_inputs=None,
                   memory=None):
        # state unused
        ori_shape = np.shape(obs)
        # state = np.zeros((*ori_shape[:-1], self.state_dim), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)

        if self.use_hgcn:
            hgcn_results, new_memory = self.hgcn_forward(state, obs, graph_inputs, hgcn_inputs, memory)
            obs = torch.cat([obs, hgcn_results], dim=2)
        else:
            hgcn_results, new_memory = None, None


        v_tot, obs_rep = self.encoder(state, obs, hgcn_results)
        return v_tot

    def hgcn_forward(self, state, obs, graph_inputs, hgcn_inputs, memory=None):

        cols = torch.argmax(obs[:, :, :self.n_agent], dim=2)
        B, N, _ = obs.size()
        rows = np.indices((B, N))[0]

        ft_dict = {}
        B = state.shape[0]
        ft_dict['carrier'] = state[:, 0, self.n_agent: self.n_agent + int(self.n_agent // 2 * 2)].view(B,
                                                                                                       self.n_agent // 2,
                                                                                                       2)
        ft_dict['uav'] = state[:, 0,
                         self.n_agent + int(self.n_agent // 2 * 2):self.n_agent + int(self.n_agent // 2 * 2) * 2].view(
            B,
            self.n_agent // 2,
            2)
        ft_dict['poi'] = state[:, 0, self.n_agent + self.n_agent * 2:].view(B, self.n_poi, 3)
        ft_dict['road'] = self.gcn_model.forward_with_origin_output(graph_inputs, ma_dim=False).permute(0, 2,
                                                                                                        1).contiguous().view(
            B, -1, 1)
        adj_dict = dict_convert(hgcn_inputs)
        x_dict, memory_new = self.hgcn1(ft_dict, adj_dict, memory)
        x_dict = self.non_linear(x_dict)
        if memory_new['uav'] is not None:
            memory_new = torch.cat([memory_new['carrier'], memory_new['uav']], dim=1)
        else:
            memory_new = None
      
        results = torch.cat([x_dict['carrier'], x_dict['uav']], dim=1)  # B,4,16
        # results = torch.cat([results, poi_embedding, node_embedding], dim=-1)
        results = results[rows, cols]

        return results, memory_new

    def gcn_forward(self, graph_inputs):
        AssertionError
        nodes, edges = graph_inputs
        nodes = nodes.view(-1, *nodes.shape[2:])
        edges = edges.view(-1, *edges.shape[2:])

        gcn_emb = self.gcn1(nodes, edges[0].to(torch.int64))
        gcn_emb = F.leaky_relu(gcn_emb, negative_slope=0.1)
        gcn_emb = self.gcn2(gcn_emb, edges[0].to(torch.int64))
        gcn_emb = gcn_emb.view(-1, self.gcn_emb_dim * self.gcn_nodes_num)
        gcn_emb = self.gcn3(gcn_emb).view(-1, self.n_agent, self.n_embd)

        return gcn_emb

    def update_rep(self, obss, acts, rews, next_obss, done, graph_inputs=None):

        if self.use_gcn:
            # gcn_emb = self.gcn_forward(graph_inputs)
            gcn_emb = self.gcn_model.forward(graph_inputs, ma_dim=True)
            obss = torch.cat([obss, gcn_emb], dim=2)

        mask = ~done.squeeze(-1)
        B, N, _ = obss.shape
        obss = obss[mask].view(-1, N, self.obs_dim + self.extra_features)
        acts = torch.nn.functional.one_hot(acts[mask].view(-1, N), self.action_dim).float()
        rews = rews[mask].view(-1, N)
        next_obss = next_obss[mask].view(-1, N, self.obs_dim)

        v_tot, obs_rep = self.encoder(obss, obss)  # have no state, use obs twice
        mu_and_var = self.vae_encoder(obs_rep)
        mu, log_var = mu_and_var[..., 0:self.task_emb_dim], mu_and_var[..., self.task_emb_dim:]

        loss = 0
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        z = self.reparameterise(mu, log_var)

        pred = self.vae_decoder(torch.cat([z, obss, acts], dim=-1))
        pred_obss = pred[..., :self.obs_dim]
        pred_rew = pred[..., -1]

        obs_loss = (next_obss - pred_obss).pow(2).mean()
        rew_loss = (pred_rew - rews).pow(2).mean()

        # compute total loss
        loss = self.obs_loss_coef * obs_loss + self.rew_loss_coef * rew_loss + self.kl_loss_coef * kl_loss
        loss.backward()

        # gradient norm

        torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)

        self.optimizer.step()

        loss_dict = {
            "AE/obs_loss": obs_loss.item(),
            "AE/rew_loss": rew_loss.item(),
            "AE/kl_loss": kl_loss.item(),
            "AE/loss": loss.item(),
        }

        return loss_dict

    def save_obs_emb(self, path):

        import os
        np.save(os.path.join(path, f'map_{self.input_args.map}_task_emb.npy'),
                torch.stack(self.emb_list, dim=0).cpu().numpy())

    def reparameterise(self, mu, log_var):
        """
        Get VAE latent sample from distribution
        :param mu: mean for encoder's latent space
        :param log_var: log variance for encoder's latent space
        :return: sample of VAE distribution
        """
        # compute standard deviation from log variance
        std = torch.exp(0.5 * log_var)
        # get random sample with same dim as std
        eps = torch.randn_like(std)
        # sample from latent space
        sample = mu + (eps * std)
        return sample

    def non_linear(self, x_dict):
        y_dict = {}
        for k, v in x_dict.items():
            y_dict[k] = F.elu(v)
        return y_dict


def shuffle_agent_grid(batch_size, nb_agent, random=False):
    x = batch_size
    y = nb_agent
    rows = np.indices((x, y))[0]
    if random:
        cols = np.stack([np.random.permutation(y) for _ in range(x)])
    else:
        cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


def dict_convert(old_dict):
    adj_dict = {
        'uav': {},
        'carrier': {},
        'poi': {},
        'road': {}
    }

    adj_dict['uav']['carrier'] = old_dict['uav-carrier']
    adj_dict['uav']['poi'] = old_dict['uav-poi']
    adj_dict['uav']['epoi'] = old_dict['uav-epoi']
    adj_dict['uav']['road'] = old_dict['uav-road']

    adj_dict['carrier']['uav'] = old_dict['carrier-uav']
    adj_dict['carrier']['epoi'] = old_dict['carrier-epoi']
    adj_dict['carrier']['poi'] = old_dict['carrier-poi']
    adj_dict['carrier']['road'] = old_dict['carrier-road']

    adj_dict['poi']['uav'] = old_dict['poi-uav']
    adj_dict['poi']['carrier'] = old_dict['poi-carrier']

    adj_dict['road']['carrier'] = old_dict['road-carrier']

    return adj_dict
