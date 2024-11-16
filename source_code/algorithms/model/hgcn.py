import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super(PositionalEmbedding, self).__init__()

        self.dim = dim
        inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class PositionwiseFF(torch.nn.Module):
    def __init__(self, d_input, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.d_input = d_input
        self.d_inner = d_inner
        self.dropout = dropout
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_input, d_inner),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_inner, d_input),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input_):
        ff_out = self.ff(input_)
        return ff_out


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_input, d_inner, n_heads=4, dropout=0.05, dropouta=0.0):
        super(MultiHeadAttention, self).__init__()

        self.d_input = d_input
        self.d_inner = d_inner
        self.n_heads = n_heads

        # Linear transformation for keys & values for all heads at once for efficiency.
        # 2 for keys & values.
        self.linear_kv = torch.nn.Linear(d_input, (d_inner * n_heads * 2), bias=False, dtype=torch.float32)
        # for queries (will not be concatenated with memorized states so separate).
        self.linear_q = torch.nn.Linear(d_input, d_inner * n_heads, bias=False, dtype=torch.float32)

        # for positional embeddings.
        self.linear_p = torch.nn.Linear(d_input, d_inner * n_heads, bias=False, dtype=torch.float32)
        self.scale = 1 / (d_inner ** 0.5)  # for scaled dot product attention
        self.dropa = torch.nn.Dropout(dropouta)

        self.lout = torch.nn.Linear(d_inner * n_heads, d_input, bias=False, dtype=torch.float32)
        self.dropo = torch.nn.Dropout(dropout)

    def _rel_shift(self, x):
        # x shape: [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        zero_pad = torch.zeros(
            (x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype
        )
        return (
            torch.cat([zero_pad, x], dim=1)
            .view(x.size(1) + 1, x.size(0), *x.size()[2:])[1:]
            .view_as(x)
        )

    def forward(self, input_, pos_embs, memory, u, v, mask=None):
        """
        + pos_embs: positional embeddings passed separately to handle relative positions.
        + Arguments
            - input: torch.FloatTensor, shape - (seq, bs, self.d_input) = (20, 5, 8)
            - pos_embs: torch.FloatTensor, shape - (seq + prev_seq, bs, self.d_input) = (40, 1, 8)
            - memory: torch.FloatTensor, shape - (prev_seq, b, d_in) = (20, 5, 8)
            - u: torch.FloatTensor, shape - (num_heads, inner_dim) = (3 x )
            - v: torch.FloatTensor, shape - (num_heads, inner_dim)
            - mask: torch.FloatTensor, Optional = (20, 40, 1)
        + Returns
            - output: torch.FloatTensor, shape - (seq, bs, self.d_input)
        + symbols representing shape of the tensors
            - cs: current sequence length, b: batch, H: no. of heads
            - d: inner dimension, ps: previous sequence length
        """
        cur_seq = input_.shape[0]
        prev_seq = memory.shape[0]
        H, d = self.n_heads, self.d_inner
        # concat memory across sequence dimension
        # input_with_memory = [seq + prev_seq x B x d_input] = [40 x 5 x 8]
        input_with_memory = torch.cat([memory, input_], dim=0)
        input_with_memory = input_with_memory.to(torch.float32)
        # k_tfmd, v_tfmd = [seq + prev_seq x B x n_heads.d_head_inner], [seq + prev_seq x B x n_heads.d_head_inner]
        k_tfmd, v_tfmd = torch.chunk(
            self.linear_kv(input_with_memory),
            2,
            dim=-1,
        )
        # q_tfmd = [seq x B x n_heads.d_head_inner] = [20 x 5 x 96]
        q_tfmd = self.linear_q(input_)

        _, bs, _ = q_tfmd.shape
        assert bs == k_tfmd.shape[1]

        # content_attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        content_attn = torch.einsum(
            "ibhd,jbhd->ijbh",
            (
                (q_tfmd.view(cur_seq, bs, H, d) + u),
                k_tfmd.view(cur_seq + prev_seq, bs, H, d),
            ),
        )

        # p_tfmd: [seq + prev_seq x 1 x n_heads.d_head_inner] = [40 x 1 x 96]
        p_tfmd = self.linear_p(pos_embs)
        # position_attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        position_attn = torch.einsum(
            "ibhd,jhd->ijbh",
            (
                (q_tfmd.view(cur_seq, bs, H, d) + v),
                p_tfmd.view(cur_seq + prev_seq, H, d),
            ),
        )

        position_attn = self._rel_shift(position_attn)
        # attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        attn = content_attn + position_attn

        if mask is not None and mask.any().item():
            # fills float('-inf') where mask is True.
            attn = attn.masked_fill(mask[..., None], -float("inf"))
        # rescale to prevent values from exploding.
        # normalize across the value sequence dimension.
        attn = torch.softmax(attn * self.scale, dim=1)
        # attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        attn = self.dropa(attn)

        # attn_weighted_values = [curr x B x n_heads.d_inner] = [20 x 5 x 96]
        attn_weighted_values = (
            torch.einsum(
                "ijbh,jbhd->ibhd",
                (
                    attn,  # (cs, cs + ps, b, H)
                    v_tfmd.view(cur_seq + prev_seq, bs, H, d),  # (cs + ps, b, H, d)
                ),
            )  # (cs, b, H, d)
            .contiguous()  # we need to change the memory layout to make `view` work
            .view(cur_seq, bs, H * d)
        )  # (cs, b, H * d)

        # output = [curr x B x d_input] = [20 x 5 x 8]
        output = self.dropo(self.lout(attn_weighted_values))
        return output


class TemporalTX(torch.nn.Module):
    def __init__(
            self,
            d_input,
            n_heads=2,
            dropouta=0.0,
            input_args=None
    ):
        super(TemporalTX, self).__init__()
        d_head_inner = d_input
        self.pos_embs = PositionalEmbedding(d_input)
        n_layers = 1
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList(
            [
                MultiHeadAttention(
                    d_input,
                    d_head_inner,
                    n_heads=n_heads)
                for _ in range(n_layers)
            ]
        )

        self.sequence_len = input_args.hgcn_rollout_len
        self.d_input = d_input
        self.n_heads = n_heads
        self.d_head_inner = d_head_inner

        self.u, self.v = (
            # [n_heads x d_head_inner] = [3 x 32]
            torch.nn.Parameter(torch.zeros(self.n_heads, self.d_head_inner)),
            torch.nn.Parameter(torch.zeros(self.n_heads, self.d_head_inner)),
        )

    def init_memory(self, device=torch.device("cpu"), size=None):
        # return [
        #     # torch.empty(0, dtype=torch.float).to(device)
        #     torch.zeros( self.sequence_len, 5, self.n_input, dtype=torch.float).to(device)
        #     for _ in range(self.n_layers + 1)
        # ]
        if size != None:
            return torch.zeros(self.n_layers + 1, self.sequence_len, size[1], size[2], dtype=float).to(device)

        return torch.zeros(self.n_layers + 1, self.sequence_len, self.d_input, dtype=torch.float).to(device)

    def update_memory(self, previous_memory, hidden_states):
        """
        + Arguments
            - previous_memory: List[torch.FloatTensor],
            - hidden_states: List[torch.FloatTensor]
        """
        assert len(hidden_states) == len(previous_memory)
        mem_len, seq_len = previous_memory[0].size(0), hidden_states[0].size(0)
        # mem_len, seq_len = 3, hidden_states[0].size(0)
        # print(mem_len, seq_len) 20 1

        with torch.no_grad():
            new_memory = []
            end_idx = mem_len + seq_len  # 21 
            beg_idx = max(0, end_idx - mem_len)  # 1 
            for m, h in zip(previous_memory, hidden_states):
                cat = torch.cat([m, h], dim=0)
                new_memory.append(cat[beg_idx:end_idx].clone().detach())
        return new_memory

    def forward(self, inputs, memory=None):
        """
        + Arguments
            - inputs - torch.FloatTensor = [T x B x d_inner] = [20 x 5 x 8]
            - memory - Optional, list[torch.FloatTensor] = [[T x B x d_inner] x 5]
        """
        # print(inputs.size()) B x d_inner
        # print(memory.size()) layer+1, T, B, d_innder
        # inputs B x d_inner
        # memory n_layer x T x B x d_inner
        if memory is None:
            memory = self.init_memory(inputs.device, inputs.size())

        memory = list(torch.unbind(memory, dim=0))
        # print(inputs.size())  1 8 762
        # print(memory[0][0][0])

        assert len(memory) == len(self.layers) + 1

        cur_seq, bs = inputs.shape[:2]
        prev_seq = memory[0].size(0)

        # dec_attn_mask = [curr x curr + prev x 1] = [20 x 40 x 1]
        dec_attn_mask = (
            torch.triu(
                torch.ones((cur_seq, cur_seq + prev_seq)),
                diagonal=1 + prev_seq,
            )
            .bool()[..., None]
            .to(inputs.device)
        )

        pos_ips = torch.arange(cur_seq + prev_seq - 1, -1, -1.0, dtype=torch.float).to(
            inputs.device
        )
        # pos_embs = [curr + prev x 1 x d_input] = [40 x 1 x 8]
        pos_embs = self.pos_embs(pos_ips)
        if self.d_input % 2 != 0:
            pos_embs = pos_embs[:, :, :-1]

        hidden_states = [inputs]
        layer_out = inputs
        for mem, layer in zip(memory, self.layers):
            # layer_out = [curr x B x d_inner] = [20 x 5 x 8]
            layer_out = layer(
                layer_out,
                pos_embs,
                mem,
                self.u,
                self.v,
                mask=dec_attn_mask,
            )
            hidden_states.append(layer_out)

        # Memory is treated as a const., don't propagate through it
        # new_memory = [[T x B x d_inner] x 4]

        memory = self.update_memory(memory, hidden_states)
        memory = torch.stack(memory)
        # print(memory.size())  n_layer x T x B x d_inner
        return layer_out, memory


class HeteGCNLayer(nn.Module):

    def __init__(self, net_schema, in_layer_shape, out_layer_shape, type_fusion, type_att_size, input_args):
        super(HeteGCNLayer, self).__init__()

        self.net_schema = net_schema
        self.in_layer_shape = in_layer_shape
        self.out_layer_shape = out_layer_shape
        if input_args.decoupled_actor:
            self.features = ['uav', 'carrier','poi']
        else:
            self.features = ['uav', 'carrier']

        self.hete_agg = nn.ModuleDict()
        for k in net_schema: #k依次取字典的键：uav 、 carrier 、 poi
            #构造的参数分别是：k：node_type，net_schema[k]：邻居类型，in_layer_shape：输入的维度，out_layer_shape[k]：输出的维度，type_fusion：融合方式，input_args：参数
            self.hete_agg[k] = HeteAggregateLayer(k, net_schema[k], in_layer_shape, out_layer_shape[k], type_fusion,
                                                  type_att_size, input_args)

    def forward(self, x_dict, adj_dict, memory=None):
        """_summary_

        Args:
            x_dict (dict): 包括三个键值对：uav、carrier、poi，每个键值对的值是一个tensor，shape为[n_thread*num_agent,obs_dim]
            adj_dict (dict): 包括uav+car+poi的六个key-key对应的邻接矩阵，uav+car+poi的三个观测+agent_id

        Returns:
            _type_: _description_
        """

        ret_x_dict = {}
        memory_new = {}
        for k in self.hete_agg.keys():
            if k in self.features:
                ret_x_dict[k], memory_new[k] = self.hete_agg[k](x_dict, adj_dict[k], memory)
            else:
                ret_x_dict[k], memory_new[k] = x_dict[k], None

        return ret_x_dict, memory_new


class HeteAggregateLayer(nn.Module):
    """
    针对不同的节点类型，聚合不同的邻居节点类型，总共有uav、carrier、poi三种节点类型，有三种HeteAggregateLayer类    
    """

    def __init__(self, curr_k, nb_list, in_layer_shape, out_shape, type_fusion, type_att_size, input_args):
        """_summary_

        Args:
            curr_k (string): node_type 
            nb_list (list): 该节点要聚合的邻居节点类型
            in_layer_shape (dict): 输入的node_feature的维度。key是uav、carrier、poi，value是维度
            out_shape (dict): 输出的node_feature的维度。key是uav、carrier、poi，value是维度
            type_fusion (string): 聚合方式，default是mean
            type_att_size (_type_): _description_
            input_args (dict): 参数
        """
        super(HeteAggregateLayer, self).__init__()

        self.nb_list = nb_list
        self.curr_k = curr_k
        self.type_fusion = type_fusion

        self.W_rel = nn.ParameterDict()
        for k in nb_list:
            if k == 'epoi':
                continue
            self.W_rel[k] = nn.Parameter(torch.FloatTensor(in_layer_shape[k], out_shape))
            nn.init.xavier_uniform_(self.W_rel[k].data, gain=1.414)

        self.w_self = nn.Parameter(torch.FloatTensor(in_layer_shape[curr_k], out_shape))
        nn.init.xavier_uniform_(self.w_self.data, gain=1.414)

        self.bias = nn.Parameter(torch.FloatTensor(1, out_shape))
        nn.init.xavier_uniform_(self.bias.data, gain=1.414)

        if type_fusion == 'att':
            self.w_query = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
            nn.init.xavier_uniform_(self.w_query.data, gain=1.414)
            self.w_keys = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
            nn.init.xavier_uniform_(self.w_keys.data, gain=1.414)
            self.w_att = nn.Parameter(torch.FloatTensor(2 * type_att_size, 1))
            nn.init.xavier_uniform_(self.w_att.data, gain=1.414)

        if input_args.use_temporal_type and curr_k in ['carrier', 'uav']:
            self.temporal_type = TemporalTX(out_shape, input_args=input_args)
        self.input_args = input_args

    def forward(self, x_dict, adj_dict, memory=None):
        """_summary_

        Args:
            x_dict (dict): 包括三个键值对：uav、carrier、poi，每个键值对的值是一个tensor，shape为[n_thread*num_agent,obs_dim]
            adj_dict (dict): 包括uav+car+poi的六个key-key对应的邻接矩阵，uav+car+poi的三个观测+agent_id

        Returns:
            _type_: _description_
        """

        self_ft = torch.matmul(x_dict[self.curr_k], self.w_self)
        B = self_ft.shape[0]
        nb_ft_list = [self_ft]
        nb_name = [self.curr_k]
        for k in self.nb_list:
            if k == 'epoi':
                nb_ft = torch.matmul(x_dict['poi'], self.W_rel['poi'])
                nb_ft = torch.matmul(adj_dict['epoi'], nb_ft)
            else:
                nb_ft = torch.matmul(x_dict[k], self.W_rel[k])
                nb_ft = torch.matmul(adj_dict[k], nb_ft)
            nb_ft_list.append(nb_ft)
            nb_name.append(k)

        if self.type_fusion == 'mean':
            agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mean(1)  # 1,2,32

        elif self.type_fusion == 'att':
            att_query = torch.matmul(self_ft, self.w_query).repeat(1, len(nb_ft_list), 1)
            att_keys = torch.matmul(torch.cat(nb_ft_list, 1), self.w_keys)
            att_input = torch.cat([att_keys, att_query], 2)  # B,6,128
            e = F.elu(torch.matmul(att_input, self.w_att))  # 1,6,1
            attention = F.softmax(e.view(B, len(nb_ft_list), -1).transpose(1, 2), dim=2)  # B,n_curk,len(nb)
            agg_nb_ft = torch.cat([nb_ft.unsqueeze(2) for nb_ft in nb_ft_list], 2).mul(attention.unsqueeze(-1)).sum(2)
            # print('curr key: ', self.curr_k, 'nb att: ', nb_name, attention.mean(0).tolist())
            # 1,3,2,32 x  1,1,2,3
            # B,n_curk,len(nb),emb_dim

        output = agg_nb_ft + self.bias
        
        return output, None
    
        if self.input_args and self.curr_k in ['carrier', 'uav']:
            output_all = []
            memory_all = []
            for agent_id in range(self.input_args.num_uav):
                real_id = agent_id if self.curr_k == 'carrier' else agent_id + self.input_args.num_uav
                if memory is not None:
                    memory_in = memory[:, real_id, :, :, :].permute(2, 3, 0,
                                                                    1)  # batch,emb,n_layer,seq ->n_layer,seq,batch,emb
                else:
                    memory_in = None
                # output [1,batch,emb]
                output_new, memory_new = self.temporal_type(output[:, agent_id, :].unsqueeze(0), memory_in)
                output_all.append(output_new.squeeze(0))
                memory_all.append(memory_new)

            output_all = torch.stack(output_all, dim=1)
            memory_all = torch.stack(memory_all, dim=3).permute(2, 3, 4, 0,
                                                                1)  # n_layer,seq,batch,n_agent,emb -> batch,n_agent,emb,n_layer,seq
            return output_all, memory_all

        return output, None
