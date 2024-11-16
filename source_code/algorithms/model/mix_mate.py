import os

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from omegaconf import OmegaConf
from torch.distributions import Normal
from algorithms.model.autoencoder import AutoEncoder
from algorithms.model.ae_models import VAETaskEncoder, ProbabilisticTaskDecoder, TaskEncoder, TaskDecoder,MixingNetwork

class MixedMATE(AutoEncoder):
    def __init__(
        self,
        observation_space,
        action_space,
        config_path,
        config,
        gcn_models,
        **kwargs,
    ):
        cfg = OmegaConf.load(config_path)
        cfg.task_emb_dim = config.task_emb_dim 
        cfg.encoder.type = config.ae_type
                
        self.use_gcn = config.use_gcn
        self.gcn_models = gcn_models
        super(MixedMATE, self).__init__(observation_space, action_space, cfg,config)
        
        self.observation_dims_encoder = copy.deepcopy(self.observation_dims)
        if self.use_gcn:
            for index in range(len(self.observation_dims_encoder)):
                self.observation_dims_encoder[index] += config.n_embd
        
        assert cfg.encoder.type == 'vae'
        self.encoders = [
            VAETaskEncoder(
                obs_dim,
                act_dim,
                1,
                cfg.task_emb_dim,
                cfg.encoder.hiddens,
                cfg.encoder.activation,
            ).to(cfg.device)
            for obs_dim, act_dim in zip(self.observation_dims_encoder, self.action_dims)
        ]
        # agents get mu + log_var as task embedding!
        self.task_emb_dim *= 2

        if cfg.decoder.type == 'probabilistic':
            self.decoder = ProbabilisticTaskDecoder(
                cfg.task_emb_dim,
                sum(self.observation_dims),
                sum(self.action_dims),
                self.n_agents,
                cfg.decoder.hiddens,
                cfg.decoder.activation,
            ).to(cfg.device)
        elif cfg.decoder.type == 'deterministic':
            self.decoder = TaskDecoder(
                cfg.task_emb_dim,
                sum(self.observation_dims),
                sum(self.action_dims),
                self.n_agents,
                cfg.decoder.hiddens,
                cfg.decoder.activation,
            ).to(cfg.device)
        else:
            raise ValueError(f"Unknown decoder type {cfg.decoder.type} not supported!")

        self.mixing_network = MixingNetwork(
            sum(self.observation_dims),
            self.n_agents,
            cfg.mixing.hiddens,
            cfg.mixing.activation,
        ).to(cfg.device)

        self.params = list(self.decoder.parameters()) + list(self.mixing_network.parameters())
        for encoder in self.encoders:
            self.params += list(encoder.parameters())
        self.optimiser = torch.optim.Adam(self.params, self.lr)

        self.saveables = {}
        self.restore_list = {}
        for i, encoder in enumerate(self.encoders):
            self.saveables[f"encoder_{i+1}"] = encoder.state_dict()
            self.restore_list[f"encoder_{i+1}"] = encoder
        self.saveables[f"mixer"] = self.mixing_network.state_dict()
        self.saveables[f"decoder"] = self.decoder.state_dict()
        self.saveables[f"optimiser"] = self.optimiser.state_dict()
        
        self.restore_list[f"mixer"] = self.mixing_network
        self.restore_list[f"decoder"] = self.decoder
        self.restore_list[f"optimiser"] = self.optimiser

    
    def save(self, path):
        torch.save(self.saveables, path)

    def restore(self, path):
        checkpoint = torch.load(path)
        for k, v in self.restore_list.items():
            v.load_state_dict(checkpoint[k])
    
    def hidden_dims(self):
        """
        Get hidden dimensions for all encoders (potentially single or multiple)
        :return List[int]: dimension of hidden states of all encoders
        """
        return [encoder.rnn_hidden_dim for encoder in self.encoders]
            
    def encode(self, obss, acts, rews, hiddens,no_grads=False,graph_inputs = None):
        """
        Encode task embedding

        :param obss: observation of each agent (num_agents, parallel_envs, obs_space)
        :param acts: action of each agent (num_agents, parallel_envs)
        :param rews: reward of each agent (num_agents, parallel_envs)
        :param hiddens: hiddens of all encoders (num_encoders, parallel_envs, hidden_dim)
        :param no_grads: boolean whether no gradients should be computed
        :return: task embedding for all agents (num_agents) x (parallel_envs, task_emb_dim),
            hiddens for all encoders (num_encoders, parallel_envs, hidden_dim)
        """
        if graph_inputs is None or graph_inputs[0] is None:
            nodes = [None for _ in range(self.n_agents)]
            edges = [None for _ in range(self.n_agents)]
        else:
            nodes,edges = graph_inputs
        task_embs = []
        hidden_list = []
        rews = rews.unsqueeze(-1)
        for obs, act, rew, hidden, node,edge, encoder, act_dim, gcn_model in zip(obss, acts, rews, hiddens,nodes,edges,self.encoders, self.action_dims, self.gcn_models):
            if self.use_gcn:
                gcn_embs = gcn_model.forward([node,edge]) 
                obs = torch.cat([obs,gcn_embs],dim=-1)
            act_onehot = torch.nn.functional.one_hot(act.long(), act_dim).float()
            if no_grads:
                with torch.no_grad():
                    task_emb, _, _, _, hiddens = encoder(obs, act_onehot, rew, hidden)
                    #_, _, _, task_emb, hiddens = encoder(obs, act_onehot, rew, hidden)
            else:
                task_emb, _, _, _, hiddens = encoder(obs, act_onehot, rew, hidden)
                #_, _, _, task_emb, hiddens = encoder(obs, act_onehot, rew, hidden)
            task_embs.append(task_emb)
            hidden_list.append(hidden)

        return task_embs, hidden_list

    def zero_grad(self):
        self.optimiser.zero_grad()
    
    def update(self, obss, acts, hiddens, rews, next_obss, done_mask, graph_inputs = None):
        """
        Update encoder and decoder
        :param obss: observations for each agent (n_agents) x (n_step, parallel_envs, obs_space)
        :param acts: actions for each agent (n_agents) x (n_step, parallel_envs, 1)
        :param hiddens: hidden states for each encoder (num_encoders) x (n_step, parallel_envs, hidden_dim)
        :param rews: rewards for each agent (n_agents) x (n_step, parallel_envs, 1)
        :param next_obss: observations for each agent (n_agents) x (n_step, parallel_envs, obs_space)
        :param done_mask: batch of done masks (joint for all agents) (n_step, parallel_envs)
        :return: loss dictionary
        """
        if graph_inputs is None or graph_inputs[0] is None:
            nodes = [None for _ in range(self.n_agents)]
            edges = [None for _ in range(self.n_agents)]
        else:
            nodes,edges = graph_inputs
        mask = ~done_mask[0].squeeze(-1)
        act_onehots = [
            torch.nn.functional.one_hot(act.squeeze(-1).long(), act_dim).float()
            for act, act_dim in zip(acts, self.action_dims)
        ]
        #act_onehots = torch.nn.functional.one_hot(acts.squeeze(-1).long(), self.action_dims[0])

        #mask out entries where episode has terminates (do not make predictions across episodes)
        joint_obs = torch.concat(obss, dim=-1)[mask]
        joint_act_onehots = torch.concat(act_onehots, dim=-1)[mask]
        joint_rews = torch.concat(rews, dim=-1)[mask]
        joint_next_obss = torch.concat(next_obss, dim=-1)[mask]
        #mask = done_mask == 1.0


        batch_size = joint_obs.shape[0]

        task_embs = []
        kl_loss = 0
        mixing_dist = Categorical(F.softmax(self.mixing_network(joint_obs), dim=-1))
        mixing_samples = mixing_dist.sample().expand(1, self.task_emb_dim//2, batch_size).swapaxes(0, -1)
        
        for i, (obs, act_onehot, rew, hidden,node,edge, encoder,gcn_model) in enumerate(zip(obss, act_onehots, rews, hiddens,nodes,edges, self.encoders,self.gcn_models)):
            # mask out entries where episode has terminates (do not make predictions across episodes)
            obs = obs[mask]
            act_onehot = act_onehot[mask]
            rew = rew[mask]
            hidden = hidden[mask]
          
            if self.use_gcn:
                node = node[mask]
                edge = edge[mask]
                gcn_embs = gcn_model.forward([node,edge])
                obs = torch.cat([obs,gcn_embs],dim=-1)

            # compute task embedding
            _, mu, log_var, z, _ = encoder(obs, act_onehot, rew, hidden)

            kl_loss += -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            task_embs.append(z)

        # select task embeddings of agents based on mixing samples
        task_embs = torch.stack(task_embs, dim=-1).gather(-1, mixing_samples).squeeze(-1)

        if self.decoder_type == 'probabilistic':
             # get prediction distributions
            joint_pred_obss_mu, joint_pred_obss_logstd, joint_pred_rews_mu, joint_pred_rews_logstd = self.decoder(task_embs, joint_obs, joint_act_onehots)
            obss_dist = Normal(joint_pred_obss_mu, joint_pred_obss_logstd.exp())
            rews_dist = Normal(joint_pred_rews_mu, joint_pred_rews_logstd.exp())

            # get log probabilities of actually encountered obs and reward
            # under predicted distributions
            obss_log_probs = obss_dist.log_prob(joint_next_obss)
            rews_log_probs = rews_dist.log_prob(joint_rews)

            # compute prediction losses as mean negative log probability
            obs_loss = -obss_log_probs.mean()
            rew_loss = -rews_log_probs.mean()
        else:
            # compute reconstruction loss as MSE of deterministic prediction
            joint_pred_obss, joint_pred_rews = self.decoder(task_embs, joint_obs, joint_act_onehots)
            obs_loss = (joint_next_obss - joint_pred_obss).pow(2).mean()
            rew_loss = (joint_rews - joint_pred_rews).pow(2).mean()

        # compute total loss
        loss = self.obs_loss_coef * obs_loss + self.rew_loss_coef * rew_loss + self.kl_loss_coef * kl_loss
        loss.backward()

        # gradient norm
        if self.max_grad_norm is not None and self.max_grad_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)

        self.optimiser.step()

        loss_dict = {
            "AE/obs_loss": obs_loss.item(),
            "AE/rew_loss": rew_loss.item(),
            "AE/kl_loss": kl_loss.item(),
            "AE/loss": loss.item(),
        }

        return loss_dict
