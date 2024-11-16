import os
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.distributions import Normal
from algorithms.model.autoencoder import AutoEncoder
from algorithms.model.ae_models import VAETaskEncoder, ProbabilisticTaskDecoder, TaskEncoder, TaskDecoder
import copy

class CentralisedMATE(AutoEncoder):
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
        super(CentralisedMATE, self).__init__(observation_space, action_space, cfg,config)
        
        self.observation_dims_encoder = copy.deepcopy(self.observation_dims)
        if self.use_gcn:
            for index in range(len(self.observation_dims_encoder)):
                self.observation_dims_encoder[index] += config.n_embd
        if cfg.encoder.type == 'vae':
            self.encoders = VAETaskEncoder(
                sum(self.observation_dims_encoder),
                sum(self.action_dims),
                self.n_agents,
                cfg.task_emb_dim,
                cfg.encoder.hiddens,
                cfg.encoder.activation,
            ).to(cfg.device)
            # agents get mu + log_var as task embedding!
            self.task_emb_dim *= 2
        elif cfg.encoder.type == 'deterministic':
            self.encoders = TaskEncoder(
                sum(self.observation_dims_encoder),
                sum(self.action_dims),
                self.n_agents,
                cfg.task_emb_dim,
                cfg.encoder.hiddens,
                cfg.encoder.activation,
            ).to(cfg.device)
        else:
            raise ValueError(f"Unknown encoder type {cfg.encoder.type} not supported!")

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

        self.optimiser = torch.optim.Adam(list(self.encoders.parameters()) + list(self.decoder.parameters()), self.lr)

        self.saveables = {}
        self.restore_list = {}
        self.saveables[f"encoder"] = self.encoders.state_dict()
        self.saveables[f"decoder"] = self.decoder.state_dict()
        self.saveables[f"optimiser"] = self.optimiser.state_dict()
        
        self.restore_list[f"encoder"] = self.encoders
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
        return [self.encoders.rnn_hidden_dim]
            
    def encode(self, obss, acts, rews, hiddens, no_grads=False,graph_inputs = None):
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

        nb_agents = obss.size()[0]
        rews = rews.unsqueeze(-1)
        obss = [obss[i] for i in range(nb_agents)]
        acts = [acts[i] for i in range(nb_agents)]
        rews = [rews[i] for i in range(nb_agents)]
        hiddens =[hiddens[i] for i in range(nb_agents)]
        
                    
        if self.use_gcn:
            for i in range(nb_agents):
                gcn_embs = self.gcn_models[i].forward([nodes[i],edges[i]]) 
                obss[i] = torch.cat([obss[i],gcn_embs],dim=-1)
     
        act_onehots = [
            torch.nn.functional.one_hot(act.long(), act_dim).float()
            for act, act_dim in zip(acts, self.action_dims)
        ]
        joint_obs = torch.concat(obss, dim=-1)
        acts_onehots = torch.concat(act_onehots, dim=-1)
        joint_rews = torch.concat(rews, dim=-1)
        hiddens = hiddens[0]

        if no_grads:
            with torch.no_grad():
                if self.encoder_type == 'vae':
                    task_emb, _, _, _, hiddens = self.encoders(joint_obs, acts_onehots, joint_rews, hiddens)
                    #_, _, _, task_emb, hiddens = self.encoders(joint_obs, acts_onehots, joint_rews, hiddens)
                else:
                    task_emb, hiddens = self.encoders(joint_obs, acts_onehots, joint_rews, hiddens)
        else:
            if self.encoder_type == 'vae':
                task_emb, _, _, _, hiddens = self.encoders(joint_obs, acts_onehots, joint_rews, hiddens)
                #_, _, _, task_emb, hiddens = self.encoders(joint_obs, acts_onehots, joint_rews, hiddens)
            else:
                task_emb, hiddens = self.encoders(joint_obs, acts_onehots, joint_rews, hiddens)

        return [task_emb] * self.n_agents, [hiddens.detach() for _ in range(nb_agents)]


    def zero_grad(self):
        self.optimiser.zero_grad()

    
    def update(self, obss, acts, hiddens, rews, next_obss, done_mask,graph_inputs = None):
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
        act_onehots = [
            torch.nn.functional.one_hot(act.squeeze(-1).long(), act_dim).float()
            for act, act_dim in zip(acts, self.action_dims)
        ]
        
        if graph_inputs is None or graph_inputs[0] is None:
            nodes = [None for _ in range(self.n_agents)]
            edges = [None for _ in range(self.n_agents)]
        else:
            nodes,edges = graph_inputs
            
        mask = ~done_mask[0].squeeze(-1)
        joint_obs_decoder = torch.concat(obss, dim=-1)[mask]
        nb_agents = len(obss)
        if self.use_gcn:
            for i in range(nb_agents):
                gcn_embs = self.gcn_models[i].forward([nodes[i],edges[i]],ma_dim=True) 
                obss[i] = torch.cat([obss[i],gcn_embs],dim=-1)
        # mask out entries where episode has terminates (do not make predictions across episodes)
        #mask = done_mask == 1.0
   
        joint_obs = torch.concat(obss, dim=-1)[mask]
        joint_act_onehots = torch.concat(act_onehots, dim=-1)[mask]
        joint_rews = torch.concat(rews, dim=-1)[mask]
        joint_next_obss = torch.concat(next_obss, dim=-1)[mask]
        hiddens = hiddens[0][mask]

        # compute task embedding
        if self.encoder_type == 'vae':
            _, mu, log_var, z, _ = self.encoders(joint_obs, joint_act_onehots, joint_rews, hiddens)
            task_emb = z
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        else:
            task_emb, _ = self.encoders(joint_obs, joint_act_onehots, joint_rews, hiddens)

        if self.decoder_type == 'probabilistic':
             # get prediction distributions
            joint_pred_obss_mu, joint_pred_obss_logstd, joint_pred_rews_mu, joint_pred_rews_logstd = self.decoder(task_emb, joint_obs, joint_act_onehots)
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
            joint_pred_obss, joint_pred_rews = self.decoder(task_emb, joint_obs_decoder, joint_act_onehots)
            obs_loss = (joint_next_obss - joint_pred_obss).pow(2).mean()
            rew_loss = (joint_rews - joint_pred_rews).pow(2).mean()

        # compute total loss
        loss = self.obs_loss_coef * obs_loss + self.rew_loss_coef * rew_loss
        if self.encoder_type == 'vae':
            loss += self.kl_loss_coef * kl_loss

        loss.backward()

        # gradient norm
        if self.max_grad_norm is not None and self.max_grad_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(list(self.encoders.parameters()) + list(self.decoder.parameters()), self.max_grad_norm)

        self.optimiser.step()

        loss_dict = {
            "AE/obs_loss": obs_loss.item(),
            "AE/rew_loss": rew_loss.item(),
            "AE/loss": loss.item(),
        }

        if self.encoder_type == 'vae':
            loss_dict["AE/kl_loss"] = kl_loss.item()

        return loss_dict
