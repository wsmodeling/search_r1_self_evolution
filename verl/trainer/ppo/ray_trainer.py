# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import re
import json
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import re
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # Add per-token reward statistics
    token_level_rewards = batch.batch['token_level_rewards']
    valid_token_rewards = torch.masked_select(token_level_rewards, response_mask)
    # if len(valid_token_rewards) > 0:
    #     metrics['critic/token_rewards/mean'] = torch.mean(valid_token_rewards).detach().item()
    #     metrics['critic/token_rewards/max'] = torch.max(valid_token_rewards).detach().item()
    #     metrics['critic/token_rewards/min'] = torch.min(valid_token_rewards).detach().item()
    #     metrics['critic/token_rewards/std'] = torch.std(valid_token_rewards).detach().item()

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())


    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_logger()
        self._init_memory_db()
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _init_memory_db(self):
        """Initialize memory database for storing low-reward and bad responses"""
        # Import memory database
        try:
            from memory_db.train.stage_rl.Memory_db.database import JSONDatabase

            # Get memory_db config from trainer config, with defaults
            memory_db_config = self.config.get('memory_db', {})
            self.use_memory_db = memory_db_config.get('enable', False)

            if not self.use_memory_db:
                print("[Memory DB] Memory database is disabled in config")
                self.memory_db = None
                return

            # Initialize database
            db_path = memory_db_config.get('db_path', './memory_db/responses.json')
            self.memory_db = JSONDatabase(db_path)

            # Memory DB parameters
            self.memory_db_low_reward_threshold = memory_db_config.get('low_reward_threshold', 0.3)
            self.memory_db_retrieval_ratio = memory_db_config.get('retrieval_ratio', 0.2)  # 20% of batch
            self.memory_db_min_score = memory_db_config.get('min_retrieval_score', -1.0)
            self.memory_db_max_score = memory_db_config.get('max_retrieval_score', 0.5)

            print(f"[Memory DB] Initialized at: {db_path}")
            print(f"[Memory DB] Low reward threshold: {self.memory_db_low_reward_threshold}")
            print(f"[Memory DB] Retrieval ratio: {self.memory_db_retrieval_ratio}")
            print(f"[Memory DB] Score range for retrieval: [{self.memory_db_min_score}, {self.memory_db_max_score}]")

        except ImportError as e:
            print(f"[Memory DB] Warning: Could not import memory_db module: {e}")
            print("[Memory DB] Memory database features will be disabled")
            self.use_memory_db = False
            self.memory_db = None

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        reward_tensor_lst = []
        data_source_lst = []

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            enable_revision=self.config.enable_revision,
            enable_transfer_learning=self.config.enable_transfer_learning,
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
        )

        print(f"\n{'='*80}")
        print(f"Validate(self) - start validation")
        print(f"{'='*80}")

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        else:
            for batch_dict in self.val_dataloader:
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                with _timer('step', timing_raw):
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop_self_evolve(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )
                    
                    test_batch = test_batch.union(final_gen_batch_output)
                    
                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_tensor = self.val_reward_fn(test_batch)

                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = 0
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            enable_revision=self.config.enable_revision,
            enable_transfer_learning=self.config.enable_transfer_learning,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            n_agent=self.config.actor_rollout_ref.rollout.n_agent,
        )

        print(f"\n{'='*80}")
        print(f"fit(self) - start training")
        print(f"{'='*80}")

        # start training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                print(f"\n{'='*100}")
                print(f'Training Batch: epoch {epoch}, batch {batch_idx}, global step {self.global_steps}')
                print(f"\n{'='*100}")

                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                # Preserve non_tensor_batch for revision (to access ground truth)
                gen_batch.non_tensor_batch = batch.non_tensor_batch

                ####################
                # original code here

                with _timer('step', timing_raw):
                    if not self.config.do_search:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                    else:
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        # # Print prompts to understand structure
                        # print(f"\n{'='*100}")
                        # print(f"PROMPT STRUCTURE ANALYSIS (Before run_llm_loop_self_evolve)")
                        # print(f"{'='*100}")
                        # print(f"Batch size: {gen_batch.batch['input_ids'].shape[0]}")
                        # print(f"Full input_ids shape: {gen_batch.batch['input_ids'].shape}")
                        # print(f"Truncated input_ids shape (first_input_ids): {first_input_ids.shape}")
                        # print(f"max_start_length: {gen_config.max_start_length}")

                        # # Decode and print first 2 samples to see prompt structure
                        # num_samples_to_show = min(2, gen_batch.batch['input_ids'].shape[0])
                        # for i in range(num_samples_to_show):
                        #     print(f"\n{'-'*100}")
                        #     print(f"Sample {i}:")
                        #     print(f"{'-'*100}")

                        #     # Decode full prompt
                        #     full_prompt = self.tokenizer.decode(gen_batch.batch['input_ids'][i], skip_special_tokens=False)
                        #     print(f"\nFull prompt (input_ids):")
                        #     print(f"{full_prompt}")

                        #     # Decode truncated prompt
                        #     truncated_prompt = self.tokenizer.decode(first_input_ids[i], skip_special_tokens=False)
                        #     print(f"\nTruncated prompt (first_input_ids, last {gen_config.max_start_length} tokens):")
                        #     print(f"{truncated_prompt}")

                        # print(f"\n{'='*100}")
                        # print(f"Analysis:")
                        # print(f"- Each sample in gen_batch contains the full prompt (with system prompt if included)")
                        # print(f"- first_input_ids takes only the last {gen_config.max_start_length} tokens")
                        # print(f"- If system prompt is present, check if it appears in BOTH full and truncated versions")
                        # print(f"- If system prompt only appears in full version, it will be lost after truncation")
                        # print(f"{'='*100}\n")

                        with _timer('gen', timing_raw):
                            generation_manager.timing_raw = timing_raw
                            # Set epoch and batch index for readable printing
                            generation_manager.current_epoch = epoch
                            generation_manager.current_batch_idx = batch_idx
                            final_gen_batch_output = generation_manager.run_llm_loop_self_evolve(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        for key in final_gen_batch_output.batch.keys():
                            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        print(f"\n{'='*80}")
                        print(f"STEP {self.global_steps} - compute_log_prob")
                        print(f"{'='*80}")
                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output)

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        #                                         dtype=object)
                        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()

                        # repeat to align with repeated responses in rollout
                        # n = 1, so we comment the line below
                        # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                        # If transfer learning is enabled and outputs are merged, we need special handling
                        if final_gen_batch_output.meta_info.get('is_merged_output', False):
                            # Transfer learning is enabled
                            # batch currently has: N × n_agent samples (already repeated at line 721)
                            # Generation output has: N × n_agent (revision) + N (transfer) = N × (n_agent + 1)
                            # Structure: [R0_a0, R0_a1, T0, R1_a0, R1_a1, T1, ...]

                            # We need to create: [P0, P0, P0, P1, P1, P1, ...] to match generation output
                            # where each original prompt appears (n_agent + 1) times

                            # Current batch: [P0, P0, P1, P1, ...] (N × n_agent)
                            # We need to rebuild it from scratch to get proper grouping

                            n_agent = self.config.actor_rollout_ref.rollout.n_agent
                            current_batch_size = len(batch.batch)
                            num_unique_prompts = current_batch_size // n_agent  # N

                            print(f"Transfer learning: current batch size {current_batch_size}, n_agent {n_agent}, unique prompts {num_unique_prompts}")
                            print(f"Creating batch structure: each prompt repeated {n_agent + 1} times")

                            # Build the new batch by repeating each unique prompt (n_agent + 1) times
                            new_indices = []
                            for prompt_idx in range(num_unique_prompts):
                                # Get the index of the first occurrence of this prompt in current batch
                                original_idx = prompt_idx * n_agent
                                # Repeat this index (n_agent + 1) times
                                new_indices.extend([original_idx] * (n_agent + 1))

                            new_indices_tensor = torch.tensor(new_indices, dtype=torch.long)
                            batch.reorder(new_indices_tensor)

                            print(f"New batch size after restructuring: {len(batch.batch)}")
                        else:
                            # No transfer learning, batch size already matches (both N × n_agent)
                            # batch was already repeated at line 721, and generation output is also N × n_agent
                            # No additional repeat needed!
                            print(f"No transfer learning: batch size {len(batch.batch)} already matches generation output {len(final_gen_batch_output.batch)}")

                        # Augment batch with bad responses from memory database
                        batch, final_gen_batch_output = self._augment_batch_with_memory_db_responses(batch, final_gen_batch_output)

                        batch = batch.union(final_gen_batch_output)

                        # Verify prompt-response matching if enabled
                        if self.config.get('enable_prompt_response_verification', False):
                            self._verify_prompt_response_matching(batch)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    # For debugging: disable batch balancing to maintain order between generation and scoring
                    if not self.config.get('disable_batch_balancing', False):
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    for key in batch.batch.keys():
                        if key != 'old_log_probs':
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        print(f"\n{'='*80}")
                        print(f"STEP {self.global_steps} - compute_ref_log_prob")
                        print(f"{'='*80}")
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:

                        print(f"\n{'='*80}")
                        print(f"STEP {self.global_steps} - compute_values")
                        print(f"{'='*80}")
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        print(f"\n{'='*80}")
                        print(f"STEP {self.global_steps} - compute_rm_score")
                        print(f"{'='*80}")

                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                            # Debug: Print scores for first N samples
                            if 'rm_scores' in reward_tensor.batch:
                                rm_scores = reward_tensor.batch['rm_scores']
                                n_samples_to_print = min(5, rm_scores.shape[0])  # Print first 5 samples
                                print(f"\n[DEBUG] RM Scores for first {n_samples_to_print} samples:")
                                print(f"{'='*80}")
                                for i in range(n_samples_to_print):
                                    score = rm_scores[i].item() if rm_scores[i].numel() == 1 else rm_scores[i].mean().item()
                                    print(f"  Sample {i}: RM Score = {score:.4f}")
                                print(f"{'='*80}\n")

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # Debug: Print final scores for first N samples after reward_fn
                        n_samples_to_print = min(5, reward_tensor.shape[0])
                        print(f"\n[DEBUG] Final Token-Level Scores for first {n_samples_to_print} samples (after reward_fn):")
                        print(f"{'='*80}")

                        # Get prompts and responses for context
                        prompts = batch.batch.get('prompts', None)
                        responses = batch.batch.get('responses', None)

                        for i in range(n_samples_to_print):
                            print(f"\n--- Sample {i} ---")

                            # Print prompt if available
                            if prompts is not None:
                                prompt_ids = prompts[i]
                                # Decode prompt
                                from transformers import AutoTokenizer
                                if not hasattr(self, '_debug_tokenizer'):
                                    # Cache tokenizer for efficiency
                                    model_path = self.config.actor_rollout_ref.model.get('path', self.config.actor_rollout_ref.model.get('pretrained_model', 'Qwen/Qwen2.5-7B'))
                                    self._debug_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

                                prompt_text = self._debug_tokenizer.decode(prompt_ids, skip_special_tokens=True)
                                # Truncate long prompts
                                if len(prompt_text) > 200:
                                    prompt_display = prompt_text[:200] + "..."
                                else:
                                    prompt_display = prompt_text
                                print(f"Prompt: {prompt_display}")

                            # Print response if available
                            if responses is not None:
                                response_ids = responses[i]
                                response_text = self._debug_tokenizer.decode(response_ids, skip_special_tokens=True)
                                # Truncate long responses
                                if len(response_text) > 200:
                                    response_display = response_text[:200] + "..."
                                else:
                                    response_display = response_text
                                print(f"Response: {response_display}")

                            # Get the score for this sample (could be token-level or sequence-level)
                            if reward_tensor.dim() == 1:
                                score = reward_tensor[i].item()
                                print(f"Score: {score:.4f}")
                            else:
                                # Token-level scores - show mean and range
                                sample_scores = reward_tensor[i]
                                mean_score = sample_scores.mean().item()
                                min_score = sample_scores.min().item()
                                max_score = sample_scores.max().item()
                                print(f"Score: Mean = {mean_score:.4f}, Min = {min_score:.4f}, Max = {max_score:.4f}")

                        print(f"\n{'='*80}\n")

                        # add token_level_scores metrics below
                        # metrics.update(
                        #     {
                        #         'token_level_scores/mean': torch.mean(reward_tensor).detach().item()
                        #     }
                        # )

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # Save low-reward responses to memory database
                        self._save_low_reward_responses_to_memory_db(batch, self.global_steps)

                        print(f"\n{'='*80}")
                        print(f"STEP {self.global_steps} - compute_advantage")
                        print(f"{'='*80}")
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  # num_repeat is not used in compute_advantage?
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n_agent)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                print(f"\n{'='*80}")
                print(f"STEP {self.global_steps} - compute_data_metrics/compute_timing_metrics")
                print(f"{'='*80}")      
                # collect metrics
                data_metrics = compute_data_metrics(batch=batch, use_critic=self.use_critic)
                metrics.update(data_metrics)
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # Debug: Print some key metrics to console
                # print(f"\n{'='*80}")
                # print(f"STEP {self.global_steps} - Training Metrics Summary")
                # print(f"{'='*80}")
                # print(f"Critic metrics enabled: {self.use_critic}")
                # if 'critic/rewards/mean' in metrics:
                #     print(f"  critic/rewards/mean: {metrics['critic/rewards/mean']:.4f}")
                # if 'critic/score/mean' in metrics:
                #     print(f"  critic/score/mean: {metrics['critic/score/mean']:.4f}")
                # if 'critic/advantages/mean' in metrics:
                #     print(f"  critic/advantages/mean: {metrics['critic/advantages/mean']:.4f}")
                # if 'env/number_of_actions/mean' in metrics:
                #     print(f"  env/number_of_actions/mean: {metrics['env/number_of_actions/mean']:.2f}")
                # print(f"Total metrics being logged: {len(metrics)}")
                # print(f"Metric names: {list(metrics.keys())[:10]}...")  # Print first 10
                # print(f"{'='*80}\n")

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
    
    def _verify_prompt_response_matching(self, batch):
        """
        Print prompt-response matching verification for debugging.
        Shows first 3 unique prompts with their responses to verify correct matching.
        """
        print(f"\n{'='*100}")
        print(f"PROMPT-RESPONSE MATCHING VERIFICATION (Step {self.global_steps})")
        print(f"{'='*100}")
        print(f"Showing first 3 unique prompts with their responses...\n")

        # Get unique UIDs
        if 'uid' in batch.non_tensor_batch:
            unique_uids = []
            seen_uids = set()
            for uid in batch.non_tensor_batch['uid']:
                if uid not in seen_uids:
                    unique_uids.append(uid)
                    seen_uids.add(uid)
                    if len(unique_uids) >= 3:  # Only show first 3 unique prompts
                        break

            # For each unique UID, show all its responses
            for uid_to_show in unique_uids:
                # Find all indices with this UID
                uid_indices = [i for i, uid in enumerate(batch.non_tensor_batch['uid']) if uid == uid_to_show]

                # Get the prompt (should be same for all indices with this UID)
                prompt_idx = uid_indices[0]
                prompt_ids = batch.batch['prompts'][prompt_idx]
                prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)

                # Truncate prompt for display (show first 2000 chars)
                prompt_display = prompt_text[:2000] + "..." if len(prompt_text) > 2000 else prompt_text

                print(f"\n{'-'*100}")
                print(f"UID: {uid_to_show}")
                print(f"Prompt: {prompt_display}")
                print(f"Number of responses: {len(uid_indices)}")
                print(f"{'-'*100}")

                # Categorize responses
                revision_responses = []
                transfer_responses = []

                for idx in uid_indices:
                    response_ids = batch.batch['responses'][idx]
                    response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                    # Check if it's revision or transfer based on position
                    # With transfer learning enabled, the structure is group-level concatenation:
                    # [R0, R1, ..., R(n_agent-1), T, R0, R1, ..., R(n_agent-1), T, ...]
                    # Each group has (n_agent + 1) responses: n_agent revisions + 1 transfer
                    # So: indices 0 to n_agent-1 are revision, index n_agent is transfer, then repeat
                    local_idx = uid_indices.index(idx)
                    n_agent = self.config.actor_rollout_ref.rollout.n_agent

                    # Determine response type based on meta_info
                    if hasattr(batch, 'meta_info') and batch.meta_info.get('is_merged_output', False):
                        # Revision + Transfer learning enabled
                        # Position within each (n_agent + 1) group
                        position_in_group = local_idx % (n_agent + 1)
                        response_type = "Revision" if position_in_group < n_agent else "Transfer"
                    elif hasattr(batch, 'meta_info') and batch.meta_info.get('generation_type') == 'revision':
                        # Only revision enabled (no transfer learning)
                        response_type = "Revision"
                    else:
                        # No revision, regular generation
                        response_type = "Regular"

                    # Truncate response for display (show first 2000 chars)
                    response_display = response_text[:2000] + "..." if len(response_text) > 2000 else response_text

                    if response_type == "Revision":
                        revision_responses.append(response_display)
                    elif response_type == "Transfer":
                        transfer_responses.append(response_display)
                    else:  # Regular
                        revision_responses.append(response_display)  # Treat regular as revision for display

                # Print categorized responses
                if revision_responses:
                    print(f"\n{'='*100}")
                    print(f"\nRevision/Regular Responses ({len(revision_responses)}):")
                    for i, resp in enumerate(revision_responses[:2], 1):  # Show first 2
                        print(f"\n{'='*100}")
                        print(f"  [{i}] {resp}")
                    if len(revision_responses) > 2:
                        print(f"  ... and {len(revision_responses) - 2} more")

                if transfer_responses:
                    print(f"\n{'='*100}")
                    print(f"\nTransfer Learning Responses ({len(transfer_responses)}):")
                    for i, resp in enumerate(transfer_responses[:2], 1):  # Show first 2
                        print(f"\n{'='*100}")
                        print(f"  [{i}] {resp}")
                    if len(transfer_responses) > 2:
                        print(f"  ... and {len(transfer_responses) - 2} more")

        print(f"\n{'='*100}")
        print(f"END OF PROMPT-RESPONSE MATCHING VERIFICATION")
        print(f"{'='*100}\n")

    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]

        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })

        return batch, metrics

    def _save_low_reward_responses_to_memory_db(self, batch, step):
        """
        Save responses with low rewards to memory database.

        Args:
            batch: DataProto containing responses and token_level_scores
            step: Current training step
        """
        if not self.use_memory_db or self.memory_db is None:
            return

        try:
            # Extract necessary data
            token_level_scores = batch.batch['token_level_scores']  # (batch_size, response_length)
            responses = batch.batch['responses']  # (batch_size, response_length)
            prompts = batch.batch['prompts']  # (batch_size, prompt_length)
            response_length = responses.shape[-1]
            attention_mask = batch.batch['attention_mask']
            response_mask = attention_mask[:, -response_length:]

            # Compute sequence-level scores (sum over tokens)
            sequence_scores = (token_level_scores * response_mask).sum(dim=-1)  # (batch_size,)

            # Identify low-reward responses
            low_reward_mask = sequence_scores < self.memory_db_low_reward_threshold
            low_reward_indices = torch.where(low_reward_mask)[0].cpu().numpy()

            if len(low_reward_indices) == 0:
                return

            print(f"\n[Memory DB] Saving {len(low_reward_indices)} low-reward responses (threshold: {self.memory_db_low_reward_threshold})")

            # Save each low-reward response
            for idx in low_reward_indices:
                idx = int(idx)

                # Decode prompt and response
                prompt_ids = prompts[idx].cpu().numpy()
                response_ids = responses[idx].cpu().numpy()

                # Remove padding tokens
                prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                # Get score
                score = sequence_scores[idx].item()

                # Get UID if available
                uid = batch.non_tensor_batch.get('uid', [None] * len(batch.batch))[idx]
                if uid is None:
                    uid = f"step{step}_idx{idx}"

                # Prepare training data entry
                training_data = {
                    'id': str(uid),
                    'problem': prompt_text,
                    'answer': '',  # We don't have ground truth here
                    'round': step,
                    'image': [],
                    'experience': []
                }

                # Insert or update training data
                training_id = self.memory_db.insert_training_data(training_data)

                # Add LLM response with score
                scores = {
                    'accuracy': score,  # Use sequence score as accuracy
                    'format': 1.0,
                    'reason': 1.0,
                    'length': 1.0
                }

                self.memory_db.add_llm_response(
                    training_data_id=training_id,
                    response=response_text,
                    scores=scores,
                    reflexion=f"Low-reward response from step {step}"
                )

            print(f"[Memory DB] Successfully saved {len(low_reward_indices)} responses")

        except Exception as e:
            print(f"[Memory DB] Error saving low-reward responses: {e}")
            import traceback
            traceback.print_exc()

    def _retrieve_bad_responses_from_memory_db(self, current_batch_size):
        """
        Retrieve bad responses from memory database to augment training.

        Args:
            current_batch_size: Current batch size

        Returns:
            List of tuples (prompt_text, response_text, score) or None if no data
        """
        if not self.use_memory_db or self.memory_db is None:
            return None

        try:
            # Calculate how many bad responses to retrieve
            num_to_retrieve = int(current_batch_size * self.memory_db_retrieval_ratio)

            if num_to_retrieve == 0:
                return None

            # Get all training data
            all_data = self.memory_db.get_all_training_data()

            if not all_data:
                print("[Memory DB] No data available in database yet")
                return None

            # Collect responses with scores in target range
            candidate_responses = []

            for training_id, training_entry in all_data.items():
                llm_responses = training_entry.get('llm_answers_and_score', [])
                problem = training_entry.get('problem', '')

                for response_data in llm_responses:
                    score = response_data.get('accuracy', 0.0)

                    # Filter by score range
                    if self.memory_db_min_score <= score <= self.memory_db_max_score:
                        candidate_responses.append({
                            'prompt': problem,
                            'response': response_data.get('ans', ''),
                            'score': score,
                            'training_id': training_id
                        })

            if not candidate_responses:
                print(f"[Memory DB] No responses found in score range [{self.memory_db_min_score}, {self.memory_db_max_score}]")
                return None

            # Sample responses (prioritize lower scores)
            candidate_responses.sort(key=lambda x: x['score'])  # Sort by score ascending
            selected_responses = candidate_responses[:num_to_retrieve]

            print(f"[Memory DB] Retrieved {len(selected_responses)} bad responses from database")
            print(f"[Memory DB] Score range of retrieved: [{selected_responses[0]['score']:.3f}, {selected_responses[-1]['score']:.3f}]")

            return selected_responses

        except Exception as e:
            print(f"[Memory DB] Error retrieving bad responses: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _augment_batch_with_memory_db_responses(self, batch: DataProto, gen_batch_output: DataProto):
        """
        Augment the current batch with bad responses retrieved from memory database.

        Args:
            batch: Original batch (prompts)
            gen_batch_output: Generated responses batch

        Returns:
            Augmented batch, augmented gen_batch_output
        """
        if not self.use_memory_db or self.memory_db is None:
            return batch, gen_batch_output

        try:
            # Retrieve bad responses
            bad_responses = self._retrieve_bad_responses_from_memory_db(len(batch.batch))

            if bad_responses is None or len(bad_responses) == 0:
                return batch, gen_batch_output

            print(f"\n[Memory DB] Augmenting batch with {len(bad_responses)} bad responses")

            # Tokenize the retrieved bad responses
            augment_prompts = []
            augment_responses = []

            for item in bad_responses:
                # Tokenize prompt and response
                prompt_tokens = self.tokenizer.encode(item['prompt'], add_special_tokens=True)
                response_tokens = self.tokenizer.encode(item['response'], add_special_tokens=False)

                # Truncate if necessary
                max_prompt_len = batch.batch['prompts'].shape[1]
                max_response_len = gen_batch_output.batch['responses'].shape[1]

                if len(prompt_tokens) > max_prompt_len:
                    prompt_tokens = prompt_tokens[:max_prompt_len]
                if len(response_tokens) > max_response_len:
                    response_tokens = response_tokens[:max_response_len]

                # Pad to match dimensions
                prompt_tokens = prompt_tokens + [self.tokenizer.pad_token_id] * (max_prompt_len - len(prompt_tokens))
                response_tokens = response_tokens + [self.tokenizer.pad_token_id] * (max_response_len - len(response_tokens))

                augment_prompts.append(prompt_tokens)
                augment_responses.append(response_tokens)

            # Convert to tensors
            augment_prompts_tensor = torch.tensor(augment_prompts, dtype=torch.long)
            augment_responses_tensor = torch.tensor(augment_responses, dtype=torch.long)

            # Create attention masks
            augment_prompt_mask = (augment_prompts_tensor != self.tokenizer.pad_token_id).long()
            augment_response_mask = (augment_responses_tensor != self.tokenizer.pad_token_id).long()

            # Concatenate with existing batch
            batch.batch['prompts'] = torch.cat([batch.batch['prompts'], augment_prompts_tensor], dim=0)

            # Update batch metadata (non_tensor_batch)
            for key in batch.non_tensor_batch.keys():
                if isinstance(batch.non_tensor_batch[key], np.ndarray):
                    # Pad with placeholder values
                    pad_values = np.array([batch.non_tensor_batch[key][0]] * len(bad_responses))
                    batch.non_tensor_batch[key] = np.concatenate([batch.non_tensor_batch[key], pad_values])

            # Update gen_batch_output
            gen_batch_output.batch['responses'] = torch.cat([gen_batch_output.batch['responses'], augment_responses_tensor], dim=0)

            # Update attention_mask
            if 'attention_mask' in gen_batch_output.batch:
                augment_attention_mask = torch.cat([augment_prompt_mask, augment_response_mask], dim=1)
                gen_batch_output.batch['attention_mask'] = torch.cat([gen_batch_output.batch['attention_mask'], augment_attention_mask], dim=0)

            # Update other fields in gen_batch_output if they exist
            if 'old_log_probs' in gen_batch_output.batch:
                # For old_log_probs, we need to fill with zeros or compute them
                # For simplicity, we'll fill with small negative values (low probability)
                augment_log_probs = torch.full((len(bad_responses), gen_batch_output.batch['old_log_probs'].shape[1]),
                                               -10.0, dtype=gen_batch_output.batch['old_log_probs'].dtype)
                gen_batch_output.batch['old_log_probs'] = torch.cat([gen_batch_output.batch['old_log_probs'], augment_log_probs], dim=0)

            print(f"[Memory DB] Batch augmented. New batch size: {len(batch.batch)}")

            return batch, gen_batch_output

        except Exception as e:
            print(f"[Memory DB] Error augmenting batch: {e}")
            import traceback
            traceback.print_exc()
            return batch, gen_batch_output
