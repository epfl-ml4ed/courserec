from __future__ import absolute_import, division, print_function

from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

SavedAction = namedtuple("SavedAction", ["log_prob", "value"])


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        gamma=0.99,
        hidden_sizes=[512, 256],
        modified_policy=False,
        embed_size=100,
    ):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        if modified_policy == True:
            self.act_dim = 2 * embed_size
        else:
            self.act_dim = act_dim
        self.gamma = gamma
        self.modified_policy = modified_policy

        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.actor = nn.Linear(hidden_sizes[1], self.act_dim)
        self.critic = nn.Linear(hidden_sizes[1], 1)

        self.saved_actions = []
        self.rewards = []
        self.entropy = []

    def forward(self, inputs):
        if self.modified_policy == True:
            return self.modified_forward(inputs)
        else:
            return self.original_forward(inputs)

    def original_forward(self, inputs):
        state, act_mask, _ = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]
        x = self.l1(state)
        x = F.dropout(F.elu(x), p=0.5)
        out = self.l2(x)
        x = F.dropout(F.elu(out), p=0.5)

        actor_logits = self.actor(x)
        actor_logits[(1 - act_mask).bool()] = -999999.0
        act_probs = F.softmax(actor_logits, dim=-1)  # Tensor of [bs, act_dim]

        state_values = self.critic(x)  # Tensor of [bs, 1]
        return act_probs, state_values

    def modified_forward(self, inputs):
        (
            state,
            act_mask,
            embeddings,
        ) = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]
        x = self.l1(state)
        x = F.dropout(F.elu(x), p=0.5)
        out = self.l2(x)
        x = F.dropout(F.elu(out), p=0.5)
        actor_logits = self.actor(x)
        actor_logits = (
            actor_logits.detach()
            .numpy()
            .reshape(actor_logits.shape[0], 1, actor_logits.shape[1])
        )
        actor_logits = np.matmul(actor_logits, embeddings.detach().numpy())
        actor_logits = torch.from_numpy(np.squeeze(actor_logits, 1))
        actor_logits[(1 - act_mask).bool()] = -999999.0
        act_probs = F.softmax(actor_logits, dim=-1)  # Tensor of [bs, act_dim]

        state_values = self.critic(x)  # Tensor of [bs, 1]
        return act_probs, state_values

    def select_action(self, batch_state, batch_act_mask, batch_act_embeddings, device):
        state = torch.FloatTensor(batch_state).to(device)  # Tensor [bs, state_dim]
        act_mask = torch.ByteTensor(batch_act_mask).to(
            device
        )  # Tensor of [bs, act_dim]
        embeddings = torch.ByteTensor(batch_act_embeddings).to(
            device
        )  # Tensor of [bs, 2*embed_size, act_dim]

        probs, value = self(
            (state, act_mask, embeddings)
        )  # act_probs: [bs, act_dim], state_value: [bs, 1]
        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ], requires_grad=False
        # [CAVEAT] If sampled action is out of action_space, choose the first action in action_space.
        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0

        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())
        return acts.cpu().numpy().tolist()

    def update(self, optimizer, device, ent_weight):
        if len(self.rewards) <= 0:
            del self.rewards[:]
            del self.saved_actions[:]
            del self.entropy[:]
            return 0.0, 0.0, 0.0

        batch_rewards = np.vstack(self.rewards).T  # numpy array of [bs, #steps]
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        num_steps = batch_rewards.shape[1]
        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += (
                self.gamma * batch_rewards[:, num_steps - i]
            )

        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        for i in range(0, num_steps):
            log_prob, value = self.saved_actions[
                i
            ]  # log_prob: Tensor of [bs, ], value: Tensor of [bs, 1]
            advantage = batch_rewards[:, i] - value.squeeze(1)  # Tensor of [bs, ]
            actor_loss += -log_prob * advantage.detach()  # Tensor of [bs, ]
            critic_loss += advantage.pow(2)  # Tensor of [bs, ]
            entropy_loss += -self.entropy[i]  # Tensor of [bs, ]
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        loss = actor_loss + critic_loss + ent_weight * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()
