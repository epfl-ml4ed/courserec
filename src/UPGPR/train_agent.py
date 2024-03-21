from __future__ import absolute_import, division, print_function

import os
import json
import argparse
from collections import namedtuple
import torch
import torch.optim as optim
from easydict import EasyDict as edict

from actor_critic import ActorCritic
from kg_env import BatchKGEnvironment
from utils import *
from validate import *

logger = None


class ACDataLoader(object):
    def __init__(self, uids, batch_size):
        self.uids = np.array(uids)
        self.num_users = len(uids)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self._rand_perm = np.random.permutation(self.num_users)
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            return None
        # Multiple users per batch
        end_idx = min(self._start_idx + self.batch_size, self.num_users)
        batch_idx = self._rand_perm[self._start_idx : end_idx]
        batch_uids = self.uids[batch_idx]
        self._has_next = self._has_next and end_idx < self.num_users
        self._start_idx = end_idx
        return batch_uids.tolist()


def train(args, kg_args):
    env = BatchKGEnvironment(
        args.tmp_dir,
        kg_args,
        args.max_acts,
        max_path_len=args.max_path_len,
        state_history=args.state_history,
        reward_function=args.reward,
        use_pattern=args.use_pattern,
    )
    uids = list(env.kg("user").keys())
    dataloader = ACDataLoader(uids, args.batch_size)
    model = ActorCritic(
        env.state_dim,
        env.act_dim,
        gamma=args.gamma,
        hidden_sizes=args.hidden,
        modified_policy=args.modified_policy,
        embed_size=env.embed_size,
    ).to(args.device)
    logger.info("Parameters:" + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    avg_reward = 0
    # epochs_no_improve = 0
    # max_hit_rate_validation_set = 0
    # patience = args.patience

    total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = (
        [],
        [],
        [],
        [],
        [],
    )
    step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        ### Start epoch ###
        dataloader.reset()
        while dataloader.has_next():
            batch_uids = dataloader.get_batch()
            ### Start batch episodes ###
            batch_state = env.reset(batch_uids)  # numpy array of [bs, state_dim]
            done = False
            while not done:
                batch_act_mask = env.batch_action_mask(
                    dropout=args.act_dropout
                )  # numpy array of size [bs, act_dim]
                batch_act_embeddings = (
                    env.batch_action_embeddings()
                )  # numpy array of size [bs, 2*embed_size, act_dim]
                batch_act_idx = model.select_action(
                    batch_state, batch_act_mask, batch_act_embeddings, args.device
                )  # int
                batch_state, batch_reward, done = env.batch_step(batch_act_idx)
                model.rewards.append(batch_reward)
            ### End of episodes ###

            lr = args.lr * max(
                1e-4, 1.0 - float(step) / (args.epochs * len(uids) / args.batch_size)
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Update policy
            total_rewards.append(np.sum(model.rewards))
            loss, ploss, vloss, eloss = model.update(
                optimizer, args.device, args.ent_weight
            )
            total_losses.append(loss)
            total_plosses.append(ploss)
            total_vlosses.append(vloss)
            total_entropy.append(eloss)
            step += 1

            # Report performance
            if step > 0 and step % 100 == 0:
                avg_reward = np.mean(total_rewards) / args.batch_size
                avg_loss = np.mean(total_losses)
                avg_ploss = np.mean(total_plosses)
                avg_vloss = np.mean(total_vlosses)
                avg_entropy = np.mean(total_entropy)
                (
                    total_losses,
                    total_plosses,
                    total_vlosses,
                    total_entropy,
                    total_rewards,
                ) = ([], [], [], [], [])
                logger.info(
                    "epoch/step={:d}/{:d}".format(epoch, step)
                    + " | loss={:.5f}".format(avg_loss)
                    + " | ploss={:.5f}".format(avg_ploss)
                    + " | vloss={:.5f}".format(avg_vloss)
                    + " | entropy={:.5f}".format(avg_entropy)
                    + " | reward={:.5f}".format(avg_reward)
                )
        ### END of epoch ###

        tmp_policy_file = "{}/tmp_policy_model_epoch_{}.ckpt".format(
            args.log_dir, epoch
        )
        policy_file = "{}/policy_model_epoch_{}.ckpt".format(args.log_dir, epoch)
        torch.save(model.state_dict(), tmp_policy_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/UPGPR/mooc.json", help="Config file."
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    args = config.TRAIN_AGENT

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project_name, name=args.wandb_run_name, config=args
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    args.log_dir = "{}/{}".format(args.tmp_dir, args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + "/train_log.txt")
    logger.info(args)

    set_random_seed(args.seed)
    train(args, config.KG_ARGS)
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
