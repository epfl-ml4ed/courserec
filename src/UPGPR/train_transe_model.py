from __future__ import absolute_import, division, print_function

import os
import argparse
import torch
import torch.optim as optim
import numpy as np

from utils import *
from data_utils import DataLoader
from transe_model import KnowledgeEmbedding
from easydict import EasyDict as edict
import json


logger = None


def train(config):
    args = config.TRAIN_EMBEDS
    kg_args = config.KG_ARGS

    dataset = load_dataset(args.tmp_dir)
    dataloader = DataLoader(
        dataset, args.batch_size, args.use_user_relations, args.use_entity_relations
    )
    interactions_to_train = args.epochs * dataset.interactions.size

    model = KnowledgeEmbedding(dataset, args, kg_args).to(args.device)
    logger.info("Parameters:" + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    smooth_loss = 0.0
    min_val_loss = np.Inf
    patience = args.patience
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        dataloader.reset()
        loss = 0
        while dataloader.has_next():
            # Set learning rate.
            lr = args.lr * max(
                1e-4,
                1.0
                - dataloader.finished_interaction_number / float(interactions_to_train),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Get training batch.
            batch_idxs = dataloader.get_batch()
            batch_idxs = torch.from_numpy(batch_idxs).to(args.device)

            # Train model.
            optimizer.zero_grad()
            train_loss = model(batch_idxs)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            smooth_loss += train_loss.item() / args.steps_per_checkpoint
            loss += train_loss.item()

            steps += 1
            if steps % args.steps_per_checkpoint == 0:
                logger.info(
                    "Epoch: {:02d} | ".format(epoch)
                    + "Interactions: {:d}/{:d} | ".format(
                        dataloader.finished_interaction_number, interactions_to_train
                    )
                    + "Lr: {:.5f} | ".format(lr)
                    + "Smooth loss: {:.5f}".format(smooth_loss)
                )
                smooth_loss = 0.0

        file_name = "{}/transe_model_sd_epoch_{}.ckpt".format(args.log_dir, epoch)
        torch.save(model.state_dict(), file_name)
        if args.use_wandb:
            wandb.log({"Loss": loss})

        if epoch > args.min_epochs:
            if loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = loss
                torch.save(model.state_dict(), file_name)
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                print("Early stopping after {} epochs".format(epoch))
                # set epochs to number of epochs of best model
                args.epochs = int(epoch - patience)
                break

            if epoch == args.epochs:
                print(
                    "Stoppping after {} epochs, best policy after {}".format(
                        epoch, int((epoch - epochs_no_improve))
                    )
                )
                args.epochs = int(epoch - epochs_no_improve)


def extract_embeddings(args, kg_args):
    """Note that last entity embedding is of size [vocab_size+1, d]."""
    model_file = "{}/transe_model_sd_epoch_{}.ckpt".format(args.log_dir, args.epochs)
    print("Load embeddings", model_file)
    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)

    embeds = {}
    embeds_entity = {
        e: state_dict[f"{e}.weight"].cpu().data.numpy()[:-1] for e in kg_args.entities
    }  # Must remove last dummy 'user' with 0 embed.

    embeds_relation = {
        r: (
            state_dict[f"{r}"].cpu().data.numpy()[0],
            state_dict[f"{r}_bias.weight"].cpu().data.numpy(),
        )
        for r in kg_args.item_relation.keys()
    }

    if args.use_user_relations == True:
        embeds_relation.update(
            {
                r: (
                    state_dict[f"{r}"].cpu().data.numpy()[0],
                    state_dict[f"{r}_bias.weight"].cpu().data.numpy(),
                )
                for r in kg_args.user_relation.keys()
            }
        )

    if args.use_entity_relations == True:
        embeds_relation.update(
            {
                r: (
                    state_dict[f"{r}"].cpu().data.numpy()[0],
                    state_dict[f"{r}_bias.weight"].cpu().data.numpy(),
                )
                for r in kg_args.entity_relation.keys()
            }
        )

    embeds_relation.update(
        {
            kg_args.interaction: (
                state_dict[f"{kg_args.interaction}"].cpu().data.numpy()[0],
                state_dict[f"{kg_args.interaction}_bias.weight"].cpu().data.numpy(),
            )
        }
    )

    embeds.update(embeds_entity)
    embeds.update(embeds_relation)

    save_embed(args.tmp_dir, embeds, args.use_wandb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/UPGPR/mooc.json", help="Config file."
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    args = config.TRAIN_EMBEDS

    assert (
        args.min_epochs < args.epochs
    ), "Minimum number of epochs should be lower than total number of epochs."

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
    train(config)
    extract_embeddings(args, config.KG_ARGS)

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
