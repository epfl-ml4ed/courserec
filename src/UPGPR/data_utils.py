from __future__ import absolute_import, division, print_function

import numpy as np
from easydict import EasyDict as edict
import random
from utils import *

class Dataset(object):
    def __init__(self, data_dir, data_args, set_name="train", word_sampling_rate=1e-4):
        self.data_dir = data_dir
        if not self.data_dir.endswith("/"):
            self.data_dir += "/"
        self.interactions = set_name + ".txt"
        self.data_args = data_args
        self.load_entities()
        self.load_item_relations()
        if self.data_args.get("user_relation", None):
            self.load_learner_relations()
        if self.data_args.get("entity_relation", None):
            self.load_entity_relations()
        self.load_interactions()

    def _load_file(self, filename):
        with open(self.data_dir + filename, "r") as f:
            return [line.strip() for line in f]

    def load_entities(self):
        entity_files = edict(self.data_args.entity_files)
        for name in entity_files:
            vocab = self._load_file(entity_files[name])
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab)))
            print("Load", name, "of size", len(vocab))

    def load_item_relations(self):
        item_relation = edict(self.data_args.item_relation)
        for name in item_relation:
            relation = edict(
                data=[],
                et_vocab=getattr(self, item_relation[name][1]).vocab,
                et_distrib=np.zeros(getattr(self, item_relation[name][1]).vocab_size),
            )
            for line in self._load_file(item_relation[name][0]):
                knowledge = []
                for x in line.split(" "):
                    if len(x) > 0:
                        x = int(x)
                        knowledge.append(x)
                        relation.et_distrib[x] += 1
                relation.data.append(knowledge)
            setattr(self, name, relation)
            print("Load", name, "of size", len(relation.data))

    def load_learner_relations(self):
        learner_relations = edict(self.data_args.user_relation)
        for name in learner_relations:
            relation = edict(
                data=[],
                et_vocab=getattr(self, learner_relations[name][1]).vocab,
                et_distrib=np.zeros(
                    getattr(self, learner_relations[name][1]).vocab_size
                ),
            )
            for line in self._load_file(learner_relations[name][0]):
                knowledge = []
                for x in line.split(" "):
                    if len(x) > 0:
                        x = int(x)
                        knowledge.append(x)
                        relation.et_distrib[x] += 1
                relation.data.append(knowledge)
            setattr(self, name, relation)
            print("Load", name, "of size", len(relation.data))
    
    def load_entity_relations(self):
        entity_relations = edict(self.data_args.entity_relation)
        for name in entity_relations:
            relation = edict(
                data=[],
                eh_type=entity_relations[name][1],
                et_type=entity_relations[name][2],
                eh_vocab=getattr(self, entity_relations[name][1]).vocab,
                et_vocab=getattr(self, entity_relations[name][2]).vocab,
                et_distrib=np.zeros(
                    getattr(self, entity_relations[name][2]).vocab_size
                ),
            )
            for line in self._load_file(entity_relations[name][0]):
                knowledge = []
                for x in line.split(" "):
                    if len(x) > 0:
                        x = int(x)
                        knowledge.append(x)
                        relation.et_distrib[x] += 1
                relation.data.append(knowledge)
            setattr(self, name, relation)
            print("Load", name, "of size", len(relation.data))

    def load_interactions(self):
        interaction_data = []
        item_distrib = np.zeros(self.item.vocab_size)
        for line in self._load_file(self.interactions):
            arr = line.split(" ")
            user_idx = int(arr[0])
            item_idx = int(arr[1])
            interaction_data.append((user_idx, item_idx))
            item_distrib[item_idx] += 1

        self.interactions = edict(
            data=interaction_data,
            size=len(interaction_data),
            item_distrib=item_distrib,
            item_uniform_distrib=np.ones(self.item.vocab_size),
            interaction_distrib=np.ones(len(interaction_data)),
        )
        print("Load interactions of size", self.interactions.size)


class DataLoader(object):
    """This class acts as the dataloader for training knowledge graph embeddings."""

    def __init__(self, dataset, batch_size, use_user_relations=False, use_entity_relations=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.interactions_size = self.dataset.interactions.size
        self.item_relations = get_ordered_item_relations(
            self.dataset.data_args
        )  # ['item_school', 'item_teacher', 'item_concept']
        self.item_relations_tails = get_ordered_item_relations_et(self.dataset.data_args)
        self.user_relations = get_user_relations(self.dataset.data_args)
        self.entity_relations = get_entity_relations(self.dataset.data_args)

        self.finished_interaction_number = 0
        self.use_user_relations = use_user_relations
        self.use_entity_relations = use_entity_relations
        self.reset()

    def reset(self):
        # Shuffle reviews order
        self.interaction_seq = np.random.permutation(self.interactions_size)
        self.cur_interaction_i = 0
        self.cur_keyword_i = 0
        self._has_next = True

    def get_batch(self):
        """Return a matrix of [batch_size x X], where each row contains
        (user_id, item_id, ...item_features, ...user_features, ...entity_features).
        """
        batch = []
        interaction_idx = self.interaction_seq[self.cur_interaction_i]
        user_idx, item_idx = self.dataset.interactions.data[interaction_idx]
        item_features = {}

        item_knowledge = {
            cr: getattr(self.dataset, cr).data[item_idx] for cr in self.item_relations
        }

        if self.use_user_relations == True:
            learner_info = {
                lr: getattr(self.dataset, lr).data[user_idx]
                for lr in self.user_relations
            }

        while len(batch) < self.batch_size:
            data = [user_idx, item_idx]
            for ir, et in zip(self.item_relations, self.item_relations_tails):
                if len(item_knowledge[ir]) <= 0:
                    data.append(-1)
                else:
                    et_id = random.choice(item_knowledge[ir])
                    data.append(et_id)
                    item_features[et] = et_id

            if self.use_user_relations == True:
                for lr in self.user_relations:
                    if len(learner_info[lr]) <= 0:
                        data.append(-1)
                    else:
                        data.append(random.choice(learner_info[lr]))

            if self.use_entity_relations == True:
                for er in self.entity_relations:
                    er_info = getattr(self.dataset, er)
                    erh= er_info.eh_type
                    erh_id = item_features[erh]
                    ert_data = er_info.data

                    if len(ert_data[erh_id]) <= 0:
                        data.append(-1)
                    else:
                        data.append(random.choice(ert_data[erh_id]))

            batch.append(data)

            # 2) Move to next interaction
            self.finished_interaction_number += 1
            self.cur_interaction_i += 1
            if self.cur_interaction_i >= self.interactions_size:
                self._has_next = False
                break
            interaction_idx = self.interaction_seq[self.cur_interaction_i]
            user_idx, item_idx = self.dataset.interactions.data[interaction_idx]
            item_knowledge = {
                cr: getattr(self.dataset, cr).data[item_idx]
                for cr in self.item_relations
            }
        return np.array(batch)

    def has_next(self):
        """Has next batch."""
        return self._has_next
