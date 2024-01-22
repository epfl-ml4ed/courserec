from __future__ import absolute_import, division, print_function

import sys
import random
import pickle
import logging
import logging.handlers
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
import torch
import wandb


def get_entities(args):
    return list(args.kg_relation.keys())


def get_relations(args, entity_head):
    return list(args.kg_relation[entity_head].keys())


def get_entity_tail(args, entity_head, relation):
    return args.kg_relation[entity_head][relation]


def get_item_relations(args):
    return args.item_relation.keys()


def compute_tfidf_fast(vocab, docs):
    """Compute TFIDF scores for all vocabs.

    Args:
        docs: list of list of integers, e.g. [[0,0,1], [1,2,0,1]]

    Returns:
        sp.csr_matrix, [num_docs, num_vocab]
    """
    # (1) Compute term frequency in each doc.
    data, indices, indptr = [], [], [0]
    for d in docs:
        term_count = {}
        for term_idx in d:
            if term_idx not in term_count:
                term_count[term_idx] = 1
            else:
                term_count[term_idx] += 1
        indices.extend(term_count.keys())
        data.extend(term_count.values())
        indptr.append(len(indices))
    tf = sp.csr_matrix(
        (data, indices, indptr), dtype=int, shape=(len(docs), len(vocab))
    )

    # (2) Compute normalized tfidf for each term/doc.
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(tf)
    return tfidf


def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s]  %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_dataset(path, dataset_obj, use_wandb):
    dataset_file = path + "/dataset.pkl"
    with open(dataset_file, "wb") as f:
        pickle.dump(dataset_obj, f)
    if use_wandb:
        wandb.save(dataset_file)


def load_dataset(path):
    dataset_file = path + "/dataset.pkl"
    dataset_obj = pickle.load(open(dataset_file, "rb"))
    return dataset_obj


def save_labels(path, labels, mode="train", use_wandb=False):
    if mode not in ["train", "test", "validation"]:
        raise Exception("mode should be one of {train, test, validation}.")
    label_file = f"{path}/{mode}_label.pkl"
    with open(label_file, "wb") as f:
        pickle.dump(labels, f)
    if use_wandb:
        wandb.save(label_file)


def load_labels(path, mode="train"):
    if mode not in ["train", "test", "validation"]:
        raise Exception("mode should be one of {train, test, validation}.")
    label_file = f"{path}/{mode}_label.pkl"
    user_products = pickle.load(open(label_file, "rb"))
    return user_products


def save_embed(path, embed, use_wandb=False):
    embed_file = "{}/transe_embed.pkl".format(path)
    pickle.dump(embed, open(embed_file, "wb"))
    if use_wandb:
        wandb.save(embed_file)


def load_embed(path):
    embed_file = "{}/transe_embed.pkl".format(path)
    print("Load embedding:", embed_file)
    embed = pickle.load(open(embed_file, "rb"))
    return embed


def save_kg(path, kg, use_wandb):
    kg_file = path + "/kg.pkl"
    pickle.dump(kg, open(kg_file, "wb"))
    if use_wandb:
        wandb.save(kg_file)


def load_kg(path):
    kg_file = path + "/kg.pkl"
    kg = pickle.load(open(kg_file, "rb"))
    return kg


def create_data_file(data_dir, data, file_name):
    with open(data_dir + "/" + file_name, "w") as f:
        for d in data:
            f.write(d)


def get_ordered_item_relations(args):
    entities = list(filter(lambda x: x not in ["user", "item"], args.entities))
    item_relations = []

    relation_tail = list(map(lambda r: (r[0], r[1][1]), args.item_relation.items()))
    for e in entities:
        for rt in relation_tail:
            if e == rt[1]:
                item_relations.append(rt[0])
                break

    return item_relations


def get_ordered_item_relations_et(args):
    entities = list(filter(lambda x: x not in ["user", "item"], args.entities))
    tail_entities = []

    relation_tail = list(map(lambda r: (r[0], r[1][1]), args.item_relation.items()))
    for e in entities:
        for rt in relation_tail:
            if e == rt[1]:
                tail_entities.append(rt[1])
                break

    return tail_entities


def get_ordered_user_relations_et(args):
    entities = list(filter(lambda x: x not in ["user", "item"], args.entities))
    tail_entities = []

    relation_tail = list(map(lambda r: (r[0], r[1][1]), args.user_relation.items()))
    for e in entities:
        for rt in relation_tail:
            if e == rt[1]:
                tail_entities.append("ur_" + rt[1])
                break

    return tail_entities


def get_ordered_entity_relations_et(args):
    entities = list(filter(lambda x: x not in ["user", "item"], args.entities))
    tail_entities = []

    relation_tail = list(map(lambda r: (r[0], r[1][2]), args.entity_relation.items()))
    for e in entities:
        for rt in relation_tail:
            if e == rt[1]:
                tail_entities.append("er_" + rt[1])
                break

    return tail_entities


def get_user_relations(args):
    if args.get("user_relation", None):
        user_relations = args.user_relation.keys()
        return user_relations


def get_entity_relations(args):
    if args.get("entity_relation", None):
        entity_relation = args.entity_relation.keys()
        return entity_relation


def get_batch_entities(kg_args):
    batch_entities = ["user", "item"]
    batch_entities.extend(get_ordered_item_relations_et(kg_args))
    batch_entities.extend(get_ordered_user_relations_et(kg_args))
    batch_entities.extend(get_ordered_entity_relations_et(kg_args))
    return batch_entities
