from __future__ import absolute_import, division, print_function

from easydict import EasyDict as edict
import numpy as np
import torch
import torch.nn as nn

from utils import *

class KnowledgeEmbedding(nn.Module):
    def __init__(self, dataset, args, kg_args):
        super(KnowledgeEmbedding, self).__init__()
        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda
        self.use_user_relations = args.use_user_relations
        self.use_entity_relations = args.use_entity_relations
        self.kg_args = kg_args

        # Initialize entity embeddings.
        self.entities = edict({
            e: edict(vocab_size=getattr(dataset, e).vocab_size) for e in self.kg_args.entities
        })
        for e in self.entities:
            embed = self._entity_embedding(self.entities[e].vocab_size)
            setattr(self, e, embed)

        # Initialize relation embeddings and relation biases.

        relations = {
                self.kg_args.interaction: edict(
                et='item',
                et_distrib=self._make_distrib(dataset.interactions.item_uniform_distrib))
        }

        relations.update({r : edict(
            et=ri[1],
            et_distrib=self._make_distrib(getattr(dataset, r).et_distrib)
        ) for r, ri in self.kg_args.item_relation.items()})

        if self.use_user_relations:
            relations.update({r : edict(
                et=ri[1],
                et_distrib=self._make_distrib(getattr(dataset, r).et_distrib)
            ) for r, ri in self.kg_args.user_relation.items()})
        
        if self.use_entity_relations:
            relations.update({r : edict(
                eh=ri[1],
                et=ri[2],
                et_distrib=self._make_distrib(getattr(dataset, r).et_distrib)
            ) for r, ri in self.kg_args.entity_relation.items()})

        self.relations = edict(relations)
        
        for r in self.relations:
            embed = self._relation_embedding()
            setattr(self, r, embed)
            bias = self._relation_bias(len(self.relations[r].et_distrib))
            setattr(self, r + '_bias', bias)

    def _entity_embedding(self, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size].
            Note that last dimension is always 0's.
        """
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False).to(self.device)
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(vocab_size + 1, self.embed_size).uniform_(-initrange, initrange).to(self.device)
        embed.weight = nn.Parameter(weight)
        return embed

    def _relation_embedding(self):
        """Create relation vector of size [1, embed_size]."""
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(1, self.embed_size).uniform_(-initrange, initrange).to(self.device)
        embed = nn.Parameter(weight).to(self.device)
        return embed

    def _relation_bias(self, vocab_size):
        """Create relation bias of size [vocab_size+1]."""
        bias = nn.Embedding(vocab_size + 1, 1, padding_idx=-1, sparse=False).to(self.device)
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    def _make_distrib(self, distrib):
        """Normalize input numpy vector to distribution."""
        distrib = np.power(np.array(distrib, dtype=float), 0.75)
        distrib = distrib / distrib.sum()
        distrib = torch.FloatTensor(distrib).to(self.device)
        return distrib

    def forward(self, batch_idxs):
        loss = self.compute_loss(batch_idxs)
        return loss

    def compute_loss(self, batch_idxs):
        """Compute knowledge graph negative sampling loss.
        batch_idxs: batch_size *  array, where each row is
        (user_id, item_id, ...item_features, ...user_features, ...entity_features).
        """
        idxs = edict({
            f"{e}_idx": batch_idxs[:, i] for i, e in enumerate(get_batch_entities(self.kg_args))
        })
       
        regularizations = []

        # user + interaction -> item
        ui_loss, ui_embeds = self.neg_loss('user', self.kg_args.interaction, 'item', idxs.user_idx, idxs.item_idx)
        regularizations.extend(ui_embeds)
        loss = ui_loss

        for r in self.kg_args.item_relation:
            et = self.kg_args.item_relation[r][1]
            it_loss, it_embed = self.neg_loss('item', r, et, idxs.item_idx, idxs[f'{et}_idx'])
            if it_loss is not None:
                regularizations.extend(it_embed)
                loss += it_loss

        if self.use_user_relations:
            for r in self.kg_args.user_relation:
                et = self.kg_args.user_relation[r][1]
                it_loss, it_embed = self.neg_loss('user', r, et, idxs.user_idx, idxs[f'ur_{et}_idx'])
                if it_loss is not None:
                    regularizations.extend(it_embed)
                    loss += it_loss
        
        if self.use_entity_relations:
            for r in self.kg_args.entity_relation:
                et = self.kg_args.entity_relation[r][2]
                eh = self.kg_args.entity_relation[r][1]
                it_loss, it_embed = self.neg_loss(eh, r, et, idxs[f'{eh}_idx'], idxs[f'er_{et}_idx'])
                if it_loss is not None:
                    regularizations.extend(it_embed)
                    loss += it_loss

        # l2 regularization
        if self.l2_lambda > 0:
            l2_loss = 0.0
            for term in regularizations:
                l2_loss += torch.norm(term)
            loss += self.l2_lambda * l2_loss

        return loss

    def neg_loss(self, entity_head, relation, entity_tail, entity_head_idxs, entity_tail_idxs):
        # Entity tail indices can be -1. Remove these indices. Batch size may be changed!
        mask = entity_tail_idxs >= 0
        fixed_entity_head_idxs = entity_head_idxs[mask]
        fixed_entity_tail_idxs = entity_tail_idxs[mask]
        if fixed_entity_head_idxs.size(0) <= 0:
            return None, []

        entity_head_embedding = getattr(self, entity_head)  # nn.Embedding
        entity_tail_embedding = getattr(self, entity_tail)  # nn.Embedding
        relation_vec = getattr(self, relation)  # [1, embed_size]
        relation_bias_embedding = getattr(self, relation + '_bias')  # nn.Embedding
        entity_tail_distrib = self.relations[relation].et_distrib  # [vocab_size]

        return kg_neg_loss(entity_head_embedding, entity_tail_embedding,
                           fixed_entity_head_idxs, fixed_entity_tail_idxs,
                           relation_vec, relation_bias_embedding, self.num_neg_samples, entity_tail_distrib)


def kg_neg_loss(entity_head_embed, entity_tail_embed, entity_head_idxs, entity_tail_idxs,
                relation_vec, relation_bias_embed, num_samples, distrib):
    """Compute negative sampling loss for triple (entity_head, relation, entity_tail).

    Args:
        entity_head_embed: Tensor of size [batch_size, embed_size].
        entity_tail_embed: Tensor of size [batch_size, embed_size].
        entity_head_idxs:
        entity_tail_idxs:
        relation_vec: Parameter of size [1, embed_size].
        relation_bias: Tensor of size [batch_size]
        num_samples: An integer.
        distrib: Tensor of size [vocab_size].

    Returns:
        A tensor of [1].
    """

    batch_size = entity_head_idxs.size(0)
    entity_head_vec = entity_head_embed(entity_head_idxs.to(torch.int64))  # [batch_size, embed_size]
    example_vec = entity_head_vec + relation_vec  # [batch_size, embed_size]
    example_vec = example_vec.unsqueeze(2)  # [batch_size, embed_size, 1]
    entity_tail_vec = entity_tail_embed(entity_tail_idxs.to(torch.int64))  # [batch_size, embed_size]
    pos_vec = entity_tail_vec.unsqueeze(1)  # [batch_size, 1, embed_size]
    relation_bias = relation_bias_embed(entity_tail_idxs.to(torch.int64)).squeeze(1)  # [batch_size]
    pos_logits = torch.bmm(pos_vec, example_vec).squeeze() + relation_bias  # [batch_size]
    pos_loss = -pos_logits.sigmoid().log()  # [batch_size]

    neg_sample_idx = torch.multinomial(distrib, num_samples, replacement=True).view(-1)
    neg_vec = entity_tail_embed(neg_sample_idx)  # [num_samples, embed_size]
    neg_logits = torch.mm(example_vec.squeeze(2), neg_vec.transpose(1, 0).contiguous())
    neg_logits += relation_bias.unsqueeze(1)  # [batch_size, num_samples]
    neg_loss = -neg_logits.neg().sigmoid().log().sum(1)  # [batch_size]

    loss = (pos_loss + neg_loss).mean()
    return loss, [entity_head_vec, entity_tail_vec, neg_vec]

