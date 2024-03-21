from __future__ import absolute_import, division, print_function
import json
from collections import Counter
import os
import argparse
from math import log
from tqdm import tqdm
from easydict import EasyDict as edict
import torch
from functools import reduce
from kg_env import BatchKGEnvironment
from actor_critic import ActorCritic
from utils import *
import wandb


def evaluate(
    topk_matches,
    test_user_products,
    use_wandb,
    tmp_dir,
    result_file_dir,
    result_file_name,
    min_courses=10,
    compute_all=True,
):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    # Compute metrics
    precisions, recalls, ndcgs, hits, hits_at_1, hits_at_3, hits_at_5 = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    (
        precisions_all,
        recalls_all,
        ndcgs_all,
        hits_all,
        hits_at_1_all,
        hits_at_3_all,
        hits_at_5_all,
    ) = ([], [], [], [], [], [], [])
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        is_invalid = False
        if uid not in topk_matches or len(topk_matches[uid]) < min_courses:
            invalid_users.append(uid)
            is_invalid = True
        pred_list, rel_set = topk_matches.get(uid, [])[::-1], test_user_products[uid]
        if len(pred_list) == 0:
            ndcgs_all.append(0.0)
            recalls_all.append(0.0)
            precisions_all.append(0.0)
            hits_all.append(0.0)
            hits_at_1_all.append(0.0)
            hits_at_3_all.append(0.0)
            hits_at_5_all.append(0.0)
            continue

        if is_invalid == False:
            dcg = 0.0
            hit_num = 0.0
            hit_at_1 = 0.0
            hit_at_3 = 0.0
            hit_at_5 = 0.0

            for i in range(len(pred_list)):
                if pred_list[i] in rel_set:
                    dcg += 1.0 / (log(i + 2) / log(2))
                    hit_num += 1
                    if i < 1:
                        hit_at_1 += 1
                    if i < 3:
                        hit_at_3 += 1
                    if i < 5:
                        hit_at_5 += 1
            # idcg
            idcg = 0.0
            for i in range(min(len(rel_set), len(pred_list))):
                idcg += 1.0 / (log(i + 2) / log(2))
            ndcg = dcg / idcg
            recall = hit_num / len(rel_set)
            precision = hit_num / len(pred_list)
            hit = 1.0 if hit_num > 0.0 else 0.0
            hit_at_1 = 1.0 if hit_at_1 > 0.0 else 0.0
            hit_at_3 = 1.0 if hit_at_3 > 0.0 else 0.0
            hit_at_5 = 1.0 if hit_at_5 > 0.0 else 0.0

            ndcgs.append(ndcg)
            recalls.append(recall)
            precisions.append(precision)
            hits.append(hit)
            hits_at_1.append(hit_at_1)
            hits_at_3.append(hit_at_3)
            hits_at_5.append(hit_at_5)

            ndcgs_all.append(ndcg)
            recalls_all.append(recall)
            precisions_all.append(precision)
            hits_all.append(hit)
            hits_at_1_all.append(hit_at_1)
            hits_at_3_all.append(hit_at_3)
            hits_at_5_all.append(hit_at_5)

        elif compute_all == True:
            dcg_all = 0.0
            hit_num_all = 0.0
            hit_at_1_all = 0.0
            hit_at_3_all = 0.0
            hit_at_5_all = 0.0
            for i in range(len(pred_list)):
                if pred_list[i] in rel_set:
                    dcg_all += 1.0 / (log(i + 2) / log(2))
                    hit_num_all += 1
                    if i < 1:
                        hit_at_1_all += 1
                    if i < 3:
                        hit_at_3_all += 1
                    if i < 5:
                        hit_at_5_all += 1
            # idcg
            idcg_all = 0.0
            for i in range(min(len(rel_set), len(pred_list))):
                idcg_all += 1.0 / (log(i + 2) / log(2))
            ndcg_all = dcg_all / idcg_all
            recall_all = hit_num_all / len(rel_set)
            precision_all = hit_num_all / len(pred_list)
            hit_all = 1.0 if hit_num_all > 0.0 else 0.0
            hit_at_1_all = 1.0 if hit_at_1_all > 0.0 else 0.0
            hit_at_3_all = 1.0 if hit_at_3_all > 0.0 else 0.0
            hit_at_5_all = 1.0 if hit_at_5_all > 0.0 else 0.0
            ndcgs_all.append(ndcg_all)
            recalls_all.append(recall_all)
            precisions_all.append(precision_all)
            hits_all.append(hit_all)
            hits_at_1_all.append(hit_at_1_all)
            hits_at_3_all.append(hit_at_3_all)
            hits_at_5_all.append(hit_at_5_all)
        else:
            ndcgs_all.append(0.0)
            recalls_all.append(0.0)
            precisions_all.append(0.0)
            hits_all.append(0.0)
            hits_at_1_all.append(0.0)
            hits_at_3_all.append(0.0)
            hits_at_5_all.append(0.0)

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_hit = np.mean(hits) * 100
    avg_hit_at_1 = np.mean(hits_at_1) * 100
    avg_hit_at_3 = np.mean(hits_at_3) * 100
    avg_hit_at_5 = np.mean(hits_at_5) * 100

    avg_precision_all = np.mean(precisions_all) * 100
    avg_recall_all = np.mean(recalls_all) * 100
    avg_ndcg_all = np.mean(ndcgs_all) * 100
    avg_hit_all = np.mean(hits_all) * 100
    avg_hit_at_1_all = np.mean(hits_at_1_all) * 100
    avg_hit_at_3_all = np.mean(hits_at_3_all) * 100
    avg_hit_at_5_all = np.mean(hits_at_5_all) * 100

    print(
        "Min courses to consider user valid={:.3f} |  Compute metrics for all users={}\n".format(
            min_courses, compute_all
        )
    )

    print(
        "NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | HR@1={:.3f} | HR@3={:.3f} | HR@5={:.3f} | Invalid users={}\n".format(
            avg_ndcg,
            avg_recall,
            avg_hit,
            avg_precision,
            avg_hit_at_1,
            avg_hit_at_3,
            avg_hit_at_5,
            len(invalid_users),
        )
    )
    print(
        "NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | HR@1={:.3f} | HR@3={:.3f} | HR@5={:.3f} | Computed for all users.\n".format(
            avg_ndcg_all,
            avg_recall_all,
            avg_hit_all,
            avg_precision_all,
            avg_hit_at_1_all,
            avg_hit_at_3_all,
            avg_hit_at_5_all,
        )
    )
    filename = os.path.join(tmp_dir, result_file_dir, result_file_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    metrics_all = {
        "ndcg": avg_ndcg_all,
        "recall": avg_recall_all,
        "hit": avg_hit_all,
        "precision": avg_precision_all,
    }

    json.dump(metrics_all, open(filename, "w"))

    if use_wandb:
        wandb.save(filename)

    return avg_precision, avg_recall, avg_ndcg, avg_hit


def evaluate_validation(topk_matches, test_user_products):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    # Compute metrics
    hits = []
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < 1:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid][::-1], test_user_products[uid]
        if len(pred_list) == 0:
            continue

        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                hit_num += 1

        hit = 1.0 if hit_num > 0.0 else 0.0
        hits.append(hit)

    avg_hit = np.mean(hits) * 100

    print(" HR={:.3f} | Invalid users={}\n".format(avg_hit, len(invalid_users)))

    return avg_hit


def batch_beam_search(env, model, kg_args, uids, device, topk=[10, 3, 1], policy=0):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(env.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    state_pool = env.reset(uids)  # numpy of [bs, dim]
    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in uids]
    model.eval()
    for hop in range(len(topk)):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        batch_act_embeddings = env.batch_action_embeddings(
            path_pool, acts_pool
        )  # numpy array of size [bs, 2*embed_size, act_dim]
        embeddings = torch.ByteTensor(batch_act_embeddings).to(device)
        probs, _ = model(
            (state_tensor, actmask_tensor, embeddings)
        )  # Tensor of [bs, act_dim]
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        topk_probs, topk_idxs = torch.topk(
            probs, topk[hop], dim=1
        )  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()

        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                if relation == kg_args.self_loop:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = kg_args.kg_relation[path[-1][1]][relation]
                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < len(topk) - 1:  # no need to update state at the last hop
            state_pool = env._batch_get_state(path_pool)
    return path_pool, probs_pool


def predict_paths(policy_file, path_file, args, kg_args, data="test"):
    print("Predicting paths...")
    env = BatchKGEnvironment(
        args.tmp_dir,
        kg_args,
        args.max_acts,
        max_path_len=args.max_path_len,
        state_history=args.state_history,
        reward_function=args.reward,
        use_pattern=args.use_pattern,
    )
    pretrain_sd = torch.load(policy_file, map_location=torch.device("cpu"))
    model = ActorCritic(
        env.state_dim,
        env.act_dim,
        gamma=args.gamma,
        hidden_sizes=args.hidden,
        modified_policy=args.modified_policy,
        embed_size=env.embed_size,
    ).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)

    test_labels = load_labels(args.tmp_dir, data)
    test_uids = list(test_labels.keys())

    batch_size = 16
    start_idx = 0
    all_paths, all_probs = [], []
    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        paths, probs = batch_beam_search(
            env,
            model,
            kg_args,
            batch_uids,
            args.device,
            topk=args.topk,
            policy=args.modified_policy,
        )
        all_paths.extend(paths)
        all_probs.extend(probs)
        start_idx = end_idx
        pbar.update(batch_size)
    predicts = {"paths": all_paths, "probs": all_probs}
    pickle.dump(predicts, open(path_file, "wb"))
    if args.use_wandb:
        wandb.save(path_file)


def evaluate_paths(
    dir_path,
    path_file,
    train_labels,
    test_labels,
    kg_args,
    use_wandb,
    result_file_dir,
    result_file_name,
    validation=False,
    sum_prob=False,
):
    embeds = load_embed(dir_path)
    user_embeds = embeds["user"]
    enroll_embeds = embeds[kg_args.interaction][0]
    course_embeds = embeds["item"]
    scores = np.dot(user_embeds + enroll_embeds, course_embeds.T)

    # 1) Get all valid paths for each user, compute path score and path probability.
    results = pickle.load(open(path_file, "rb"))
    pred_paths = {uid: {} for uid in test_labels}
    for path, probs in zip(results["paths"], results["probs"]):
        if path[-1][1] != "item":
            continue
        uid = path[0][2]
        if uid not in pred_paths:
            continue
        pid = path[-1][2]
        if pid not in pred_paths[uid]:
            pred_paths[uid][pid] = []
        path_score = scores[uid][pid]
        path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[uid][pid].append((path_score, path_prob, path))

    # 2) Compute the sum of probabilities for each user-course pair
    if sum_prob == True:
        user_course_probs = {}
        for uid in pred_paths:
            user_course_probs[uid] = Counter()
            for pid in pred_paths[uid]:
                prob_sum = sum(path[1] for path in pred_paths[uid][pid])
                user_course_probs[uid][pid] = prob_sum
                user_course_probs[uid].most_common(10)

        topk_matches = {}
        for uid, prob_dict in user_course_probs.items():
            topk_matches[uid] = [pid for pid, _ in prob_dict.most_common(10)]

        with open("json_data.json", "w") as f:
            json.dump(user_course_probs, f)

        if validation:
            return evaluate_validation(topk_matches, test_labels)

        else:
            for min_courses in [1, 10]:
                for compute_all in [True, False]:
                    evaluate(
                        topk_matches,
                        test_labels,
                        use_wandb,
                        args.tmp_dir,
                        result_file_dir=result_file_dir,
                        result_file_name=result_file_name,
                        min_courses=min_courses,
                        compute_all=compute_all,
                        sum_prob=sum_prob,
                    )

    # 3) Pick best path for each user-product pair, also remove pid if it is in train set.
    if sum_prob == False:
        best_pred_paths = {}
        for uid in pred_paths:
            train_pids = set(train_labels.get(uid, []))
            if len(train_pids) == 0:
                continue
            best_pred_paths[uid] = []
            for pid in pred_paths[uid]:
                if pid in train_pids:
                    continue
                # Get the path with highest probability
                sorted_path = sorted(
                    pred_paths[uid][pid], key=lambda x: x[1], reverse=True
                )
                best_pred_paths[uid].append(sorted_path[0])

        path_patterns = {}
        for uid in best_pred_paths:
            for path in best_pred_paths[uid]:
                path_pattern = path[2]
                pattern_key = ""
                for node in path_pattern:
                    pattern_key += node[0] + "_" + node[1] + "-->"
                path_patterns[pattern_key] = path_patterns.get(pattern_key, 0) + 1

        print(path_patterns)

        # 3) Compute top 10 recommended products for each user.
        sort_by = "score"
        pred_labels = {}
        for uid in best_pred_paths:
            if sort_by == "score":
                sorted_path = sorted(
                    best_pred_paths[uid], key=lambda x: (x[0], x[1]), reverse=True
                )
            elif sort_by == "prob":
                sorted_path = sorted(
                    best_pred_paths[uid], key=lambda x: (x[1], x[0]), reverse=True
                )
            top10_pids = [
                p[-1][2] for _, _, p in sorted_path[:10]
            ]  # from largest to smallest

            pred_labels[uid] = top10_pids[
                ::-1
            ]  # change order to from smallest to largest!

        if validation == True:
            return evaluate_validation(pred_labels, test_labels)

        else:
            for min_courses in [10]:
                for compute_all in [True]:
                    evaluate(
                        pred_labels,
                        test_labels,
                        use_wandb,
                        args.tmp_dir,
                        result_file_dir=result_file_dir,
                        result_file_name=result_file_name,
                        min_courses=10,
                        compute_all=compute_all,
                    )


def test(args, kg_args):
    policy_file = args.log_dir + "/tmp_policy_model_epoch_{}.ckpt".format(args.epochs)
    path_file = args.log_dir + "/policy_paths_epoch_{}.pkl".format(args.epochs)

    train_labels = load_labels(args.tmp_dir, "train")
    test_labels = load_labels(args.tmp_dir, "test")

    if args.run_path:
        predict_paths(policy_file, path_file, args, kg_args)
    if args.run_eval:
        evaluate_paths(
            args.tmp_dir,
            path_file,
            train_labels,
            test_labels,
            kg_args,
            args.use_wandb,
            args.result_file_dir,
            args.result_file_name,
            validation=False,
            sum_prob=args.sum_prob,
        )


if __name__ == "__main__":
    boolean = lambda x: (str(x).lower() == "true")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/UPGPR/mooc.json", help="Config file."
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    args = config.TEST_AGENT

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project_name, name=args.wandb_run_name, config=args
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    if args.early_stopping == True:
        with open("early_stopping.txt", "r") as f:
            args.epochs = int(f.read())

    args.log_dir = args.tmp_dir + "/" + args.name
    test(args, config.KG_ARGS)

    if args.use_wandb:
        wandb.finish()
