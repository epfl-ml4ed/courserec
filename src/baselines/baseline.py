import yaml
import os
import argparse
import json

from collections import defaultdict

import numpy as np

from recbole.config import Config
from recbole.utils import init_seed, get_model, get_trainer
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction


def load_config(config_file):
    """Load config and model from config file.

    Args:
        config_file (str): yaml config file path

    Returns:
        model_name (str): model name
        config (Config): config object
    """
    with open(config_file, "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    model_name = config["model"]
    config = Config(model=model_name, config_file_list=[config_file])
    return model_name, config


def evaluate(topk_matches, test_user_products):
    """Compute the ranking metrics

    Args:
        topk_matches (dict): TopK Predictions from the model for each user
        test_user_products (dict): Ground truth items for each user

    Returns:
        dict: ranking metrics
    """
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

        dcg_all = 0.0
        hit_num_all = 0.0
        hit_at_1_all = 0.0
        hit_at_3_all = 0.0
        hit_at_5_all = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                dcg_all += 1.0 / (np.log(i + 2) / np.log(2))
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
            idcg_all += 1.0 / (np.log(i + 2) / np.log(2))
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

    avg_precision_all = np.mean(precisions_all) * 100
    avg_recall_all = np.mean(recalls_all) * 100
    avg_ndcg_all = np.mean(ndcgs_all) * 100
    avg_hit_all = np.mean(hits_all) * 100
    avg_hit_at_1_all = np.mean(hits_at_1_all) * 100
    avg_hit_at_3_all = np.mean(hits_at_3_all) * 100
    avg_hit_at_5_all = np.mean(hits_at_5_all) * 100

    print(
        "NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | HR@1={:.3f} | HR@3={:.3f} | HR@5={:.3f} \n".format(
            avg_ndcg_all,
            avg_recall_all,
            avg_hit_all,
            avg_precision_all,
            avg_hit_at_1_all,
            avg_hit_at_3_all,
            avg_hit_at_5_all,
        )
    )

    metrics_dict = {
        "NDCG": avg_ndcg_all,
        "Recall": avg_recall_all,
        "HR": avg_hit_all,
        "Precision": avg_precision_all,
    }

    return metrics_dict


def get_topk_and_interactions(model, test_data):
    """Get the model predictions and ground truth interactions.

    Args:
        model (recbole.model): recommender model
        test_data (recbole.data.dataset): test set

    Returns:
        dict,dict: model predictions and ground truth interactions
    """
    num_items = test_data._dataset.item_num
    num_users = test_data._dataset.user_num
    topk_matches = defaultdict(list)
    interactions = defaultdict(list)
    model.to("cpu")
    for user_id in range(1, num_users):
        row = test_data._dataset.inter_matrix().getrow(user_id).nonzero()[1]
        interactions[user_id] = row.tolist()
        tmp = Interaction(
            {
                "user_id": [user_id for i in range(num_items)],
                "item_id": [i for i in range(num_items)],
            }
        )
        prediction = (
            model.predict(tmp).cpu().detach().numpy().argsort()[-10:][::-1].tolist()
        )
        topk_matches[user_id] = prediction
    return topk_matches, interactions


def train(config, model_name, train_data, valid_data, test_data):
    """Train and evaluate the model.

    Args:
        config (Config): config object
        model_name (str): name of the model
        train_data (recbole.data.dataset): train dataset
        valid_data (recbole.data.dataset): valid dataset
        test_data (recbole.data.dataset): test dataset

    Returns:
        dict: ranking metrics on test_data

    """
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    trainer.fit(train_data, valid_data, verbose=False)

    topk_matches, interactions = get_topk_and_interactions(model, test_data)

    metrics_dict = evaluate(topk_matches, interactions)
    return metrics_dict


def run(config_file):
    """Train and evaluate the model multiple times and save the mean and std results.

    Args:
        config_file (str): yaml config file path
    """

    model_name, config = load_config(config_file)
    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)

    print(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    all_metrics = {
        "NDCG": [],
        "Recall": [],
        "HR": [],
        "Precision": [],
    }

    for num in range(config["run_num"]):
        print("run_num: ", num)
        metrics_dict = train(config, model_name, train_data, valid_data, test_data)

        for key in all_metrics:
            all_metrics[key].append(metrics_dict[key])

        results_path = os.path.join(
            config["results_dir"], model_name, str(num), "metrics.json"
        )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(metrics_dict, f)

    for key in all_metrics:
        all_metrics[key] = [np.mean(all_metrics[key]), np.std(all_metrics[key])]

    print("All metrics: ")
    print(all_metrics)
    all_results_path = os.path.join(
        config["results_dir"], model_name, "all_metrics.json"
    )

    with open(all_results_path, "w") as f:
        json.dump(all_metrics, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/Pop.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
