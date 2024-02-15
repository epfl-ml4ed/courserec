from test_agent import predict_paths, evaluate_paths
from utils import *


def validate(policy_file, epoch, args):
    print("Validating the model...")
    path_file = args.log_dir + "/validate_paths_epoch_{}.pkl".format(epoch)
    train_labels = load_labels(args.dataset, "train")
    validation_labels = load_labels(args.dataset, "validation")

    predict_paths(policy_file, path_file, args, data="validation")
    avg_hit = evaluate_paths(
        args.dataset,
        path_file,
        train_labels,
        validation_labels,
        args.use_wandb,
        f"validation_result_epoch_{epoch}.txt",
        validation=True,
        sum_prob=args.sum_prob,
    )

    return avg_hit
