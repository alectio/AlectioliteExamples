import os
import torch
import sys
import yaml
import torch
import argparse

# Add the hedwig repo to syspath
cwd = os.path.dirname(os.path.abspath(__file__))
syspath = os.path.join(cwd)
sys.path.append(syspath)

# Import the Bert model
from hedwig.models.bert import __main__
from models.bert.args import get_args


def getdatasetstate(config_args={}):
    return {k: k for k in range(config_args["trainsize"])}


def train(config_args, labeled, resume_from: int = 0, ckpt_file: str = ""):
    print('inside train')
    args = get_args()
    # Add the args from the config file
    args.__dict__.update(config_args)
    args.labeled = labeled
    args.model = "bert-base-uncased"
    args.snapshot_path = os.path.join(cwd, args.EXPT_DIR, ckpt_file)
    args.data_dir = os.path.join(syspath, "hedwig-data", "datasets")
    args.infer = False

    if not os.path.isdir(os.path.join(cwd, args.EXPT_DIR)):
        os.mkdir(os.path.join(cwd, args.EXPT_DIR))

    __main__.main(args)
    return


def test(config_args, ckpt_file):
    args = get_args()
    # Add the args from the config file
    args.__dict__.update(config_args)
    args.split = "dev"
    args.infer = True
    args.trained_model = os.path.join(cwd, args.EXPT_DIR, ckpt_file)
    args.data_dir = os.path.join(syspath, "hedwig-data", "datasets")

    _, _, predictions, labels = __main__.main(args)

    return {"predictions": predictions, "labels": labels}


def infer(config_args, unlabeled, ckpt_file):
    args = get_args()
    # Add the args from the config file
    args.__dict__.update(config_args)
    args.split = "train"
    args.infer = True
    args.trained_model = os.path.join(cwd, args.EXPT_DIR, ckpt_file)
    args.data_dir = os.path.join(syspath, "hedwig-data", "datasets")

    scores = __main__.main(args)
    _, predicted = torch.max(scores, 1)

    k = 0 
    outputs_fin = {}
    for j in range(len(outputs)):
        outputs_fin[k] = {}
        outputs_fin[k]["prediction"] = predicted[j].item()
        outputs_fin[k]["logits"] = scores[j].cpu().numpy().tolist()
        k += 1

    return {"outputs": outputs_fin}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(cwd, "config.yaml"),
        type=str,
        help="Path to config.yaml",
    )
    args = parser.parse_args()
    print('args.config path', args.config)

    with open(args.config, "r") as stream:
        args = yaml.safe_load(stream)
    print('args ', args)
    labeled = list(range(100))
    resume_from = None
    ckpt_file = "ckpt_0.pt"

    print("Running Training")
    train(args, labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    print("Running Testing")
    test(args, ckpt_file=ckpt_file)
    print("Running Inference")
    print(infer(args, unlabeled=[10, 20, 30], ckpt_file=ckpt_file))
