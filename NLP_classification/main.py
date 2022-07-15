import argparse
import yaml, json
import os
import logging

from tqdm import tqdm
from rich.console import Console

import alectiolite
from alectiolite.callbacks import CurateCallback

from processes import train, test, infer, getdatasetstate

console = Console(style="green")
# put the train/test/infer processes into the constructor

cwd = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default=os.path.join(cwd, "config.yaml"),
    type=str,
    help="Path to config.yaml",
)
args = parser.parse_args()

with open(args.config, "r") as stream:
    args = yaml.safe_load(stream)

if __name__ == "__main__":
    TOKEN = "c2a5c135211049c287def7957adbdfaa"
    status = "Start"
    while status != "Complete":
        print("Preparing to run Toxic Comment Classification experiment")
        # Step 1 Get experiment config
        config = alectiolite.experiment_config(token=TOKEN)
        print(config)
        # Step 2 Initialize your callback
        cb = CurateCallback()

        # Step 3 Tap what type of experiment you want to run
        alectiolite.curate_classification(config=config, callbacks=[cb])
        # Step 4 Tap overrideables
        datasetsize = 1200
        datasetstate = {ix: str(n) for ix, n in enumerate(range(datasetsize))}
        # On ready to start experiment
        cb.on_experiment_start(monitor="datasetstate", data=datasetstate, config=config)
        console.print("Calling train start !!!! ")
        # Get selected indices
        labeled = cb.on_train_start(monitor="selected_indices", config=config)
        train_outs = train(args,labeled, resume_from=None, ckpt_file="ckpt_0")
        cb.on_train_end(monitor="insights", data = train_outs, config=config)

        test_outs = test(args,ckpt_file="ckpt_0")
        cb.on_test_end(monitor="metrics", data = test_outs, config=config)
        
        unlabeled = cb.on_infer_start()
        infer_outs = infer(args,unlabeled, ckpt_file="ckpt_0")
        cb.on_infer_end(monitor="logits", data=infer_outs, config=config)

        status = cb.on_experiment_end(token=TOKEN)
