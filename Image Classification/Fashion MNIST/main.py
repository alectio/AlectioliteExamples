import alectiolite
from alectiolite.callbacks import CurateCallback


from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from model import NeuralNet
import torch.optim as optim
import os
import yaml
import argparse

from rich.console import Console
console = Console(style="green")

device = "cuda" if torch.cuda.is_available() else "cpu"


def getdatasetstate(args={}):
    return {k: k for k in range(args["train_size"])}


def processData(args, stageFor="train", indices=None):

    # images from pytorch are grey-scale 0-1 in pixel values, scale each image by subtracting a mean of 0.5,
    # and dividing by a std of 0.5 to bring the range of values between [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    # download the data if it doesn't exisit already

    if args["DATASET"] == "Fashion":
        print("Downloading Fashion-MNIST Data")
        trainset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, transform=transform, download=True
        )
        testset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, transform=transform, download=True
        )
    else:
        print("Downloading MNIST Data")

        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, transform=transform, download=True
        )
        testset = torchvision.datasets.MNIST(
            root="./data", train=False, transform=transform, download=True
        )

    # 60k train, 10k test
    data_subset = None
    loader = None

    if stageFor == "train":
        data_subset = Subset(trainset, indices)
        loader = DataLoader(data_subset, batch_size=args["batch_size"], shuffle=True)
    elif stageFor == "test":
        loader = DataLoader(testset, batch_size=args["batch_size"], shuffle=False)
    elif stageFor == "infer":
        data_subset = Subset(trainset, indices)
        loader = DataLoader(data_subset, batch_size=args["batch_size"], shuffle=True)

    return loader


def train(args, labeled, resume_from, ckpt_file):
    print("========== In the train step ==========")
    batch_size = args["batch_size"]
    lr = args["learning_rate"]
    momentum = args["momentum"]
    epochs = args["train_epochs"]
    train_split = args["split_train"]

    loader = processData(args, stageFor="train", indices=labeled)

    net = NeuralNet()
    net = net.to(device=device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=float(lr), momentum=momentum)

    if resume_from is not None:
        ckpt = torch.load(os.path.join(args["EXPT_DIR"], resume_from + ".pth"))
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        getdatasetstate(args)

    net.train()

    for epoch in tqdm(range(args["train_epochs"]), desc="Training"):

        running_loss = 0

        for i, batch in enumerate(loader, start=0):
            data, labels = batch

            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 1000:
                print(
                    "epoch: {} batch: {} running-loss: {}".format(
                        epoch + 1, i + 1, running_loss / 1000
                    ),
                    end="\r",
                )
                running_loss = 0

    print("Finished Training. Saving the model as {}".format(ckpt_file))

    ckpt = {"model": net.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(ckpt, os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))

    return


def test(args, ckpt_file):
    print("========== In the test step ==========")
    batch_size = args["batch_size"]
    lr = args["learning_rate"]
    momentum = args["momentum"]
    epochs = args["train_epochs"]
    train_split = args["split_train"]

    loader = processData(args, stageFor="test")

    net = NeuralNet()
    net = net.to(device=device)

    net.load_state_dict(
        torch.load(os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))["model"]
    )

    net.eval()
    predix = 0
    predictions = {}
    truelabels = {}

    predictions, targets,logits = [], [],[]
    n_val = args["test_size"]
    soft = torch.nn.Softmax(dim=0)
    with tqdm(total=n_val, desc="Testing round", unit="batch", leave=False) as pbar:
        for step, (batch_x, batch_y) in enumerate(loader):
            with torch.no_grad():
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = net(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy().tolist())
                targets.extend(batch_y.cpu().numpy().tolist())
                logits.extend(soft(outputs).cpu().numpy().tolist())
            # for logit, label in zip(prediction, batch_y):
            #     # predictions[predix] = logit.cpu().numpy().tolist()
            #     truelabels[predix] = label.cpu().numpy().tolist()
            #     class_probabilities = logit.cpu().numpy().tolist()
            #     index_max = np.argmax(class_probabilities)
            #     predictions[predix] = index_max

            #     predix += 1

            pbar.update()



    return {"predictions": predictions, "labels": targets, "logits": logits}


def infer(args, unlabeled, ckpt_file):
    print("========== In the inference step ==========")
    batch_size = args["batch_size"]
    lr = args["learning_rate"]
    momentum = args["momentum"]
    epochs = args["train_epochs"]
    train_split = args["split_train"]

    loader = processData(args, stageFor="infer", indices=unlabeled)

    net = NeuralNet()
    net = net.to(device=device)

    net.load_state_dict(
        torch.load(os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))["model"]
    )

    net.eval()

    n_val = len(unlabeled)
    predictions = {}
    predix = 0

    with tqdm(total=n_val, desc="Inference round", unit="batch", leave=False) as pbar:
        for step, (batch_x, batch_y) in enumerate(loader):
            with torch.no_grad():
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                prediction = net(batch_x)

            for logit in prediction:
                predictions[unlabeled[predix]] = {}

                class_probabilities = logit.cpu().numpy().tolist()
                predictions[unlabeled[predix]]["logits"] = class_probabilities
                index_max = np.argmax(class_probabilities)
                predictions[unlabeled[predix]]["prediction"] = index_max
                predix += 1

            pbar.update()

    return {"outputs": predictions}


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--config",
#         default=os.path.join(os.getcwd(), "config.yaml"),
#         type=str,
#         help="Path to config.yaml",
#     )
#     args = parser.parse_args()

#     with open(args.config, "r") as stream:
#         args = yaml.safe_load(stream)

#     labeled = list(range(5000))
#     resume_from = None
#     ckpt_file = "ckpt_0"

#     print("Testing getdatasetstate")
#     getdatasetstate(args=args)
#     train(args=args, labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
#     test(args=args, ckpt_file=ckpt_file)
#     infer(args=args, unlabeled=[10, 20, 30], ckpt_file=ckpt_file)


if __name__ == "__main__":
    status = "Start"
    TOKEN = "0834dad5b2114c7999db00540f7b418a"
    with open("config.yaml", "r") as stream:
        args = yaml.safe_load(stream)
    while status != "Complete":
        print("Preparing to run FASHION-MNIST experiment")
        # Step 1 Get experiment config
        config = alectiolite.experiment_config(token=TOKEN)
        print(config)
        # Step 2 Initialize your callback
        cb = CurateCallback()

        # Step 3 Tap what type of experiment you want to run
        alectiolite.curate_classification(config=config, callbacks=[cb])

        # Step 4 Tap overrideables
        datasetstate = getdatasetstate(args=args)

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

        status = cb.on_experiment_end(token= TOKEN)
