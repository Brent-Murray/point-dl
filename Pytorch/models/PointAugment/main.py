import logging
import os
import sys
import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.tools import (
    IOStream,
    PointCloudsInPickle,
    _init_,
    delete_files,
    notifi,
    plot_stats,
)
from utils.augmentation import AugmentPointCloudsInPickle
from utils.train import test, train

warnings.filterwarnings("ignore")

def main(params):
    # set up folder structure
    _init_(params["exp_name"])

    # initiate IOStream
    io = IOStream("checkpoints/" + params["exp_name"] + "/run.log")
    io.cprint(params["exp_name"])
    io.cprint(str(params))

    if params["cuda"]:
        io.cprint("Using GPU")
    else:
        io.cprint("Using CPU")

    # Load datasets
    train_data_path = os.path.join(params["train_path"], str(params["num_points"]))
    train_pickle = params["train_pickle"]
    trainset = PointCloudsInPickle(train_data_path, train_pickle)
    if params["augment"] == True:
        for i in range(params["n_augs"]):
            aug_trainset = AugmentPointCloudsInPickle(
                train_data_path, 
                train_pickle,
            )
            
            trainset = torch.utils.data.ConcatDataset(
                [trainset, aug_trainset]
            )
    trainloader = DataLoader(trainset, batch_size=params["batch_size"], shuffle=True, pin_memory=True)

    test_data_path = os.path.join(params["test_path"], str(params["num_points"]))
    test_pickle = params["test_pickle"]
    testset = PointCloudsInPickle(test_data_path, test_pickle)
    testloader = DataLoader(testset, batch_size=params["batch_size"], shuffle=False, pin_memory=True)

    if not params["eval"]:
        train(params, io, trainloader, testloader)
        torch.cuda.empty_cache()
    else:
        test(params, io, testloader)
        
        
if __name__ == "__main__":
    params = {
        "exp_name": "dgcnn_pointaugment_7168",  # experiment name
        "model": "dgcnn",  # model
        "batch_size": 10,  # batch size
        "train_path": r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\train\trainingsets\fps",
        "train_pickle": r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\train\plot_comp.pkl",
        "test_path": r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\val\trainingsets\fps",
        "test_pickle": r"D:\MurrayBrent\data\RMF_ITD\plots\plot_laz\val\plot_comp.pkl",
        "augment": True, # augment
        "n_augs": 5, # number of augmentations
        "classes": ["BF", "BW", "CE", "LA", "PT", "PJ", "SB"],  # classes
        "n_gpus": torch.cuda.device_count(),  # number of gpus
        "epochs": 300,  # total epochs
        "optimizer_a": "adam",  # augmentor optimizer,
        "optimizer_c": "adam",  # classifier optimizer
        "lr_a": 1e-6,  # augmentor learning rate
        "lr_c": 1e-6,  # classifier learning rate
        "adaptive_lr": True,  # adaptive learning rate
        "patience": 20,  # patience
        "step_size": 30,  # step size
        "momentum": 0.9,  # sgd momentum
        "num_points": 7168,  # number of points
        "dropout": 0.5,  # dropout rate
        "emb_dims": 1024,  # dimension of embeddings
        "k": 20,  # k nearest points
        "model_path": "",  # pretrained model path
        "cuda": True,  # use cuda
        "eval": False,  # run testing
    }

    main(params)