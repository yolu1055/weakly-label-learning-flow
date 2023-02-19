import os
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
import datetime
from models import FlowModel
from utils import Hyperparameters_model, Hyperparameters_optim, count_parameters
import fashion_mnist_dataset
from learner import Learner
import random

def main(args):


    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.cuda = torch.cuda.is_available()

    print("device: {}".format(args.device))
    print("number of gpus: {}".format(args.num_gpus))


    # output dir
    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
        .replace(":", "")\
        .replace(" ", "_")

    args.out_root = os.path.join(args.out_root, "log_" + date)
    if not os.path.exists(args.out_root):
        os.makedirs(args.out_root)

    raw_data = fashion_mnist_dataset.load_dataset(args.data_path, args.num_weak_signals, args.classes)

    trainset = fashion_mnist_dataset.FashionMNISTDataset(raw_data["training_data"],
                                                         raw_data["training_labels"],
                                                         args.num_classes,
                                                         raw_data["training_ub"],
                                                         raw_data["training_weak_signal_probs"])



    testset = fashion_mnist_dataset.FashionMNISTDataset(raw_data["test_data"],
                                                        raw_data["test_labels"],
                                                        args.num_classes,
                                                        raw_data["test_ub"],
                                                        raw_data["test_weak_signal_probs"])





    training_loader = None
    if trainset is not None:
        training_loader = DataLoader(trainset,
                                     batch_size=len(trainset),
                                     shuffle=True,
                                     drop_last=True)

    test_loader = None
    if testset is not None:
        test_loader = DataLoader(testset,
                                 batch_size=len(testset),
                                 shuffle=False,
                                 drop_last=False)


    print("train set size: {}".format(len(trainset)))
    print("test set size: {}".format(len(testset)))

    args.x_dim = trainset.feature_dim

    model_hypers = Hyperparameters_model(
        dataset = "fashion-mnist",
        num_class = 2,
        x_dim = trainset.feature_dim,
        latent_dim = args.latent_dim,
        n_flows = args.n_flows,
        lambda1 = args.lambda1,
        lambda2 = args.lambda2,
        lambda3 = args.lambda3,
        lambda4 = args.lambda4
    )

    model_hypers.save(os.path.join(args.out_root, "model_hyparameters.json"))

    model = FlowModel(model_hypers)

    if args.cuda:
        model = model.cuda()

    model_size = count_parameters(model)/1e7
    print("model size: {:.2f}M".format(model_size))

    # optim

    optim = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.lr_decay)


    if args.scheduler_decay <= 0:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, args.scheduler_decay)

    optim_hypers = Hyperparameters_optim(
        optim = args.optimizer,
        lr = args.lr,
        betas = args.betas,
        eps = args.eps,
        lr_decay = args.lr_decay,
        scheduler_decay = args.scheduler_decay,
        max_grad_norm = args.max_grad_norm
    )

    optim_hypers.save(os.path.join(args.out_root, "optim_hyparameters.json"))

    # begin to train
    learner = Learner(args, model, optim, scheduler, training_loader, test_loader)

    learner.train()


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Label Learning Flow')

    # dataset
    parser.add_argument("--data_path", type=str, default="./datasets/fashion-mnist")
    parser.add_argument("--num_weak_signals", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classes", type=str, default="3,7", choices=["3,7", "4,8", "5,9"])


    # output
    parser.add_argument("--out_root", type=str, default="./results")

    # models
    parser.add_argument("--x_dim", type=int, default=784)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--n_flows", type=int, default=8)
    parser.add_argument("--lambda1", type=float, default=10.0)
    parser.add_argument("--lambda2", type=float, default=10.0)
    parser.add_argument("--lambda3", type=float, default=10.0)
    parser.add_argument("--lambda4", type=float, default=10.0)


    # optim
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--betas", type=tuple, default=(0.9,0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--lr_decay", type=float, default=0.0)
    parser.add_argument("--scheduler_decay", type=float, default=0.996)


    # Learner
    parser.add_argument("--max_grad_norm", type=float, default=1000)


    parser.add_argument("--num_epochs", type=int, default=501)
    parser.add_argument("--nll_gap", type=int, default=10)
    parser.add_argument("--save_gap", type=int, default=100)
    parser.add_argument("--valid_gap", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10)


    # devices
    args = parser.parse_args()

    main(args)
