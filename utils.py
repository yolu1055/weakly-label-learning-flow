import json
import numpy as np
from tabulate import tabulate
import torch


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.numpy().tolist()
        return json.JSONEncoder.default(self, obj)

class Hyperparameters_model:

    def __init__(self,
                 dataset,
                 num_class,
                 x_dim,
                 latent_dim,
                 n_flows,
                 lambda1,
                 lambda2,
                 lambda3,
                 lambda4
                 ):

        self.dataset = dataset
        self.num_class = num_class
        self.x_dim = x_dim
        self.latent_dim = latent_dim
        self.n_flows = n_flows
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4


    def save(self, path):
        self.path = path
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True, cls=NumpyEncoder)

    def print(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        print(tabulate(rows))



class Hyperparameters_optim:

    def __init__(self,
                 optim,
                 lr,
                 betas,
                 eps,
                 lr_decay,
                 scheduler_decay,
                 max_grad_norm,
                 ):

        self.optim = optim
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.lr_decay = lr_decay
        self.scheduler_decay = scheduler_decay
        self.max_grad_norm = max_grad_norm


    def save(self, path):
        self.path = path
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True, cls=NumpyEncoder)

    def print(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        print(tabulate(rows))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def load_state(path, cuda):
    if cuda:
        print ("load to gpu")
        state = torch.load(path)

    else:
        print ("load to cpu")
        state = torch.load(path, map_location=lambda storage, loc: storage)

    return state



def compute_acc(y_, y):

    n = len(y)
    correct = 0.0

    for i in range(0, n):
        if y_[i] == y[i]:
            correct = correct + 1.0
    return correct / float(n)



def print_results(results, output_path):

    with open(output_path, "a") as file:

        line = ""

        for i, result in enumerate(results):
            if i == 0:
                line += f"{result}"
            elif i == 1:
                line += f"{result:.2f}"
            else:
                line += f"{result:.5f}"
            line += "\t"

        file.write(line)
        file.write("\n")


