import torch
import torch.nn as nn
import numpy as np
from utils import Hyperparameters_model

class Linear_0(nn.Linear):

    def __init__(self, in_dim, out_dim, bias, std):

        super().__init__(in_features=in_dim, out_features=out_dim, bias=bias)

        if std > 0:
            self.weight.data.normal_(std=std)

        else:
            self.weight.data.zero_()

        if bias is True:
            self.bias.data.zero_()

        #self.bias.data.normal_(std=0.01)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)



class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        return -0.5 * (logs * 2. + ((x - mean) ** 2.) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return torch.sum(likelihood, dim=1)

    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean),
                           std=torch.ones_like(logs) * eps_std)
        #return mean + torch.exp(2. * logs) * eps
        return mean + torch.exp(logs) * eps
        # sampling: x = N(mu, sigma^2), then x = mu + sigma*eps, where sigma^2 is variance, and sigma is deviation

    @staticmethod
    def batchsample(batchsize, mean, logs, eps_std=None):
        eps_std = eps_std or 1
        sample = GaussianDiag.sample(mean, logs, eps_std)
        for i in range(1, batchsize):
            s = GaussianDiag.sample(mean, logs, eps_std)
            sample = torch.cat((sample, s), dim=0)
        return sample



class CondAffine(nn.Module):

    def __init__(self, in_x_dim, in_y_dim, latent_dim, warp_dims=[0]):
        super().__init__()


        self.in_x_dim = in_x_dim
        self.in_y_dim = in_y_dim
        self.latent_dim = latent_dim
        self.weight_std = 0.01
        self.eps = 1e-6
        self.warp_dims = warp_dims
        self.keep_dims = list(range(0, in_y_dim))
        for ind in self.warp_dims:
            self.keep_dims.remove(ind)


        s_cond_w = [
            nn.Linear(self.in_x_dim, self.latent_dim, bias=False),
            nn.BatchNorm1d(self.latent_dim),
            #Swish(),
            nn.SiLU(),
            #nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim, bias=True)
        ]

        s_cond_b = [
            nn.Linear(self.in_x_dim, self.latent_dim, bias=False),
            nn.BatchNorm1d(self.latent_dim),
            #Swish(),
            nn.SiLU(),
            #nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim, bias=True)
        ]


        s_y_keep = [
            nn.Linear(len(self.keep_dims), self.latent_dim, bias=False),
            nn.BatchNorm1d(self.latent_dim),
            #nn.ReLU(inplace=True),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim, bias=True),
            nn.BatchNorm1d(self.latent_dim, affine=False)
        ]

        s_merge = [
            #nn.ReLU(inplace=True),
            nn.SiLU(),
            nn.Linear(self.latent_dim, len(self.warp_dims), bias=True),
        ]

        self.s_cond_w = nn.Sequential(*s_cond_w)
        self.s_cond_b = nn.Sequential(*s_cond_b)
        self.s_y_keep = nn.Sequential(*s_y_keep)
        self.s_merge = nn.Sequential(*s_merge)
        self.s_last_active = nn.Softsign()


        b_cond_w = [
            nn.Linear(self.in_x_dim, self.latent_dim, bias=False),
            nn.BatchNorm1d(self.latent_dim),
            #Swish(),
            nn.SiLU(),
            #nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim, bias=True)
        ]

        b_cond_b = [
            nn.Linear(self.in_x_dim, self.latent_dim, bias=False),
            nn.BatchNorm1d(self.latent_dim),
            #Swish(),
            nn.SiLU(),
            #nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim, bias=True)
        ]


        b_y_keep = [
            nn.Linear(len(self.keep_dims), self.latent_dim, bias=False),
            nn.BatchNorm1d(self.latent_dim),
            #nn.ReLU(inplace=True),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim, bias=True),
            nn.BatchNorm1d(self.latent_dim, affine=False)
        ]

        b_merge = [
            #nn.ReLU(inplace=True),
            nn.SiLU(),
            nn.Linear(self.latent_dim, len(self.warp_dims), bias=True),
        ]

        self.b_cond_w = nn.Sequential(*b_cond_w)
        self.b_cond_b = nn.Sequential(*b_cond_b)
        self.b_y_keep = nn.Sequential(*b_y_keep)
        self.b_merge = nn.Sequential(*b_merge)

        with torch.no_grad():
            self.s_cond_w[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.s_cond_w[-1].bias.data, 0.0)
            self.s_cond_b[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.s_cond_b[-1].bias.data, 0.0)
            self.s_merge[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.s_merge[-1].bias.data, 0.0)


            self.b_cond_w[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.b_cond_w[-1].bias.data, 0.0)
            self.b_cond_b[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.b_cond_b[-1].bias.data, 0.0)
            self.b_merge[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.b_merge[-1].bias.data, 0.0)


    def compute_weights(self, x, y):

        logvar = torch.zeros_like(y)
        mu = torch.zeros_like(y)

        s_cond_b = self.s_cond_b(x)
        s_cond_w = torch.add(self.eps, torch.exp(self.s_cond_w(x)))
        s_y_keep = self.s_y_keep(y[:, self.keep_dims].contiguous())
        log_s = self.s_merge(s_cond_w * s_y_keep + s_cond_b)
        logvar[:,self.warp_dims] = self.s_last_active(log_s)

        b_cond_b = self.b_cond_b(x)
        b_cond_w = torch.add(self.eps, torch.exp(self.b_cond_w(x)))
        b_y_keep = self.b_y_keep(y[:, self.keep_dims].contiguous())
        b = self.b_merge(b_cond_w * b_y_keep + b_cond_b)
        mu[:,self.warp_dims] = b


        logvar = logvar.contiguous()
        mu = mu.contiguous()

        return logvar, mu



    def forward(self, x, y):

        logs, b = self.compute_weights(x, y)
        s = torch.sqrt(torch.add(self.eps, torch.exp(logs)))


        z = s * y + b

        logdet = torch.sum(torch.log(s), dim=1)
        return z, logdet

    def reverse(self, x, z):

        logs, b = self.compute_weights(x, z)
        s = torch.sqrt(torch.add(self.eps, torch.exp(logs)))



        y = (z - b) / s

        logdet = -torch.sum(torch.log(s), dim=1)

        return y, logdet


class FlowStep(nn.Module):

    def __init__(self, in_x_dim, in_y_dim, latent_dim):

        super().__init__()
        self.in_x_dim = in_x_dim
        self.in_y_dim = in_y_dim
        self.latent_dim = latent_dim

        self.nvp1 = CondAffine(in_x_dim, in_y_dim, latent_dim, warp_dims=[0])
        self.nvp2 = CondAffine(in_x_dim, in_y_dim, latent_dim, warp_dims=[1])



    def forward(self, x, y):

        z = y
        z, logdet1 = self.nvp1(x, z)
        z, logdet2 = self.nvp2(x, z)

        return z, logdet1+logdet2

    def reverse(self, x, z):

        y = z
        y, logdet2 = self.nvp2.reverse(x, y)
        y, logdet1 = self.nvp1.reverse(x, y)

        return y, logdet1+logdet2


class NVPFlow(nn.Module):

    def __init__(self, n_flows, in_x_dim, in_y_dim, latent_dim):

        super().__init__()

        self.n_flows = n_flows
        self.f_n_features = in_x_dim
        self.g_n_features = in_y_dim
        self.latent_dim = latent_dim

        self.flows = nn.ModuleList(
            [FlowStep(in_x_dim, in_y_dim, latent_dim) for i in range(n_flows)]
        )


    def forward(self, x, y):

        slogdet = 0.0
        z = y
        for flowstep in self.flows:
            z, logdet = flowstep(x, z)
            slogdet = slogdet + logdet

        return z, slogdet


    def reverse(self, x, z):

        slogdet = 0.0

        y = z
        for flowstep in reversed(self.flows):
            y, logdet = flowstep.reverse(x, y)
            slogdet = slogdet + logdet

        return y, slogdet



class FlowModel(nn.Module):

    def __init__(self, hyparams:Hyperparameters_model):

        super().__init__()

        self.flows = None

        self.x_dim = hyparams.x_dim
        self.y_dim = hyparams.num_class
        self.latent_dim = hyparams.latent_dim

        self.n_flows = hyparams.n_flows
        self.flows = NVPFlow(hyparams.n_flows, hyparams.x_dim, hyparams.num_class, hyparams.latent_dim)


        # prior
        self.register_parameter("new_mean",
                                nn.Parameter(
                                    torch.zeros([1, self.y_dim]
                                )))


        self.register_parameter("new_logs",
                                nn.Parameter(torch.zeros(
                                    [1,
                                     self.y_dim]
                                )))


    def prior(self):
        mean = torch.zeros_like(self.new_mean)
        logvar = torch.ones_like(self.new_mean)
        scale = 0
        logvar = scale * logvar


        return mean, logvar


    def forward(self, x):

        B = x.size(0)
        mean, logvar = self.prior()
        mean = torch.repeat_interleave(mean, repeats=B, dim=0)
        logvar = torch.repeat_interleave(logvar, repeats=B, dim=0)
        z = GaussianDiag.sample(mean, logvar)
        y, logdet = self.flows(x, z)

        p_z = GaussianDiag.logp(mean, logvar, z)
        p_y = p_z - logdet
        p_y = p_y / float(np.log(2.) * self.y_dim)
        nll = - p_y

        return y, nll
