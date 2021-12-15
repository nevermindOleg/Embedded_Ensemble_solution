import torch
import torch.nn as nn


from typing import Any, Callable, Mapping, Optional, Tuple
from tensorboardX import SummaryWriter


#TODO: everything random should have seeds


class EnsembleSet(nn.Module):
    """
    Set of Z ensembles of two-layer perceptrons.
    We need it for massive parallel training
    of several models with different hyperparameters

    """

    def __init__(self,
                input_dim: int,
                hid_dim: int,
                output_dim: int,  #TODO: output_dim
                n_models: int,
                n_ensembles: int,
                p: torch.Tensor,
                activation: Optional[nn.Module]=None):
        """
        :input_dim:   d, shape of input data
        :hid_dim:     N, shape of hidden layer
        :output_dim:  F, shape of output data
        :n_models:    M, number of models we ensemble
        :n_ensembles: E, number of ensembles, evaluatied in parallel
        :p:           p, vector of floats of shape E
        :gamma:          optional vector of shape E. If not specified it is computed automatically
        :activation:     nn.Module used as activation function. nn.ReLU by default

        """  #TODO: output_dim

        assert ((0 <= p) & (p <= 1)).all(), "parameter p should lie between 0 and 1"
        assert p.shape == (n_ensembles, ), "p should be of shape E"
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim  #TODO: output_dim
        self.n_models = n_models
        self.p = p
        self.activation = activation if activation is None else nn.ReLU()

        self.W = nn.Parameter(torch.randn((n_ensembles, hid_dim, input_dim)))
        self.U = p[:,None,None,None] + (1 - p**2)[:,None,None,None] * torch.randn(
                    (n_ensembles, n_models, output_dim, hid_dim),  #TODO: output_dim
                    requires_grad=False
                )


    @staticmethod
    def choose_gamma(p: torch.Tensor) -> torch.Tensor:
        """
        Automatic choice of gamma
        :p: scale parameter for matrix U

        """
        raise NotImplementedError()

    
    @staticmethod
    def loss(f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Loss function
        :f: predicted value, tensor of shape (E, M, F)
        :y: true value

        returns: MSE
        """  #TODO: forward return shape
             #TODO: loss func
             # returns MSE or RMSE or logloss or smth, correct if wrong
        
        raise NotImplementedError()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.input_dim and len(x.shape) == 2, f"x should be Bxd (any x {self.input_dim})"
        f_a = torch.einsum('emfn,end,bd->bemf', self.U, self.W, x)   # (E,N,d) @ (B,d) = (B,E,N)
        f_ens = f_a.mean(2)  #TODO: forward return shape
        # above string is 1/M sum_alpha=1^M
        return f_ens  # (B,E,F) or (B,E,F,M)  #TODO: forward return shape



class EnsembleModel(EnsembleSet):
    """
    EnsembleSet, but n_ensembles is 1
    
    """
    def __init__(self, input_dim: int,
                hid_dim: int,
                output_dim: int,  #TODO: output_dim
                n_models: int,
                # n_ensembles: int,
                p: float,
                activation: Optional[nn.Module]=None
        ):
        """
        :input_dim:   d, shape of input data
        :hid_dim:     N, shape of hidden layer
        :output_dim:  F, shape of output data
        :n_models:    M, number of models we ensemble
        :n_ensembles: E, number of ensembles, evaluatied in parallel
        :p:           p, vector of floats of shape E
        :gamma:          optional vector of shape E. If not specified it is computed automatically
        :activation:     nn.Module used as activation function. nn.ReLU by default

        """  #TODO: output_dim
        super().__init__(input_dim=input_dim,
                    hid_dim=hid_dim,
                    output_dim=output_dim,  #TODO output_dim
                    n_models=n_models,
                    n_ensembles=1,
                    p=torch.tensor([p], requires_grad=False),
                    activation=activation)

    def forward(self, x):
        return super().forward(x).squeeze(1)
    
    @staticmethod
    def choose_gamma(p: float) -> float:
        return super().choose_gamma(torch.tensor([p])).item()

    @staticmethod
    def loss(f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        super().loss(f.unsqueeze(1), y)


def generate_data(
        input_dim: int,
        output_dim: int,
        func: Callable[[torch.Tensor],torch.Tensor],
        interval:Tuple[float, float],
        n_points: int,
        epsilon: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates data

    :func: vectorised operation R^d -> R^F to be approximated
    :interval: tuple(a,b) -- region for sampling x from uniform [a,b]^d
    :n_points: B, batch_size
    :epsilon: gaussian noise variance

    :return: (x,y) -- pair of features and targets
    """
    # sample x from interval (x of shape Bxd)
    # do not forget `requires_grad=False`
    #TODO: data generator function
    x = torch.empty(n_points, input_dim, requires_grad=False).uniform_(*interval)
    y = func(x) + epsilon * torch.randn(n_points, output_dim, requires_grad=False)
    return x, y


def train_loop(model: EnsembleSet, n_epochs: int,
        optimizer,
        x: torch.Tensor,
        y: torch.Tensor,
        gammas: Optional[torch.Tensor]=None,
        logger: Optional[SummaryWriter]=None
    ):
    """
    Trining loop.
    :model: model to train
    :n_epochs: number of epochs

    :return: IDK
    """

    raise NotImplementedError()

    # some initializing actions, if needed
    # not sure about a single line down
    if logger is None: logger = SummaryWriter()
    model.train()
    for i in range(n_epochs):
        f = model(x)
        loss = model.loss(f, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #TODO: training loop
        # do not forget about tensorboard
        # lr scheduler etc are a point of discussions
    
    return None  # should it even return