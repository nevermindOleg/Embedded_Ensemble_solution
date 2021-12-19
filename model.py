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
                output_dim: int,
                n_models: int,
                n_ensembles: int,
                p: torch.Tensor,
                activation: Optional[nn.Module]=None,
                device:str = "cpu"
                ):
        """
        :input_dim:   d, shape of input data
        :hid_dim:     N, shape of hidden layer
        :output_dim:  F, shape of output data
        :n_models:    M, number of models we ensemble
        :n_ensembles: E, number of ensembles, evaluatied in parallel
        :p:           p, vector of floats of shape E
        :activation:     nn.Module used as activation function. nn.ReLU by default
        :device"         device
        """

        assert ((0 <= p) & (p <= 1)).all(), "parameter p should lie between 0 and 1"
        assert p.shape == (n_ensembles, ), "p should be of shape E"
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_models = n_models
        self.n_ensembles = n_ensembles
        self.device = device
        self.p = p
        self.activation = activation if activation is not None else nn.ReLU()

        self.W = nn.Parameter(torch.randn((n_ensembles, hid_dim, input_dim), device=device))
        self.U = p[:,None,None,None] + (1 - p**2)[:,None,None,None] * torch.randn(
                    (n_ensembles, n_models, output_dim, hid_dim),
                    requires_grad=False,
                    device=device
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
        :f: predicted value, tensor of shape (B, E, F)
        :y: true value, tensor of shape (B, F)

        returns: MSE loss per ensemble, vector of length E
        """
        B, E, F = f.shape  # consider correct shape of f
        mse = (f - y[:, None, :]) ** 2
        return mse.sum(dim=[0, 2])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.input_dim and len(x.shape) == 2, f"x should be Bxd (any x {self.input_dim})"
        # f_a = torch.einsum('emfn,end,bd->bemf', self.U, self.W, x)   # (E,N,d) @ (B,d) = (B,E,N)
        h = torch.einsum('end,bd->ben', self.W, x)
        h = self.activation(h/self.hid_dim ** .5)
        f_a = torch.einsum('emfn,ben->bemf', self.U, h)
        f_ens = f_a.mean(2)
        # above string is 1/M sum_{alpha=1}^{M}
        return f_ens  # (B,E,F)



class EnsembleModel(EnsembleSet):
    """
    EnsembleSet, but n_ensembles is 1
    not used anywhere, but who cares
    """
    def __init__(self, input_dim: int,
                hid_dim: int,
                output_dim: int,
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
        :p:           p, float
        :activation:     nn.Module used as activation function. nn.ReLU by default

        """
        super().__init__(input_dim=input_dim,
                    hid_dim=hid_dim,
                    output_dim=output_dim,
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
    x = torch.rand(n_points, input_dim, )
    a, b = interval
    x = (b - a) * x + a
    y = func(x) + epsilon * torch.randn(n_points, output_dim, )
    return x, y


def train_loop(model: EnsembleSet, n_epochs: int,
        optimizer,
        data_train: Tuple[torch.Tensor, torch.Tensor],
        data_test:  Tuple[torch.Tensor, torch.Tensor],
        gammas: Optional[torch.Tensor]=None,
    ) -> Mapping[str, Any]:
    """
    Trining loop.
    :model: model to train
    :n_epochs: number of epochs
    :optimizer: instance of optimizer from torch.optim, i.e. Adam 

    :return: model after training
    """
    x_train, y_train = data_train
    x_test, y_test = data_test
    p = model.p

    if gammas is None:
        try:
            gammas = model.choose_gamma(p)
        except NotImplementedError:
            gammas = torch.ones(model.n_ensembles, device=model.device) / model.n_models
    
    writer = SummaryWriter()
    writer.add_scalars("p",
                {str(i):p_ for i, p_ in enumerate(p.detach().cpu())}
            )
    writer.add_scalars("gamma",
                {str(i):g for i, g in enumerate(gammas.detach().cpu())}
            )        
                
    for epoch in range(n_epochs):
        model.train()
        f_train = model(x_train)
        loss_train = model.loss(f_train, y_train) * gammas  # per-ensemble loss
        loss_train.sum().backward(retain_graph=True)

        optimizer.step()
        optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            f_test = model(x_test)
            loss_test = model.loss(f_test, y_test) * gammas
            
            #TODO: tensorboard
            writer.add_scalars("loss/train",
                        {str(i):loss for i, loss in enumerate(loss_train.detach().cpu())}
                        , epoch)
            writer.add_scalars("loss/test",
                        {str(i):loss for i, loss in enumerate(loss_test.detach().cpu())}
                        , epoch)
    
    writer.flush()
    writer.close()
    return {"model": model, "optimizer": optimizer}