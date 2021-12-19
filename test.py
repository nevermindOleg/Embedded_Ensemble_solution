from math import pi
from model import *
from random import randint
from tqdm import tqdm
from torchsummary import summary
from sklearn.model_selection import train_test_split
from torch.optim import Adam


device = 'cpu'

def test(d=5,N=20,F=3,M=4,E=2,B=200, v=False):
    if v: print("Start a test for:     %d, %d, %d, %d, %d, %d" % (d, N, F, M, E, B))

    p = torch.rand(E)
    model = EnsembleSet(d,N,F,M,E,p).to(device)

    # model2 = EnsembleModel(d,N,F,M,p[0])

    x,y=generate_data(d,F,
            nn.Sequential(
                nn.Tanh(),
                nn.Linear(d,F),
                nn.Tanh(),
                nn.Linear(F,F),
                nn.ReLU()
            ), (-pi/2, pi/2), B, 0.1
        )

    x = x.to(device)
    y = y.to(device)
    assert x.shape == (B, d), f"{x.shape} == {B, d}"
    assert y.shape == (B, F)
    assert torch.all(x <= pi/2) and torch.all(x >= -pi/2)
    

    f = model(x)
    assert f.shape == (B, E, F)

    loss = model.loss(f, y)
    assert loss.shape == (E, )

    Loss = loss.sum()
    assert torch.isfinite(Loss)

    # this test interfere with further training...
    # Loss.backward(retain_graph=True)

    if v: print("Assertions passed for %d, %d, %d, %d, %d, %d" % (d, N, F, M, E, B))

    return x, y, model

test()

for _ in tqdm(range(10)):
    d = randint(1, 20)
    N = randint(1, 20)
    F = randint(1, 20)
    M = randint(1, 20)
    E = randint(1, 20)
    B = randint(5, 200)

    gammas = torch.ones(E)

    x, y, model = test(d,N,F,M,E,B)
    # x1, x2, y1, y2 = train_test_split(x, y)
    x2, x1 = x[:B//3], x[B//3:]
    y2, y1 = y[:B//3], y[B//3:]

    optimizer = Adam(model.parameters())

    with torch.autograd.detect_anomaly():
        res = train_loop(model, 300,
                    optimizer,
                    (x1,y1), (x2,y2),
                )
    