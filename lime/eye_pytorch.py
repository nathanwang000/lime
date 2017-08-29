import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
import math

# the linear model in pytorch (caution, this is not logistic regression because target is continous)
def getModel(indim):
    return nn.Linear(indim, 1)

# training 1 step with regularization
def train(model, loss, optimizer, x_val, y_val):
    x = Variable(torch.from_numpy(x_val).float(), requires_grad=False)
    y = Variable(torch.from_numpy(y_val).float(), requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss(fx, y, model.weight)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    #print(output.data[0])
    return output.data[0] # this really is loss

def predict(model, x_val):
    x = Variable(torch.from_numpy(x_val).float(), requires_grad=False)
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)

# run the model untill convergence
def unison_shuffled_copies(X, y):
    assert X.shape[0] == y.shape[0]
    p = np.random.permutation(y.shape[0])
    return X[p], y[p]

def eye_loss(loss, alpha, r=None): # loss is the data loss

    def eye(x):
        nonlocal r # default to all unknown
        if r is None:
            r = Variable(torch.zeros(x.numel()), requires_grad=False)
        # r = Variable(torch.ones(x.numel()), requires_grad=False)
        l1 = torch.abs(x * (1-r)).sum()
        # l1 = ((1-r) * x).norm(1)
        l2sq = ((r*x) * (r*x)).sum()
        # l2sq = (r * x).dot(r * x)
        return  l1 + torch.sqrt(l1**2 + l2sq)

    def mEYE(x):
        nonlocal r # default to all unknown
        r = r or Variable(torch.zeros(x.numel()), requires_grad=False)
        l1 = torch.abs(x * (1-r)).sum()        
        l2sq = (r * x).dot(r * x)
        return  l1 + l1**2 + l2sq

    return lambda X, y, theta: loss(X, y) + eye(theta) * alpha
    # return lambda X, y, theta: loss(X, y) + mEYE(theta) * alpha

def fit(X, y, alpha=0.01, risk=None):

    n_examples = X.shape[0]
    model = getModel(X.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss = eye_loss(torch.nn.MSELoss(size_average=True), alpha=alpha, r=risk)
    batch_size = 100

    for i in range(30): # was 100 epochs
        X, y = unison_shuffled_copies(X, y)
        cost = 0.
        num_batches = math.ceil(n_examples / batch_size)
        for k in range(num_batches):
            start, end = k * batch_size, min((k + 1) * batch_size, n_examples)
            cost += train(model, loss, optimizer, X[start:end], y[start:end])
            
    # print("Epoch %d, train loss = %f" % (i + 1, cost / num_batches))
    ret = model.weight.data.numpy()[0]
    ret[np.abs(ret) < np.max(np.abs(ret)) * 1e-2] = 0 # set too small weight to 0
    return ret

# eye_path to select at most k features
def eye_path(X, y, num_features, n_alphas=60, risk=None): # n_alphas was 30
    n_samples, n_features = X.shape
    alpha_max = 1
    eps = 1e-5
    alphas = np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max), n_alphas)
    # coefs = np.empty((n_features, n_alphas), dtype=X.dtype)

    best_nonzero = None
    best_c = None
    best_len = n_features + 1

    for i, alpha in enumerate(alphas):
        c = fit(X, y, alpha, risk=risk)
        nonzero = c.nonzero()[0]
        print("%.3e non_zero: %d" % (alphas[i], len(nonzero)))
        if len(nonzero) < best_len:
            best_nonzero = nonzero
            best_len = len(nonzero)
            best_c = c
        if len(nonzero) <= num_features:
            break
    if len(nonzero) > num_features:
        # choose the highest weights if too little non zeros
        feature_weights = sorted(zip(range(best_c.size), best_c),
                                 key=lambda x: np.abs(x[1]), reverse=True)
        return np.array([x[0] for x in feature_weights[:num_features]])
    used_features = best_nonzero
    return used_features
