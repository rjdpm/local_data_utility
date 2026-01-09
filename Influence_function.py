import torch
from torch.autograd import grad
import numpy as np

__all__ = [
    'flatten_tensors',
    'unflatten_like',
    'hvp',
    'conjugate_gradient',
    'grad_z',
    'make_hvp_fn',
    'influence_function',
]


def flatten_tensors(tensors):
    return torch.cat([t.contiguous().view(-1) for t in tensors])

def unflatten_like(vector, like_tensors):
    outputs = []
    idx = 0
    for t in like_tensors:
        numel = t.numel()
        outputs.append(vector[idx:idx+numel].view_as(t))
        idx += numel
    return outputs

def hvp(loss, params, vector):
    grads = grad(loss, params, create_graph=True)
    flat_grads = flatten_tensors(grads)
    hv = grad(flat_grads, params, grad_outputs=vector)
    return flatten_tensors(hv)


def conjugate_gradient(hvp_fn, b, max_iter=50, tol=1e-5):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rs_old = torch.dot(r, r)

    for _ in range(max_iter):
        Ap = hvp_fn(p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x

def grad_z(model, loss_fn, x, y, input_dims):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.zero_grad()
    x = torch.split(x.to(device), input_dims, dim=1)
    pred = model(x)
    loss = loss_fn(pred, y)
    grads = grad(loss, model.parameters())
    
    return flatten_tensors(grads)


def make_hvp_fn(model, loss_fn, train_loader):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = list(model.parameters())
    model.to(device)

    def hvp_fn(v):
        hv = torch.zeros_like(v)
        for batch in train_loader:
            y = batch.pop('labels').to(device)
            try:
                batch = {key: val.to(device) for key, val in batch.items()}
            except:
                batch = {key: [temp_v.to(device) for temp_v in batch[key] if temp_v is not None] for key in batch.keys()}
            model.zero_grad()
            pred = model(**batch)
            loss = loss_fn(pred, y)
            hv += hvp(loss, params, v)
        return hv / len(train_loader)

    return hvp_fn


def influence_function(
    model,
    loss_fn,
    train_loader,
    test_point,
    train_point,
    input_dims,
    cg_iters=50
):
    x_test, y_test = test_point
    x_train, y_train = train_point

    # ∇θ ℓ(z_test)
    v = grad_z(model=model, loss_fn=loss_fn, x=x_test, y=y_test, input_dims=input_dims)
    # H⁻¹ v
    hvp_fn = make_hvp_fn(model, loss_fn, train_loader)
    inv_hvp = conjugate_gradient(hvp_fn, v, max_iter=cg_iters)

    # ∇θ ℓ(z_train)
    grad_train = grad_z(model, loss_fn, x_train, y_train, input_dims=input_dims)

    # Influence
    influence = -torch.dot(inv_hvp, grad_train)

    return influence.item()
