import os

import numpy as np
import torch
from torch.autograd import Variable

from dpipe.model import Model, FrozenModel, get_model_path


class TorchModel(Model):
    """`Model` interface implementation for the PyTorch framework."""

    def __init__(self, model_core: torch.nn.Module, logits2pred: callable, logits2loss: callable,
                 optimize: torch.optim.Optimizer, cuda: bool = True):
        """
        Parameters
        ----------
        model_core: torch.nn.Module
            torch model structure
        logits2pred: callable(logits) -> prediction
            last layer nonlinearity that maps logits to predictions
        logits2loss: callable(logits) -> loss
            the loss function
        optimize: torch.optim.Optimizer
            the optimizer
        cuda: bool, optional
            whether to move the model's parameters to CUDA
        """
        if cuda:
            model_core.cuda()
            if hasattr(logits2loss, 'cuda'):
                logits2loss.cuda()

        self.cuda = cuda
        self.model_core = model_core
        self.logits2pred = logits2pred
        self.logits2loss = logits2loss
        self.optimizer = optimize(model_core.parameters())

    def do_train_step(self, *inputs, lr):
        self.model_core.train()
        inputs = [to_var(x, self.cuda) for x in inputs]
        *inputs, target = inputs

        logits = self.model_core(*inputs)
        loss = self.logits2loss(logits, target)

        set_lr(self.optimizer, lr)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return to_np(loss)[0]

    def do_val_step(self, *inputs):
        self.model_core.eval()
        inputs = [to_var(x, self.cuda, volatile=True) for x in inputs]
        *inputs, target = inputs

        logits = self.model_core(*inputs)
        y_pred = self.logits2pred(logits)
        loss = self.logits2loss(logits, target)

        return to_np(y_pred), to_np(loss)[0]

    def do_inf_step(self, *inputs):
        self.model_core.eval()
        inputs = [to_var(x, self.cuda, volatile=True) for x in inputs]

        logits = self.model_core(*inputs)
        y_pred = self.logits2pred(logits)

        return to_np(y_pred)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        path = get_model_path(path)
        state_dict = self.model_core.state_dict()
        torch.save(state_dict, path)

    def load(self, path: str):
        path = get_model_path(path)
        self.model_core.load_state_dict(torch.load(path))

    def transfer_load(self, model_path: str):
        print("glebgleb transferring weights from {}".format(model_path))
        state_to_transfer = torch.load(model_path)
        del state_to_transfer['fc.weight']
        del state_to_transfer['fc.bias']
        state = self.model_core.state_dict()
        state.update(state_to_transfer)
        self.model_core.load_state_dict(state)


class TorchFrozenModel(FrozenModel):
    def __init__(self, model_core: torch.nn.Module, logits2pred: callable, restore_model_path: str, cuda=True):
        """
        Parameters
        ----------
        model_core: torch.nn.Module
            torch model structure
        logits2pred: callable(logits) -> prediction
            last layer nonlinearity that maps logits to predictions
        restore_model_path: str
            the path to the trained model
        cuda: bool, optional
            whether to move the model's parameters to CUDA
        """
        if cuda:
            model_core.cuda()
        self.cuda = cuda
        self.model_core = model_core
        self.logits2pred = logits2pred

        path = get_model_path(restore_model_path)
        self.model_core.load_state_dict(torch.load(path))

    def do_inf_step(self, *inputs):
        self.model_core.eval()
        inputs = [to_var(x, self.cuda) for x in inputs]

        logits = self.model_core(*inputs)
        y_pred = self.logits2pred(logits)

        return to_np(y_pred)


def to_np(x: Variable):
    """
    Convert a autograd Variable to a numpy array.

    Parameters
    ----------
    x: Variable
    """
    return x.cpu().data.numpy()


validate_dtype = {
    np.bool: np.float32,
}


def to_var(x: np.array, cuda: bool, volatile: bool = False):
    """
    Convert a numpy array to a torch Tensor

    Parameters
    ----------
    x: np.array
        the input tensor
    cuda: bool
        move tensor to cuda
    volatile: bool, optional
        make tensor volatile
    """
    # torch doesn't support conversion from all numpy types:
    for dtype in validate_dtype:
        if x.dtype == dtype:
            x = x.astype(validate_dtype[dtype])
            break
    x = Variable(torch.from_numpy(x), volatile=volatile)
    if (torch.cuda.is_available() and cuda is None) or cuda:
        x = x.cuda()
    return x


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    """
    Change an optimizer's learning rate.

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
    lr: float

    Returns
    -------
    optimizer: torch.optim.Optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
