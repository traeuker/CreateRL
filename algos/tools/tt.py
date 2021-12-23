import torch
from torch.autograd import Variable

def tt(ndarray):
    """
    Helper Function to cast observation to correct type/device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == "cuda":
        return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
    else:
        return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)
