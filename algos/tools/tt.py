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


def decay(current_timestep, decay_length = 10_000, initial_eps = 1, final_eps = 0.1):
    
    if current_timestep < decay_length:
        eps = current_timestep * (final_eps-initial_eps) / decay_length + initial_eps
    else:
        eps = final_eps

    return eps

