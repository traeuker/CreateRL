import torch
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tt(ndarray):
    """
    Helper Function to cast observation to correct type/device
    """
    
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


def network_update(target, source, tau = 1):
    """
    Simple Helper for updating target-network parameters
    :param target: target network
    :param source: policy network
    :param tau: weight to regulate how strongly to update (1 -> copy over weights)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def transform_visual_input(input):
    if input.shape == (256, 256, 3):
        output= (input[:,:,0]+input[:,:,1]+input[:,:,2])/255
        # output = torch.Tensor(output)
        output = output[None, None, :]
        return output 
    output = input[None, None, :]
    return output
