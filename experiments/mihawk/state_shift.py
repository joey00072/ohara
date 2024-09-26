import torch

from torch import Tensor


def states_shift(tensor, n):
    """
    Shifts the tensor elements to the right along the sequence length dimension by n positions.
    The shape of the tensor is expected to be (B, seq_len, *, *, ...).
    Elements shifted into the tensor are filled with zeros.

    Args:
    - tensor (torch.Tensor): The input tensor of shape (B, seq_len, *, *, ...).
    - n (int): The number of positions to shift the elements to the right.

    Returns:
    - torch.Tensor: The shifted tensor with the same shape as the input tensor.
    """
    if n <= 0:
        return tensor
    B, seq_len, *remaining_dims = tensor.shape
    # Ensure n does not exceed seq_len
    n = min(n, seq_len)
    zero_pad = torch.zeros(B, n, *remaining_dims, dtype=tensor.dtype, device=tensor.device)
    shifted_tensor = torch.cat((tensor[:, n:], zero_pad), dim=1)
    return shifted_tensor


def state_predition_loss(tensor:Tensor,shift:int)->Tensor:
    tensor_ones = torch.ones_like(tensor)
    ones_shifted = states_shift(tensor_ones, shift) 
    tensor_shifted = states_shift(tensor, shift) 
    loss = torch.nn.functional.mse_loss(tensor*ones_shifted,tensor_shifted)
    return loss
 
    
if __name__ == "__main__":
    # Example usage
    B,T,C = 1,10,2
    tensor = torch.rand(B,T,C) 
    shifted_tensor = states_shift(tensor, 2)  # Shift by 2 positions

    print(f"{state_predition_loss(tensor,2)=}")

    print(tensor)
    print(shifted_tensor)
