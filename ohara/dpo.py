import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor

#######################################################################################################
# https://arxiv.org/pdf/2305.18290 
# ... eqn 7
# dpo_loss = - log( sigmoid( beta * (  log(pi_win/ref_win) - log(pi_lose /ref_lose)  ) ) )
#
# remember log property: log(x/y) = log(x) - log(y)
# lets start witn sub eq,
# =  log(pi_win/ref_win) - log(pi_lose /ref_lose)
# = log( (pi_win/ref_win) /  ( pi_lose /ref_lose) )
# = log( (pi_win/ref_win) *  ( ref_lose / pi_lose) )
# = log( (pi_win/ ref_win) *  ( ref_lose / ref_win) )
# = log( (pi_win/ ref_win) /  (   ref_win/ ref_lose) )
# = log(pi_win/ ref_win) - log(ref_win/ ref_lose)
# = (log(pi_win) - log(ref_win) ) -  (log(ref_win) - log(ref_lose))
#
# so now we have
# logits = win_logprop - lose_logprop
# where:
# win_logprop = log(pi_win) - log(ref_win)
# lose_logprop = log(ref_win) - log(ref_lose)
#
# and eqn is
# dpo_loss = - log(sigmoid(beta * logits))


def dpo_loss(
    pi_logps:Tensor,
    ref_logps:Tensor, 
    win_output_idxs:Tensor, 
    lose_output_idxs:Tensor,
    beta:float
    ) -> tuple[Tensor, Tensor]:
    """
    paper: https://arxiv.org/pdf/2305.18290.pdf
    """

    # extracting only outputs log probabilities
    pi_win_logps, pi_lose_logps = pi_logps[win_output_idxs], pi_logps[lose_output_idxs]
    ref_win_logps, ref_lose_logps = ref_logps[win_output_idxs], ref_logps[lose_output_idxs]
    
    # log(a/b) = log(a) - log(b)
    
    pi_logratios = pi_win_logps - pi_lose_logps
    ref_logratios = ref_win_logps - ref_lose_logps
    
    logits = pi_logratios - ref_logratios
    
    # Dpo loss = - log( sigmoid(beta)
    losses: Tensor = -F.logsigmoid(beta * logits)
    rewards: Tensor = beta * (pi_logps - ref_logps).detach()
    return losses, rewards


def cdpo_loss(
    pi_logps:Tensor,
    ref_logps:Tensor, 
    win_output_idxs:Tensor, 
    lose_output_idxs:Tensor,
    beta:float,
    label_smoothing=0.2,
    ) -> tuple[Tensor, Tensor]:
    """
    paper:https://ericmitchell.ai/cdpo.pdf
    """

    # extracting only outputs log probabilities
    pi_win_logps, pi_lose_logps = pi_logps[win_output_idxs], pi_logps[lose_output_idxs]
    ref_win_logps, ref_lose_logps = ref_logps[win_output_idxs], ref_logps[lose_output_idxs]
    
    # log(a/b) = log(a) - log(b)
    
    pi_logratios = pi_win_logps - pi_lose_logps
    ref_logratios = ref_win_logps - ref_lose_logps
    
    logits = pi_logratios - ref_logratios
    
    # Dpo loss = - log( sigmoid(beta)
    losses: Tensor = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )
    rewards: Tensor = beta * (pi_logps - ref_logps).detach()
    return losses, rewards


def ipo_loss(
    pi_logps:Tensor,
    ref_logps:Tensor, 
    win_output_idxs:Tensor, 
    lose_output_idxs:Tensor,
    beta:float
    ) -> tuple[Tensor, Tensor]:
    """
    paper: https://arxiv.org/pdf/2310.12036v2.pdf
    """

    # extracting only outputs log probabilities
    pi_win_logps, pi_lose_logps = pi_logps[win_output_idxs], pi_logps[lose_output_idxs]
    ref_win_logps, ref_lose_logps = ref_logps[win_output_idxs], ref_logps[lose_output_idxs]
    
    # log(a/b) = log(a) - log(b)
    
    pi_logratios = pi_win_logps - pi_lose_logps
    ref_logratios = ref_win_logps - ref_lose_logps
    
    logits = pi_logratios - ref_logratios
    
    # Dpo loss = - log( sigmoid(beta)
    losses: Tensor = (logits - 1/(2 * beta)) ** 2 
    rewards: Tensor = beta * (pi_logps - ref_logps).detach()
    return losses, rewards

