import torch
import torch.nn.functional as F

from torch import Tensor

IGNORE_INDEX = -100


def sequence_logps(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """
    Compute per-sample summed log-probs for a sequence.
    logits: (B, T, V)
    labels: (B, T) with ignore_index for masked tokens
    returns: (B,)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    mask = labels != ignore_index
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def _dpo_logits(
    pi_win_logps: Tensor,
    pi_lose_logps: Tensor,
    ref_win_logps: Tensor,
    ref_lose_logps: Tensor,
) -> Tensor:
    pi_logratios = pi_win_logps - pi_lose_logps
    ref_logratios = ref_win_logps - ref_lose_logps
    return pi_logratios - ref_logratios


def dpo_loss_from_logps(
    pi_win_logps: Tensor,
    pi_lose_logps: Tensor,
    ref_win_logps: Tensor,
    ref_lose_logps: Tensor,
    beta: float,
) -> tuple[Tensor, Tensor]:
    logits = _dpo_logits(pi_win_logps, pi_lose_logps, ref_win_logps, ref_lose_logps)
    losses: Tensor = -F.logsigmoid(beta * logits)
    rewards: Tensor = beta * (
        torch.stack([pi_win_logps, pi_lose_logps], dim=0)
        - torch.stack([ref_win_logps, ref_lose_logps], dim=0)
    ).detach()
    return losses, rewards


def cdpo_loss_from_logps(
    pi_win_logps: Tensor,
    pi_lose_logps: Tensor,
    ref_win_logps: Tensor,
    ref_lose_logps: Tensor,
    beta: float,
    label_smoothing: float = 0.2,
) -> tuple[Tensor, Tensor]:
    logits = _dpo_logits(pi_win_logps, pi_lose_logps, ref_win_logps, ref_lose_logps)
    losses: Tensor = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )
    rewards: Tensor = beta * (
        torch.stack([pi_win_logps, pi_lose_logps], dim=0)
        - torch.stack([ref_win_logps, ref_lose_logps], dim=0)
    ).detach()
    return losses, rewards


def ipo_loss_from_logps(
    pi_win_logps: Tensor,
    pi_lose_logps: Tensor,
    ref_win_logps: Tensor,
    ref_lose_logps: Tensor,
    beta: float,
) -> tuple[Tensor, Tensor]:
    logits = _dpo_logits(pi_win_logps, pi_lose_logps, ref_win_logps, ref_lose_logps)
    losses: Tensor = (logits - 1 / (2 * beta)) ** 2
    rewards: Tensor = beta * (
        torch.stack([pi_win_logps, pi_lose_logps], dim=0)
        - torch.stack([ref_win_logps, ref_lose_logps], dim=0)
    ).detach()
    return losses, rewards

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

    losses, _ = dpo_loss_from_logps(
        pi_win_logps, pi_lose_logps, ref_win_logps, ref_lose_logps, beta
    )
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

    losses, _ = cdpo_loss_from_logps(
        pi_win_logps, pi_lose_logps, ref_win_logps, ref_lose_logps, beta, label_smoothing
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

    losses, _ = ipo_loss_from_logps(
        pi_win_logps, pi_lose_logps, ref_win_logps, ref_lose_logps, beta
    )
    rewards: Tensor = beta * (pi_logps - ref_logps).detach()
    return losses, rewards
