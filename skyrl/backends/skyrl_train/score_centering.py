"""Native RF++ score centering; advantages are supplied by the caller."""

import math

import torch


def add_score_centering_inputs(batch, prepared, response_length):
    configs = prepared.all_loss_fn_configs
    if (
        len(set(prepared.all_loss_fns)) != 1
        or prepared.all_loss_fns[0] not in {"ppo_score_centered", "reinforce_score_centered"}
        or any(c != configs[0] for c in configs)
    ):
        raise ValueError("Score-centered batches require one loss and one configuration")
    config = configs[0] or {}
    cap = config.get("importance_cap", 2.0)
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("importance_cap must be finite and positive")
    k = int(config.get("score_centering_k", 0))
    if k < 1 or k != config.get("score_centering_k"):
        raise ValueError("score_centering_k must be a positive integer")
    if (
        not 0 <= config.get("eps_clip_low", 0.2) < 1
        or not math.isfinite(config.get("eps_clip_high", 0.2))
        or config.get("eps_clip_high", 0.2) < 0
    ):
        raise ValueError("Invalid PPO clipping thresholds")
    kl = config.get("kl_loss_coef", 0.0)
    if not math.isfinite(kl) or kl < 0:
        raise ValueError("kl_loss_coef must be finite and nonnegative")
    count = len(prepared.all_targets)
    if len(prepared.all_topk_token_ids) != count or len(prepared.all_topk_logprobs) != count:
        raise ValueError("Score centering requires decode heads for every datum")
    ids, logps, refs = [], [], []
    for i, targets in enumerate(prepared.all_targets):
        n = len(targets)
        if any(
            len(values[i]) != n
            for values in (
                prepared.all_token_weights,
                prepared.all_sampling_logprobs,
                prepared.all_advantages,
            )
        ):
            raise ValueError("Score centering inputs must align with target_tokens")
        if len(prepared.all_topk_token_ids[i]) != n * k or len(prepared.all_topk_logprobs[i]) != n * k:
            raise ValueError("Decode heads must have target-token length times score_centering_k entries")
        raw_ids = torch.tensor(prepared.all_topk_token_ids[i]).reshape(n, k)
        if raw_ids.is_floating_point() and not torch.equal(raw_ids, raw_ids.round()):
            raise ValueError("Decode head IDs must be integers")
        head_ids = raw_ids.to(torch.long)
        head_q = torch.tensor(prepared.all_topk_logprobs[i], dtype=torch.float32).reshape(n, k)
        active = torch.tensor(prepared.all_token_weights[i]) > 0
        if (head_ids[active] < 0).any() or not torch.isfinite(head_q[active]).all() or (head_q[active] > 0).any():
            raise ValueError("Invalid active decode head")
        if k > 1 and (head_ids[active].sort(-1).values.diff(dim=-1) == 0).any():
            raise ValueError("Decode head token IDs must be distinct")
        for values in (prepared.all_sampling_logprobs[i], prepared.all_advantages[i]):
            if not torch.isfinite(torch.tensor(values)[active]).all():
                raise ValueError("Nonfinite active logprob or advantage")
        # Inactive token payloads must not enter exponentials or vocabulary gathers.
        head_ids[~active] = 0
        head_q[~active] = 0
        padding = response_length - n
        ids.append(torch.nn.functional.pad(head_ids, (0, 0, padding, 0)))
        logps.append(torch.nn.functional.pad(head_q, (0, 0, padding, 0)))
        if kl:
            if len(prepared.all_reference_logprobs) != count or len(prepared.all_reference_logprobs[i]) != n:
                raise ValueError("k2 requires reference_logprobs for every target")
            ref = torch.tensor(prepared.all_reference_logprobs[i], dtype=torch.float32)
            if not torch.isfinite(ref[active]).all():
                raise ValueError("Nonfinite reference logprob")
            refs.append(torch.nn.functional.pad(ref.masked_fill(~active, 0), (padding, 0)))
    batch["topk_token_ids"] = torch.stack(ids)
    batch["topk_logprobs"] = torch.stack(logps)
    if refs:
        batch["reference_logprobs"] = torch.stack(refs)
    weights = batch["loss_mask"]
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Score centering requires finite nonnegative action weights with positive sum")
    normalized = weights.clone()
    for _, _, start, end in prepared.request_batch_slices:
        count = weights[start:end].sum()
        if count <= 0:
            raise ValueError("Each score-centered request needs action tokens")
        normalized[start:end] /= count
    batch["loss_mask"] = normalized


def score_centered_ppo_loss(
    logp,
    old,
    advantages,
    head_p,
    head_q,
    weights,
    eps_low,
    eps_high,
    reference=None,
    kl_coef=0.0,
):
    """RF++ PPO plus A.2 detached residual, summed with caller-normalized weights.

    Reference: martin-marek/score-centering@7c56e9ee losses.py. Its dual-clip
    bound is deliberately absent: RF++ uses the ordinary sign-dependent PPO derivative.
    """
    active = weights > 0
    logp, old, advantages = (x[active] for x in (logp, old.detach(), advantages.detach()))
    head_p, head_q = head_p[active], head_q.detach()[active]
    weights = weights[active]
    ratio = (logp - old).exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("Nonfinite PPO importance ratio")
    surrogate = -torch.minimum(ratio * advantages, ratio.clamp(1 - eps_low, 1 + eps_high) * advantages)
    with torch.no_grad():
        positive = advantages >= 0

        def effective_mass(p, q):
            inside = torch.where(
                positive[:, None],
                p <= q + math.log1p(eps_high),
                p >= q + math.log1p(-eps_low),
            )
            return torch.where(inside, p, -torch.inf).exp()

        p = head_p.exp()
        q_tail = (1 - head_q.exp().sum(-1)).clamp_min(0)
        p_tail = (1 - p.sum(-1)).clamp_min(1e-6)
        rho = q_tail / p_tail
        alpha = effective_mass(torch.zeros_like(rho[:, None]), rho.log()[:, None]).squeeze(-1)
        residual = effective_mass(head_p, head_q) - alpha[:, None] * p
    correction = (residual * head_p).sum(-1)
    loss = surrogate + advantages * correction
    if kl_coef:
        loss = loss + kl_coef * 0.5 * (logp - reference.detach()[active]).square()
    if not torch.isfinite(loss).all():
        raise ValueError("Nonfinite score-centered PPO loss")
    return (loss * weights).sum()


def score_centered_reinforce_loss(logp, old, advantages, head_p, head_q, weights, importance_cap):
    """Top-k TIS + score centering, arXiv:2609.20807 equation 12."""
    active = weights > 0
    logp = logp[active]
    old, advantages = old.detach()[active], advantages.detach()[active]
    head_p, head_q = head_p[active], head_q.detach()[active]
    with torch.no_grad():
        p = head_p.exp()
        q_tail = (1 - head_q.exp().sum(-1)).clamp_min(0)
        p_tail = (1 - p.sum(-1)).clamp_min(1e-6)
        # q*f(p/q) = min(p, cap*q), including a zero-mass sampler tail.
        alpha = torch.minimum(torch.ones_like(p_tail), importance_cap * q_tail / p_tail)
        residual = torch.minimum(p, importance_cap * head_q.exp()) - alpha[:, None] * p
        sampled_weight = (logp - old).clamp_max(math.log(importance_cap)).exp()
    loss = -advantages * (sampled_weight * logp - (residual * head_p).sum(-1))
    if not torch.isfinite(loss).all():
        raise ValueError("Nonfinite score-centered REINFORCE loss")
    return (loss * weights[active]).sum()
