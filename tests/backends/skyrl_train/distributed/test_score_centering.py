"""CPU gradient checks; Gloo covers vocabulary and packed context sharding."""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl.backends.skyrl_train.score_centering import (
    add_score_centering_inputs,
    score_centered_ppo_loss,
)


def model_utils():
    # Only the Megatron import is stubbed; tested collectives are real Gloo ops.
    names = ("megatron", "megatron.core", "megatron.core.parallel_state")
    modules = {name: ModuleType(name) for name in names}
    modules["megatron.core"].parallel_state = modules[names[-1]]
    with patch.dict(sys.modules, modules):
        return importlib.import_module("skyrl.backends.skyrl_train.distributed.megatron.model_utils")


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("k", [2, 5, 32])
def test_reference_gradient(sign, k):
    torch.manual_seed(7)
    logits = torch.randn(1, 3, 41 if k == 32 else 5, dtype=torch.float64, requires_grad=True)
    q = torch.randn_like(logits).softmax(-1)
    ids = q.topk(k).indices
    lp = logits.log_softmax(-1)
    hp, hq = lp.gather(-1, ids), q.log().gather(-1, ids)
    chosen = lp[..., 0]
    old = q[..., 0].log()
    advantages = torch.full_like(chosen, sign)
    mask = torch.tensor([[0.5, 0.0, 0.5]], dtype=torch.float64)
    ref = chosen.detach() - 0.1
    actual, _ = score_centered_ppo_loss(chosen, old, advantages, hp, hq, mask, 0.2, 0.3, ref, 0.001)
    # Reference losses.py top-k estimator, with ordinary PPO (no dual-clip3).
    detached = lp.detach()
    ratio = (chosen - old).exp()
    base = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.3) * advantages)
    pp, qq = hp.detach().exp(), hq.exp()
    rho = (1 - qq.sum(-1)).clamp_min(0) / (1 - pp.sum(-1)).clamp_min(1e-6)
    in_head = (pp <= 1.3 * qq) if sign > 0 else (pp >= 0.8 * qq)
    alpha = (rho >= 1 / 1.3).double() if sign > 0 else (rho <= 1 / 0.8).double()
    center = (pp * in_head * hp).sum(-1)
    center += alpha * ((detached.exp() * lp).sum(-1) - (pp * hp).sum(-1))
    expected = ((base + advantages * center + 0.001 * 0.5 * (chosen - ref).square()) * mask).sum()
    a = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    b = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
    assert a[:, 1].count_nonzero() == 0


def test_full_vocab_expected_score_zero_group_one():
    for sign in [-1.0, 1.0]:
        logits = torch.tensor([0.2, -0.5, 1.0], dtype=torch.float64, requires_grad=True)
        lp = logits.log_softmax(-1)
        q = torch.tensor([0.3, 0.5, 0.2], dtype=torch.float64)
        loss, _ = score_centered_ppo_loss(
            lp, q.log(), torch.full_like(lp, sign), lp.expand(3, 3), q.log().expand(3, 3), q, 0.2, 0.2
        )
        grad = torch.autograd.grad(loss, logits)[0]
        torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-12, rtol=0)


@pytest.mark.parametrize("head_q", [[0.0, -1000.0], [-1000.0, -1000.0]])
def test_tail_extremes_and_mask(head_q):
    hp = torch.tensor([[[-1e-9, -1000.0], [float("nan"), float("nan")]]], requires_grad=True)
    logp = torch.tensor([[-1.0, float("nan")]], requires_grad=True)
    loss, _ = score_centered_ppo_loss(
        logp,
        logp.detach(),
        torch.tensor([[-1.0, 1.0]]),
        hp,
        torch.tensor([[head_q, head_q]]),
        torch.tensor([[1.0, 0.0]]),
        0.2,
        0.2,
    )
    loss.backward()
    assert torch.isfinite(hp.grad).all() and torch.isfinite(logp.grad).all()
    assert hp.grad[0, 1].count_nonzero() == 0


def test_alignment_and_caller_normalized_weights():
    prepared = SimpleNamespace(
        all_loss_fns=["ppo_score_centered"] * 2,
        all_loss_fn_configs=[{"score_centering_k": 2}] * 2,
        all_targets=[[2, 3, 4], [5]],
        all_token_weights=[[1, 0, 1], [1]],
        all_sampling_logprobs=[[-1, 0, -1], [-1]],
        all_advantages=[[1, 0, 1], [-1]],
        all_topk_token_ids=[[0, 1, -99, -99, 2, 3], [4, 5]],
        all_topk_logprobs=[[-1, -2, float("nan"), float("nan"), -1, -2], [-1, -2]],
        all_reference_logprobs=[[], []],
        request_batch_slices=[("r", "m", 0, 2)],
    )
    weights = torch.tensor([[0.25, 0, 0.25], [0, 0, 0.5]])
    batch = {"loss_mask": weights.clone()}
    add_score_centering_inputs(batch, prepared, 3)
    # Weights stay as the client normalized them across every request of its forward_backward.
    torch.testing.assert_close(batch["loss_mask"], weights)
    assert batch["topk_token_ids"][1].tolist() == [[0, 0], [0, 0], [4, 5]]
    assert torch.isfinite(batch["topk_logprobs"]).all()


def distributed_worker(rank, size, init_file, mode):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=size)
    utils = model_utils()
    torch.manual_seed(37)
    h = torch.randn(1, 8, 4, requires_grad=True)
    w = torch.randn(12, 4, requires_grad=True)
    ids = torch.randint(0, 12, (1, 8, 4))
    ids[..., -1] = ids[..., 0]  # sampled token may also be in its top-k head
    expected = (h @ w.T).log_softmax(-1).gather(-1, ids)
    upstream = torch.randn_like(expected)
    if mode == "tp":
        local_w = w.detach().chunk(size)[rank].clone().requires_grad_()
        local_h = h.detach().clone().requires_grad_()
        actual = utils.FusedLinearChunkedDistributedLogprob.apply(
            local_h, local_w, ids, rank * 6, (rank + 1) * 6, 3, dist.group.WORLD, False
        )
        (actual * upstream).sum().backward()
        (expected * upstream).sum().backward()
        dist.all_reduce(local_h.grad)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(local_h.grad, h.grad, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(local_w.grad, w.grad.chunk(size)[rank], atol=3e-6, rtol=3e-6)
    else:
        # Two four-token packed segments, split into mirrored CP chunks.
        cu = torch.tensor([0, 4, 8], dtype=torch.int32)
        local_idx = torch.tensor([rank, 3 - rank, 4 + rank, 7 - rank])
        local_h = h.detach()[:, local_idx].clone().requires_grad_()
        local_w = w.detach().clone().requires_grad_()
        groups = [dist.new_group([r]) for r in range(size)]
        actual = utils.from_parallel_hidden_to_logprobs_packed_sequences(
            local_h,
            local_w,
            ids,
            cu,
            4,
            0,
            12,
            groups[rank],
            cp_group=dist.group.WORLD,
            chunk_size=2,
            attention_mask=torch.ones(2, 4, dtype=torch.bool),
        )
        hp = h.reshape(2, 4, 4)
        targets = ids.reshape(2, 4, 4)[:, 1:]
        expected = (hp[:, :-1] @ w.T).log_softmax(-1).gather(-1, targets)
        upstream = torch.randn_like(expected)
        (actual * upstream).sum().backward()
        (expected * upstream).sum().backward()
        dist.all_reduce(local_w.grad)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(local_h.grad, h.grad[:, local_idx], atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(local_w.grad, w.grad, atol=3e-6, rtol=3e-6)
    dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["tp", "cp"])
def test_two_rank_gradient_parity(tmp_path, mode):
    mp.spawn(distributed_worker, args=(2, str(tmp_path / "init"), mode), nprocs=2, join=True)


def test_k2_and_token_weights_are_invariant_to_microbatch_split():
    torch.manual_seed(31)
    logits = torch.randn(2, 3, 5, requires_grad=True)
    lp = logits.log_softmax(-1)
    q = torch.randn_like(lp).log_softmax(-1)
    heads = q.topk(2).indices
    kwargs = dict(eps_low=0.2, eps_high=0.2, kl_coef=0.001)
    args = [
        lp[..., 0],
        q[..., 0],
        torch.tensor([[1.0, 1.0, 0.0], [-1.0, 0.0, -1.0]]),
        lp.gather(-1, heads),
        q.gather(-1, heads),
        torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]) / 4,
    ]
    reference = lp[..., 0].detach() - 0.4
    full, _ = score_centered_ppo_loss(*args, reference=reference, **kwargs)
    split = sum(
        score_centered_ppo_loss(*(x[i : i + 1] for x in args), reference=reference[i : i + 1], **kwargs)[0]
        for i in range(2)
    )
    torch.testing.assert_close(full, split)
    torch.testing.assert_close(
        torch.autograd.grad(full, logits, retain_graph=True)[0], torch.autograd.grad(split, logits)[0]
    )


def test_nonfinite_importance_ratio_aborts_before_backward():
    lp = torch.tensor([-1.0], requires_grad=True)
    with pytest.raises(ValueError, match="importance ratio"):
        score_centered_ppo_loss(
            lp, torch.tensor([-1000.0]), torch.ones(1), lp[:, None], lp.detach()[:, None], torch.ones(1), 0.2, 0.2
        )


def test_padding_microbatch_has_zero_gradient():
    lp = torch.randn(1, 3, requires_grad=True)
    heads = torch.randn(1, 3, 2, requires_grad=True)
    loss, _ = score_centered_ppo_loss(
        lp, lp.detach(), torch.ones_like(lp), heads, heads.detach(), torch.zeros_like(lp), 0.2, 0.2
    )
    loss.backward()
    assert loss.item() == 0 and lp.grad.count_nonzero() == 0 and heads.grad.count_nonzero() == 0


def test_unpacked_heads_follow_shifted_targets_with_left_padding(tmp_path):
    utils = model_utils()
    dist.init_process_group("gloo", init_method=f'file://{tmp_path / "single"}', rank=0, world_size=1)
    try:
        torch.manual_seed(43)
        hidden = torch.randn(2, 6, 4, requires_grad=True)
        weight = torch.randn(11, 4, requires_grad=True)
        targets = torch.randint(0, 11, (2, 6, 4))
        targets[0, :2] = 0
        actual = utils.from_parallel_hidden_to_logprobs(hidden, weight, targets, 0, 11, dist.group.WORLD, chunk_size=2)
        expected = (hidden @ weight.T).log_softmax(-1)[:, :-1].gather(-1, targets[:, 1:])
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        mask = torch.ones_like(actual)
        mask[0, :2] = 0
        a = torch.autograd.grad((actual * mask).sum(), (hidden, weight), retain_graph=True)
        b = torch.autograd.grad((expected * mask).sum(), (hidden, weight))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=3e-6, rtol=3e-6)
    finally:
        dist.destroy_process_group()


def test_center_scores_false_is_plain_ppo():
    torch.manual_seed(11)
    logits = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
    lp = logits.log_softmax(-1)
    q = torch.randn_like(lp).log_softmax(-1)
    heads = q.topk(3).indices
    chosen = lp[..., 0]
    old = chosen.detach() + torch.tensor([[0.5, -0.5, 0.0, 0.1], [-0.4, 0.3, 0.2, -0.1]], dtype=torch.float64)
    advantages = torch.tensor([[1.0, -1.0, 0.5, -2.0], [-0.5, 2.0, 1.0, -1.0]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]], dtype=torch.float64) / 6
    ref = chosen.detach() - 0.2
    args = (chosen, old, advantages, lp.gather(-1, heads), q.gather(-1, heads), mask, 0.2, 0.28, ref, 0.001)
    off, metrics = score_centered_ppo_loss(*args, center_scores=False)
    on, _ = score_centered_ppo_loss(*args)
    ratio = (chosen - old).exp()
    ppo = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.28) * advantages)
    expected = ((ppo + 0.001 * 0.5 * (chosen - ref).square()) * mask).sum()
    a = torch.autograd.grad(off, logits, retain_graph=True)[0]
    b = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    torch.testing.assert_close(off, expected, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
    assert not torch.allclose(torch.autograd.grad(on, logits)[0], b)
    assert metrics["score_centering/correction_l1_sum"] == 0


def test_metrics_count_zero_gradient_tokens_and_sum_head_statistics():
    nan = float("nan")
    logp = torch.tensor([[0.5, 0.1, nan]], dtype=torch.float64).log().requires_grad_()
    old = torch.tensor([[0.25, 0.2, 0.3]], dtype=torch.float64).log()
    advantages = torch.tensor([[1.0, -2.0, 1.0]], dtype=torch.float64)
    p = torch.tensor([[[0.5, 0.2], [0.1, 0.6], [nan, nan]]], dtype=torch.float64)
    q = torch.tensor([[[0.25, 0.25], [0.2, 0.3], [0.5, 0.5]]], dtype=torch.float64)
    weights = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64)
    args = (logp, old, advantages, p.log(), q.log(), weights, 0.2, 0.2)
    _, on = score_centered_ppo_loss(*args)
    _, off = score_centered_ppo_loss(*args, center_scores=False)
    # Ratios are 2 with A > 0 and 0.5 with A < 0: both outside the clip range.
    assert on["score_centering/action_tokens"] == 2
    assert on["score_centering/zero_grad_tokens"] == 2
    # Residuals: [-0.5, 0] (only the second head token is in range, alpha = 1) and
    # [0, 0.6] (only the second head token is in range, rho = 5/3 > 1/0.8 so alpha = 0).
    assert on["score_centering/correction_l1_sum"] == pytest.approx(0.5 * 1.0 + 0.6 * 2.0)
    assert on["score_centering/q_tail_sum"] == pytest.approx(1.0)
    head_kl = (q[0, :2] * (q[0, :2] / p[0, :2]).log()).sum().item()
    assert on["score_centering/head_kl_sum"] == pytest.approx(head_kl)
    assert off == {**on, "score_centering/correction_l1_sum": 0.0}
    parts = [score_centered_ppo_loss(*(x[:, s] for x in args[:6]), 0.2, 0.2)[1] for s in (slice(0, 1), slice(1, 3))]
    for key, value in on.items():
        assert value == pytest.approx(sum(part[key] for part in parts))


def test_center_scores_config_must_be_boolean():
    prepared = SimpleNamespace(
        all_loss_fns=["ppo_score_centered"],
        all_loss_fn_configs=[{"score_centering_k": 1, "center_scores": 0.5}],
    )
    with pytest.raises(ValueError, match="center_scores"):
        add_score_centering_inputs({}, prepared, 1)
