import pytest
import torch

from skyrl.backends.skyrl_train.score_centering import score_centered_reinforce_loss


@pytest.mark.parametrize("cap", [0.5, 2.0, 100.0])
def test_constant_reward_has_zero_expected_gradient_under_mismatch(cap):
    logits = torch.tensor([1.2, -0.4, 0.3], dtype=torch.float64, requires_grad=True)
    lp = logits.log_softmax(-1)
    q = torch.tensor([0.1, 0.7, 0.2], dtype=torch.float64)
    loss = score_centered_reinforce_loss(lp, q.log(), torch.ones(3), lp.expand(3, 3), q.log().expand(3, 3), q, cap)
    gradient = torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)


def test_generalized_topk_matches_full_vocabulary_tail_model_gradient():
    logits = torch.tensor([0.4, -0.5, 1.3, 0.1], dtype=torch.float64, requires_grad=True)
    lp = logits.log_softmax(-1)
    q_head = torch.tensor([0.5, 0.3], dtype=torch.float64)
    head = torch.tensor([0, 2])
    q = 0.2 * lp.detach().exp() / lp.detach().exp()[[1, 3]].sum()
    q[head] = q_head
    rewards = torch.tensor([0.0, 1.0, -0.5, 0.3], dtype=torch.float64)
    actual = score_centered_reinforce_loss(
        lp, q.log(), rewards, lp[head].expand(4, 2), q_head.log().expand(4, 2), q, 2.0
    )
    p = lp.detach().exp()
    w = (p / q).clamp_max(2)
    correction = (q * w * lp).sum()
    expected = -(q * rewards * (w * lp - correction)).sum()
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
        atol=1e-12,
        rtol=1e-10,
    )


def test_extreme_ratio_is_bounded_before_exponentiation():
    lp = torch.tensor([-1.0], requires_grad=True)
    loss = score_centered_reinforce_loss(
        lp,
        torch.tensor([-10000.0]),
        torch.ones(1),
        lp[:, None],
        torch.tensor([[-1.0]]),
        torch.ones(1),
        2.0,
    )
    loss.backward()
    assert torch.isfinite(lp.grad).all()


@pytest.mark.parametrize("importance_sampling", [False, True])
def test_disabling_centering_preserves_the_constant_reward_bias(importance_sampling):
    q = torch.tensor([0.6, 0.25, 0.15], dtype=torch.float64)
    gradients = []
    for center_scores in (False, True):
        logits = torch.tensor([-0.3, 0.1, 1.0], dtype=torch.float64, requires_grad=True)
        lp = logits.log_softmax(-1)
        loss = score_centered_reinforce_loss(
            lp,
            q.log(),
            torch.ones(3),
            lp.expand(3, -1),
            q.log().expand(3, -1),
            q,
            2.0,
            importance_sampling=importance_sampling,
            center_scores=center_scores,
        )
        gradients.append(torch.autograd.grad(loss, logits)[0])
    assert gradients[0].norm() > 0.1
    torch.testing.assert_close(gradients[1], torch.zeros(3, dtype=torch.float64), atol=1e-12, rtol=0)


def test_k2_gradient_uses_the_fixed_reference_with_zero_advantage():
    logits = torch.tensor([-0.3, 0.1, 1.0], dtype=torch.float64, requires_grad=True)
    lp = logits.log_softmax(-1)
    q = torch.tensor([0.6, 0.25, 0.15], dtype=torch.float64)
    reference = torch.tensor([-0.9, -1.8, -1.2], dtype=torch.float64)
    loss = score_centered_reinforce_loss(
        lp,
        q.log(),
        torch.zeros(3),
        lp.expand(3, -1),
        q.log().expand(3, -1),
        q,
        2.0,
        reference=reference,
        kl_coef=0.001,
    )
    actual = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    expected = torch.autograd.grad((0.0005 * (lp - reference).square() * q).sum(), logits)[0]
    torch.testing.assert_close(actual, expected)


def test_without_ratio_the_expected_gradient_is_the_sampler_advantage_covariance():
    logits = torch.tensor([1.2, -0.4, 0.3, 0.9], dtype=torch.float64, requires_grad=True)
    lp = logits.log_softmax(-1)
    q = torch.tensor([0.1, 0.5, 0.15, 0.25], dtype=torch.float64)
    advantages = torch.tensor([0.0, 1.0, -0.5, 0.3], dtype=torch.float64)
    loss = score_centered_reinforce_loss(
        lp, q.log(), advantages, lp.expand(4, -1), q.log().expand(4, -1), q, 2.0, importance_sampling=False
    )
    expected = -q * (advantages - (q * advantages).sum())
    torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], expected, atol=1e-12, rtol=0)


def test_importance_band_matches_the_masked_rollout_is_correction():
    """The band mode is SkyRL #2244's rollout_is: masked weight f on the sampled token, head and modeled tail."""
    torch.manual_seed(0)
    rows, vocab, k = 16, 40, 6
    base = torch.randn(rows, vocab)
    sampler = torch.log_softmax(base + 0.3 * torch.randn(rows, vocab), -1)
    actions = torch.distributions.Categorical(logits=sampler).sample()
    heads = sampler.topk(k, -1).indices
    advantages = torch.randn(rows)

    def masked(r):
        return torch.where((r > 0.8) & (r < 1.2), r, torch.zeros_like(r))

    def gradient(use_band):
        logits = base.clone().requires_grad_(True)
        p = torch.log_softmax(logits, -1)
        logp, old = p.gather(-1, actions[:, None]).squeeze(-1), sampler.gather(-1, actions[:, None]).squeeze(-1)
        head_p, head_q = p.gather(-1, heads), sampler.gather(-1, heads)
        if use_band:
            loss = score_centered_reinforce_loss(
                logp, old, advantages, head_p, head_q, torch.ones(rows), 2.0, importance_band=(0.2, 0.2)
            )
        else:
            with torch.no_grad():
                q_head, p_head = head_q.exp(), head_p.exp()
                rho = (1 - q_head.sum(-1)).clamp_min(1e-6) / (1 - p_head.sum(-1)).clamp_min(1e-6)
                residual = q_head * masked(p_head / q_head) - (rho * masked(1 / rho))[:, None] * p_head
                weight = masked((logp - old).exp())
            loss = (-(weight * advantages * logp) + advantages * (residual * head_p).sum(-1)).sum()
        loss.backward()
        return logits.grad

    torch.testing.assert_close(gradient(True), gradient(False), rtol=1e-6, atol=1e-7)
