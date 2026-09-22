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
