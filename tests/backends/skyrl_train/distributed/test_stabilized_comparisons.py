"""Ratio-free score-centered REINFORCE on recorded comparison histograms is the stabilized_reinforce candidate.

Candidate at kl_coeff=0: loss = -A (z_a - sum_v f_v z_v), f the histogram of K draws from the sampler's law.
Heads go through the vLLM carrier columns, vLLM's per-position dict and the API server's decoding.
"""

import itertools
import math

import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.patches.vllm.patch_stabilized_comparisons import (
    carrier_columns,
)
from skyrl.backends.skyrl_train.score_centering import score_centered_reinforce_loss
from skyrl.backends.utils import (
    COMPARISON_PAD_LOGPROB,
    convert_comparison_heads,
    split_comparison_carriers,
)
from skyrl.tinker.decode_heads import hash_tokens, subsampled_comparison_heads

K = 16
# Padding needs K distinct ids, so the learner vocabulary is larger than the sampler's support.
VOCAB = 20


def recorded_heads(sampled: int, draws: list[int], top_k: tuple[int, ...] = (0, 1, 2)) -> tuple[np.ndarray, np.ndarray]:
    """The [K] head ids and float32 logprobs the API server records for one generated position."""
    carrier_ids, carrier_values = carrier_columns(torch.tensor([draws]), torch.tensor([[sampled, *top_k]]))
    columns = zip(
        [sampled, *top_k, *carrier_ids[0].tolist()],
        [-0.5, *[-1.0] * len(top_k), *carrier_values[0].tolist()],
    )
    # vLLM folds a position's columns into a dict keyed by token id; a later column overwrites an earlier one.
    top_logprobs = {f"token_id:{token}": logprob for token, logprob in columns}
    _, row_draws = split_comparison_carriers([top_logprobs])
    head_ids, head_logprobs = convert_comparison_heads(row_draws, K)
    return head_ids[0], head_logprobs[0]


def float64_heads(head_logprobs: np.ndarray, k: int = K) -> np.ndarray:
    """log(count / k) in float64 from the recorded float32 values; padding stays at COMPARISON_PAD_LOGPROB."""
    counts = np.round(k * np.exp(head_logprobs.astype(np.float64)))
    return np.where(counts > 0, np.log(np.maximum(counts, 1) / k), COMPARISON_PAD_LOGPROB)


def kernel_loss(z, sampled, heads, advantages, weights, dtype=torch.float64, nll_value=False):
    """Sum of the ratio-free kernel over rows that share the learner logits z."""
    head_ids = torch.as_tensor(np.stack([ids for ids, _ in heads]), dtype=torch.long)
    head_q = torch.as_tensor(np.stack([q for _, q in heads]), dtype=dtype)
    lp = z.log_softmax(-1).expand(len(heads), -1)
    return score_centered_reinforce_loss(
        lp.gather(-1, sampled[:, None]).squeeze(-1),
        torch.zeros(len(heads), dtype=dtype),
        advantages,
        lp.gather(-1, head_ids),
        head_q,
        weights,
        2.0,
        importance_sampling=False,
        nll_value=nll_value,
    )


@pytest.mark.parametrize(
    ("sampled", "draws"),
    [
        (1, [1, 1, 1, 3, 0, 0, 4, 1, 3, 3, 1, 1, 0, 2, 1, 1]),
        (2, [0] * 9 + [3] * 7),
        (4, [4] * K),
    ],
    ids=["sampled-drawn", "sampled-not-drawn", "all-draws-sampled"],
)
def test_kernel_on_recorded_heads_is_the_candidate(sampled, draws):
    z = torch.arange(VOCAB, dtype=torch.float64).sin().requires_grad_()
    advantage = 0.7
    head_ids, head_logprobs = recorded_heads(sampled, draws)
    counts = np.bincount(draws, minlength=VOCAB)
    drawn = head_logprobs > COMPARISON_PAD_LOGPROB
    assert len(set(head_ids.tolist())) == K and set(head_ids[drawn].tolist()) == set(np.flatnonzero(counts).tolist())
    np.testing.assert_array_equal(np.round(K * np.exp(head_logprobs[drawn])), counts[head_ids[drawn]])

    inputs = (z, torch.tensor([sampled]), [(head_ids, float64_heads(head_logprobs))])
    advantages = torch.tensor([advantage], dtype=torch.float64)
    loss = kernel_loss(*inputs, advantages, torch.ones(1))
    f = torch.tensor(counts / K, dtype=torch.float64)
    expected = -advantage * (z[sampled] - (f * z).sum())
    gradient = torch.autograd.grad(expected, z)[0]
    torch.testing.assert_close(loss, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(loss, z)[0], gradient, atol=1e-12, rtol=0)

    # Native stabilized_reinforce's forward value is the weighted NLL; the gradient is unchanged.
    nll = kernel_loss(*inputs, advantages, torch.ones(1), nll_value=True)
    torch.testing.assert_close(nll, -advantage * z.log_softmax(-1)[sampled], atol=1e-12, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(nll, z)[0], gradient, atol=1e-12, rtol=0)


def test_float32_head_rounding_moves_the_gradient_by_the_rounding_only():
    # The heads hold all but ~5e-8 of the learner's mass, so the kernel floors the learner tail at 1e-6.
    z = torch.tensor([10.0, 8.0] + [-8.0] * (VOCAB - 2), requires_grad=True)
    draws = [0] * 13 + [1] * 3
    head_ids, head_logprobs = recorded_heads(1, draws)
    # A 2e-7 relative error in exp(head_q), about two float32 ulps, leaves sampler tail mass q_tail > 0.
    drawn = head_logprobs > COMPARISON_PAD_LOGPROB
    head_logprobs[drawn] -= np.float32(2e-7)
    q_tail = 1 - torch.tensor(head_logprobs).exp().sum()
    assert q_tail / 1e-6 > 0.1  # the residual subtracts this multiple of the learner's head probabilities
    loss = kernel_loss(
        z, torch.tensor([1]), [(head_ids, head_logprobs)], torch.tensor([1.0]), torch.ones(1), dtype=torch.float32
    )
    expected = -(torch.eye(VOCAB)[1] - torch.bincount(torch.tensor(draws), minlength=VOCAB) / K)
    torch.testing.assert_close(torch.autograd.grad(loss, z)[0], expected, atol=1e-6, rtol=0)


def expected_gradient(p: np.ndarray, advantages: list[float], k: int = K, leave_in: bool = False) -> torch.Tensor:
    """E over a ~ p and the k-draw heads a lookup builds of the kernel gradient, by exact enumeration.

    With leave_in the lookup takes k - 1 i.i.d. draws and the sampled token.
    """
    z = torch.arange(VOCAB, dtype=torch.float64).cos().requires_grad_()
    sampled, heads, weights = [], [], []
    for a in range(len(p)):
        for draws in itertools.combinations_with_replacement(range(len(p)), k - leave_in):
            counts = np.bincount(draws, minlength=len(p))
            arrangements = math.factorial(len(draws)) / np.prod([math.factorial(c) for c in counts])
            if k == K and not leave_in:
                head_ids, head_logprobs = recorded_heads(a, list(draws))
            else:
                recorded = np.asarray([draws], dtype=np.int32)
                head_ids, head_logprobs = subsampled_comparison_heads(hash_tokens(draws), recorded, [a], k, leave_in)
                head_ids, head_logprobs = head_ids[0], head_logprobs[0]
            sampled.append(a)
            heads.append((head_ids, float64_heads(head_logprobs, k)))
            weights.append(p[a] * arrangements * np.prod(p**counts))
    sampled = torch.tensor(sampled)
    weights = torch.tensor(weights, dtype=torch.float64)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-12)
    loss = kernel_loss(z, sampled, heads, torch.tensor(advantages, dtype=torch.float64)[sampled], weights)
    return torch.autograd.grad(loss, z)[0]


# A sampler law after temperature 0.7, as vLLM's processed distribution would be.
P = torch.tensor([1.0, 0.2, -0.5], dtype=torch.float64).div(0.7).softmax(-1).numpy()


@pytest.mark.parametrize(("k", "leave_in"), [(K, False), (4, False), (4, True)])
def test_constant_advantage_has_zero_expected_gradient(k, leave_in):
    gradient = expected_gradient(P, [1.0, 1.0, 1.0], k, leave_in)
    torch.testing.assert_close(gradient, torch.zeros(VOCAB, dtype=torch.float64), atol=1e-12, rtol=0)


def test_expected_gradient_is_the_sampler_advantage_covariance():
    advantages = [1.0, -0.5, 0.3]
    # The mean is the descent direction of Cov_p(A, e_a).
    mean = (P * advantages).sum()
    covariance = torch.zeros(VOCAB, dtype=torch.float64)
    covariance[: len(P)] = torch.tensor(P * (np.array(advantages) - mean))
    torch.testing.assert_close(expected_gradient(P, advantages), -covariance, atol=1e-12, rtol=0)
    torch.testing.assert_close(expected_gradient(P, advantages, 4), -covariance, atol=1e-12, rtol=0)


@pytest.mark.parametrize("k", [4, K])
def test_leave_in_mean_is_k_minus_one_over_k_of_the_independent_mean(k):
    advantages = [1.0, -0.5, 0.3]
    independent = expected_gradient(P, advantages, k)
    torch.testing.assert_close(
        expected_gradient(P, advantages, k, leave_in=True), independent * (k - 1) / k, atol=1e-12, rtol=0
    )
