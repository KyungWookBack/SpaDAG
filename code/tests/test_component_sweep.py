"""Composition order, residual observer and checkpoint kernel contract."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from spadag.cli import DEFAULT_M_REV, _checkpoint_signature, _resolved_m_rev, parse_args, sweep_shared_gp_pg
from spadag.component_reversal import MeshComponents, attempt_component_reversal
from spadag.mcmc_data import load_inference_data
from spadag.mcmc_steps import SamplerConfig, init_state, _alpha_at_obs, _parent_pred, update_sigma_b2
from spadag.shared_gp_pg import build_gp_decomposition, init_shared_gp_state
from tests.toy_data_fixture import temporary_toy_bundle


def test_composed_sweep_matches_five_explicit_attempts():
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        cfg = SamplerConfig()
        rng = np.random.default_rng(9810)
        st = init_state(data, cfg, rng)
        shared = init_shared_gp_state(data, st, mu_init=0.,
                    decomposition=build_gp_decomposition(data))
        other, other_shared, other_rng = deepcopy(st), deepcopy(shared), deepcopy(rng)
        kwargs = dict(common_gp_var=1., edge_gp_var=1., a_tau=3., b_tau=2., a_sigma0=2.,
                      b_sigma0=1., mu_prior_mean=0., mu_prior_sd=2.)
        omega = lambda eta: np.full(np.shape(eta), .25)
        mesh = MeshComponents(data)
        for _ in range(3):
            ordinary = sweep_shared_gp_pg(data, st, shared, cfg, rng, omega, **kwargs)
            events = [attempt_component_reversal(data, st, cfg, rng, mesh) for _ in range(5)]
            composed = sweep_shared_gp_pg(data, other, other_shared, cfg, other_rng, omega,
                                         **kwargs, m_rev=5, component_mesh=mesh)
            assert ordinary[1:] == composed[1:]
            assert events == other._component_events
            for name in ('b', 'gamma', 'sigma2', 'sigma_b2', 'theta', 'a0', 'a_spatial', 'pi'):
                np.testing.assert_array_equal(getattr(st, name), getattr(other, name))
            assert rng.bit_generator.state == other_rng.bit_generator.state
            rss = np.array([np.sum((data.X[:, j] - _alpha_at_obs(data, other, j)
                                  - _parent_pred(data, other, j))**2) for j in range(data.p)])
            np.testing.assert_allclose(other._rss, rss)
            expected = -.5 * np.sum(data.n * np.log(2*np.pi*other.sigma2) + rss / other.sigma2)
            assert other._log_likelihood == expected
            kwargs.update(edge_gp_var=ordinary[1], common_gp_var=ordinary[2])


def test_signature_distinguishes_kernel_and_free_prior():
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        args = parse_args(['--data', str(bundle), '--trace-dir', 'trace'])
        assert args.m_rev is None and args.a_b == args.b_b == 1.
        latest = _checkpoint_signature(data, args)
        args.m_rev = 0
        baseline = _checkpoint_signature(data, args)
        assert latest != baseline and baseline['settings']['m_rev'] == 0
        args.m_rev = latest['settings']['m_rev'] + 1
        assert _checkpoint_signature(data, args) != latest
        cfg = SamplerConfig(a_b=1., b_b=1.)
        rng = np.random.default_rng(4)
        st = init_state(data, cfg, rng)
        st.gamma.fill(0)
        update_sigma_b2(data, st, cfg, np.random.default_rng(19))
        assert st.sigma_b2 == 1 / np.random.default_rng(19).gamma(1., scale=1.)
        SamplerConfig(a_b=.5, b_b=3.).validate()


def test_default_is_fifty_and_explicit_override():
    assert DEFAULT_M_REV == 50
    assert _resolved_m_rev(SimpleNamespace(m_rev=None)) == 50
    assert _resolved_m_rev(SimpleNamespace(m_rev=0)) == 0
    assert _resolved_m_rev(SimpleNamespace(m_rev=7)) == 7


def test_auto_matches_explicit_count_and_checkpoint(tmp_path):
    from tests.test_compact_storage import args_for, run, same_state
    from spadag.compact_runner import read_checkpoint
    with temporary_toy_bundle() as bundle:
        data = load_inference_data(bundle)
        auto = args_for(bundle, tmp_path/'auto', iters=7)
        explicit = args_for(bundle, tmp_path/'explicit', iters=7,
                            extra=['--m-rev', str(DEFAULT_M_REV)])
        assert _checkpoint_signature(data, auto) == _checkpoint_signature(data, explicit)
        a, b = run(data, auto), run(data, explicit)
        assert auto.m_rev is None  # The API namespace can be reused for other p.
        ca, cb = read_checkpoint(auto.checkpoint), read_checkpoint(explicit.checkpoint)
        same_state(ca, cb)
        assert ca['signature'] == cb['signature']
        assert a['m_rev'] == b['m_rev'] == DEFAULT_M_REV
        np.testing.assert_array_equal(a['edge_prob'], b['edge_prob'])
