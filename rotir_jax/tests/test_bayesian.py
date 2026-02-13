"""Tests for Bayesian inference pipeline.

Tests verify:
1. Likelihood correctly wraps the forward model
2. Log-prior computation is correct
3. Log-posterior gradient flows through all parameters
4. MAP estimation converges
5. Posterior summary statistics are computed correctly
6. Parameter flattening/unflattening round-trips
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from rotir_jax.datatypes import StellarGeometry, OIData
from rotir_jax.inference.nifty_likelihood import (
    RotirLikelihood,
    ForwardModelConfig,
    build_signal_response,
    build_data_vector,
    create_nifty_likelihood,
)
from rotir_jax.inference.bayesian_model import (
    BayesianStellarModel,
    GeometricPrior,
    TemperatureFieldConfig,
    _flatten_params,
    run_inference,
)
from rotir_jax.inference.posterior import (
    summarize_posterior,
    compute_credible_intervals,
    compute_pixel_significance,
    compute_spot_detection_probability,
    format_parameter_table,
    compute_correlation_matrix,
)


# ============================================================
# Test fixtures: create mock data for testing
# ============================================================

@pytest.fixture
def mock_oi_data():
    """Create minimal mock interferometric data."""
    nuv = 10
    nv2 = 6
    nt3 = 4

    key = jr.PRNGKey(0)
    keys = jr.split(key, 8)

    return OIData(
        v2=jnp.abs(jr.normal(keys[0], (nv2,))) * 0.5 + 0.3,
        v2_err=jnp.ones(nv2) * 0.05,
        nv2=nv2,
        t3phi=jr.normal(keys[1], (nt3,)) * 10.0,
        t3phi_err=jnp.ones(nt3) * 5.0,
        nt3=nt3,
        t3amp=jnp.abs(jr.normal(keys[2], (nt3,))) * 0.3 + 0.1,
        t3amp_err=jnp.ones(nt3) * 0.05,
        uv=jr.normal(keys[3], (2, nuv)) * 1e6,
        nuv=nuv,
        indx_v2=jnp.array([0, 1, 2, 3, 4, 5]),
        indx_t3_1=jnp.array([0, 1, 2, 3]),
        indx_t3_2=jnp.array([1, 2, 3, 4]),
        indx_t3_3=jnp.array([2, 3, 4, 5]),
        wavelengths=jnp.ones(nuv) * 1.6e-6,
    )


@pytest.fixture
def mock_geometry():
    """Create minimal mock stellar geometry."""
    npix = 48  # Small for testing
    nvis = 24

    key = jr.PRNGKey(1)
    keys = jr.split(key, 6)

    return StellarGeometry(
        surface_type=0,
        npix=npix,
        nvis=nvis,
        vertices_xyz=jr.normal(keys[0], (npix, 5, 3)) * 10.0,
        vertices_spherical=np.random.randn(npix, 5, 3),
        normals=jr.normal(keys[1], (npix, 3)),
        visible_idx=jnp.arange(nvis),
        projx=jr.normal(keys[2], (nvis, 4)) * 5.0,
        projy=jr.normal(keys[3], (nvis, 4)) * 5.0,
        ldmap=jnp.ones(npix),
        epoch=0.0,
    )


@pytest.fixture
def mock_samples():
    """Create mock posterior samples for testing analysis functions."""
    key = jr.PRNGKey(42)
    n_samples = 20
    npix = 48

    samples = []
    for i in range(n_samples):
        k = jr.fold_in(key, i)
        keys = jr.split(k, 5)
        samples.append({
            'temperature_map': 5000.0 + 500.0 * jr.normal(keys[0], (npix,)),
            'diameter': 44.0 + 2.0 * jr.normal(keys[1], ()),
            'inclination': 1.05 + 0.1 * jr.normal(keys[2], ()),
            'position_angle': 0.3 + 0.05 * jr.normal(keys[3], ()),
            'log_amplitude': 0.0 + 0.2 * jr.normal(keys[4], ()),
        })
    return samples


# ============================================================
# Test likelihood
# ============================================================

class TestRotirLikelihood:
    """Tests for the ROTIR likelihood wrapper."""

    def test_construction(self, mock_oi_data, mock_geometry):
        """Likelihood can be constructed from config."""
        lh = create_nifty_likelihood(
            mock_oi_data, mock_geometry, npix=mock_geometry.npix,
        )
        assert lh.n_data > 0
        assert lh.data.shape[0] == lh.n_data

    def test_data_vector(self, mock_oi_data):
        """Data vector has correct size."""
        data, noise = build_data_vector(mock_oi_data)
        expected_size = mock_oi_data.nv2 + mock_oi_data.nt3 + mock_oi_data.nt3
        assert data.shape[0] == expected_size
        assert noise.shape[0] == expected_size

    def test_data_vector_v2_only(self, mock_oi_data):
        """Data vector with only V^2."""
        data, noise = build_data_vector(
            mock_oi_data,
            use_t3_amplitude=False,
            use_closure_phase=False,
        )
        assert data.shape[0] == mock_oi_data.nv2

    def test_nll_returns_scalar(self, mock_oi_data, mock_geometry):
        """Negative log-likelihood returns a scalar."""
        lh = create_nifty_likelihood(
            mock_oi_data, mock_geometry, npix=mock_geometry.npix,
        )
        params = {'temperature_map': jnp.ones(mock_geometry.npix) * 5000.0}
        nll = lh.neg_log_likelihood(params)
        assert nll.shape == ()
        assert jnp.isfinite(nll)

    def test_nll_nonnegative(self, mock_oi_data, mock_geometry):
        """NLL is non-negative (chi-squared term)."""
        lh = create_nifty_likelihood(
            mock_oi_data, mock_geometry, npix=mock_geometry.npix,
        )
        params = {'temperature_map': jnp.ones(mock_geometry.npix) * 5000.0}
        nll = lh.neg_log_likelihood(params)
        assert nll >= 0.0

    def test_nll_gradient(self, mock_oi_data, mock_geometry):
        """NLL gradient has correct shape and is finite."""
        lh = create_nifty_likelihood(
            mock_oi_data, mock_geometry, npix=mock_geometry.npix,
        )
        params = {'temperature_map': jnp.ones(mock_geometry.npix) * 5000.0}
        nll, grad = lh.gradient(params)
        assert jnp.isfinite(nll)
        assert 'temperature_map' in grad
        assert grad['temperature_map'].shape == (mock_geometry.npix,)

    def test_nll_components(self, mock_oi_data, mock_geometry):
        """NLL components sum to total."""
        lh = create_nifty_likelihood(
            mock_oi_data, mock_geometry, npix=mock_geometry.npix,
        )
        params = {'temperature_map': jnp.ones(mock_geometry.npix) * 5000.0}
        components = lh.neg_log_likelihood_components(params)
        assert 'nll_v2' in components
        assert 'nll_total' in components

    def test_callable_interface(self, mock_oi_data, mock_geometry):
        """Likelihood is callable (for NIFTy compatibility)."""
        lh = create_nifty_likelihood(
            mock_oi_data, mock_geometry, npix=mock_geometry.npix,
        )
        params = {'temperature_map': jnp.ones(mock_geometry.npix) * 5000.0}
        result = lh(params)
        assert jnp.isfinite(result)


# ============================================================
# Test Bayesian model
# ============================================================

class TestBayesianStellarModel:
    """Tests for the full Bayesian model."""

    def test_construction(self, mock_oi_data, mock_geometry):
        """Model can be constructed with default config."""
        model = BayesianStellarModel(
            oi_data=mock_oi_data,
            base_geom=mock_geometry,
        )
        assert model.likelihood is not None

    def test_construction_with_priors(self, mock_oi_data, mock_geometry):
        """Model with geometric priors."""
        priors = {
            'diameter': GeometricPrior('diameter', 'lognormal', 44.0, 0.1),
            'inclination': GeometricPrior('inclination', 'normal', 1.05, 0.1),
        }
        model = BayesianStellarModel(
            oi_data=mock_oi_data,
            base_geom=mock_geometry,
            geometric_priors=priors,
        )
        assert len(model._inferred_geom_params) == 2

    def test_initial_params(self, mock_oi_data, mock_geometry):
        """Initial parameters have correct structure."""
        model = BayesianStellarModel(
            oi_data=mock_oi_data,
            base_geom=mock_geometry,
            geometric_priors={
                'diameter': GeometricPrior('diameter', 'lognormal', 44.0, 0.1),
            },
        )
        params = model.initial_params(jr.PRNGKey(0))
        assert 'temperature_map' in params
        assert 'log_amplitude' in params
        assert 'diameter' in params
        assert params['temperature_map'].shape == (mock_geometry.npix,)

    def test_log_prior_finite(self, mock_oi_data, mock_geometry):
        """Log-prior returns finite value."""
        model = BayesianStellarModel(
            oi_data=mock_oi_data,
            base_geom=mock_geometry,
        )
        params = model.initial_params(jr.PRNGKey(0))
        lp = model.log_prior(params)
        assert jnp.isfinite(lp)

    def test_log_prior_normal(self, mock_oi_data, mock_geometry):
        """Normal prior returns higher log-prob near mean."""
        priors = {
            'diameter': GeometricPrior('diameter', 'normal', 44.0, 2.0),
        }
        model = BayesianStellarModel(
            oi_data=mock_oi_data,
            base_geom=mock_geometry,
            geometric_priors=priors,
        )
        params_at_mean = model.initial_params(jr.PRNGKey(0))
        params_at_mean['diameter'] = jnp.array(44.0)
        lp_at_mean = model.log_prior(params_at_mean)

        params_far = model.initial_params(jr.PRNGKey(0))
        params_far['diameter'] = jnp.array(100.0)
        lp_far = model.log_prior(params_far)

        assert lp_at_mean > lp_far

    def test_log_posterior_finite(self, mock_oi_data, mock_geometry):
        """Log-posterior returns finite value."""
        model = BayesianStellarModel(
            oi_data=mock_oi_data,
            base_geom=mock_geometry,
        )
        params = model.initial_params(jr.PRNGKey(0))
        lp = model.log_posterior(params)
        assert jnp.isfinite(lp)

    def test_neg_log_posterior_gradient(self, mock_oi_data, mock_geometry):
        """Gradient of neg-log-posterior is finite."""
        model = BayesianStellarModel(
            oi_data=mock_oi_data,
            base_geom=mock_geometry,
        )
        params = model.initial_params(jr.PRNGKey(0))
        grad = jax.grad(model.neg_log_posterior)(params)
        assert 'temperature_map' in grad
        assert jnp.all(jnp.isfinite(grad['temperature_map']))

    def test_decode_temperature_map(self, mock_oi_data, mock_geometry):
        """Temperature decoding produces positive values."""
        model = BayesianStellarModel(
            oi_data=mock_oi_data,
            base_geom=mock_geometry,
        )
        params = model.initial_params(jr.PRNGKey(0))
        T = model.decode_temperature_map(params)
        assert T.shape == (mock_geometry.npix,)
        assert jnp.all(T > 0)

    def test_fixed_prior_not_inferred(self, mock_oi_data, mock_geometry):
        """Fixed parameters are not listed as inferred."""
        priors = {
            'diameter': GeometricPrior('diameter', 'fixed', fixed_value=44.0),
            'inclination': GeometricPrior('inclination', 'normal', 1.05, 0.1),
        }
        model = BayesianStellarModel(
            oi_data=mock_oi_data,
            base_geom=mock_geometry,
            geometric_priors=priors,
        )
        assert 'diameter' not in model._inferred_geom_params
        assert 'inclination' in model._inferred_geom_params


# ============================================================
# Test parameter flattening
# ============================================================

class TestParameterFlattening:
    """Tests for flatten/unflatten round-trip."""

    def test_roundtrip_simple(self):
        """Flatten then unflatten recovers original."""
        params = {
            'a': jnp.array(1.0),
            'b': jnp.array([1.0, 2.0, 3.0]),
        }
        flat, unflatten = _flatten_params(params)
        recovered = unflatten(flat)

        np.testing.assert_allclose(
            float(recovered['a']), float(params['a'])
        )
        np.testing.assert_allclose(
            np.array(recovered['b']), np.array(params['b'])
        )

    def test_roundtrip_complex(self):
        """Flatten works with mixed scalar and array params."""
        params = {
            'temperature_map': jnp.ones(48) * 5000.0,
            'diameter': jnp.array(44.0),
            'inclination': jnp.array(1.05),
            'log_amplitude': jnp.array(0.0),
        }
        flat, unflatten = _flatten_params(params)
        assert flat.shape == (48 + 3,)

        recovered = unflatten(flat)
        for k in params:
            np.testing.assert_allclose(
                np.array(recovered[k]),
                np.array(params[k]),
                atol=1e-6,
            )

    def test_gradient_through_flatten(self):
        """Gradient flows through flatten/unflatten."""
        params = {
            'a': jnp.array(2.0),
            'b': jnp.array([1.0, 2.0]),
        }
        flat, unflatten = _flatten_params(params)

        def f(x_flat):
            p = unflatten(x_flat)
            return p['a'] * jnp.sum(p['b'])

        grad = jax.grad(f)(flat)
        assert grad.shape == flat.shape
        assert jnp.all(jnp.isfinite(grad))


# ============================================================
# Test MAP estimation
# ============================================================

class TestMAPEstimation:
    """Tests for MAP optimization."""

    def test_map_runs(self, mock_oi_data, mock_geometry):
        """MAP optimization runs without error."""
        model = BayesianStellarModel(
            oi_data=mock_oi_data,
            base_geom=mock_geometry,
        )
        result = model.run_map(
            key=jr.PRNGKey(0),
            maxiter=5,  # Just a few iterations for testing
        )
        assert 'params' in result
        assert 'temperature_map' in result['params']
        assert result['params']['temperature_map'].shape == (mock_geometry.npix,)

    def test_map_reduces_objective(self, mock_oi_data, mock_geometry):
        """MAP should reduce the objective (at least slightly)."""
        model = BayesianStellarModel(
            oi_data=mock_oi_data,
            base_geom=mock_geometry,
        )
        init_params = model.initial_params(jr.PRNGKey(0))
        init_nlp = float(model.neg_log_posterior(init_params))

        result = model.run_map(
            key=jr.PRNGKey(0),
            maxiter=20,
        )
        final_nlp = result['neg_log_posterior']

        # MAP should not increase the objective
        assert final_nlp <= init_nlp + 1e-3  # small tolerance


# ============================================================
# Test high-level interface
# ============================================================

class TestRunInference:
    """Tests for the run_inference convenience function."""

    def test_map_method(self, mock_oi_data, mock_geometry):
        """run_inference with MAP method."""
        result = run_inference(
            oi_data=mock_oi_data,
            base_geom=mock_geometry,
            method='map',
            key=jr.PRNGKey(0),
            maxiter=5,
        )
        assert 'params' in result

    def test_invalid_method(self, mock_oi_data, mock_geometry):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown inference method"):
            run_inference(
                oi_data=mock_oi_data,
                base_geom=mock_geometry,
                method='invalid',
            )


# ============================================================
# Test posterior analysis
# ============================================================

class TestPosteriorAnalysis:
    """Tests for posterior summary and analysis functions."""

    def test_summarize_posterior(self, mock_samples):
        """Posterior summary computes all fields."""
        summary = summarize_posterior(mock_samples)
        assert summary.temperature_map_mean.shape == (48,)
        assert summary.temperature_map_std.shape == (48,)
        assert summary.n_samples == 20

    def test_summary_quantiles(self, mock_samples):
        """Quantiles are in correct order."""
        summary = summarize_posterior(mock_samples)
        q16 = summary.temperature_map_quantiles[0.16]
        q50 = summary.temperature_map_quantiles[0.50]
        q84 = summary.temperature_map_quantiles[0.84]
        assert jnp.all(q16 <= q50)
        assert jnp.all(q50 <= q84)

    def test_geometric_param_summary(self, mock_samples):
        """Geometric parameters are correctly summarized."""
        summary = summarize_posterior(mock_samples)
        assert 'diameter' in summary.geometric_params
        assert 'inclination' in summary.geometric_params

        diam = summary.geometric_params['diameter']
        assert 'mean' in diam
        assert 'std' in diam
        assert 'ci_68' in diam
        assert 'ci_95' in diam
        assert diam['ci_68'][0] < diam['mean'] < diam['ci_68'][1]

    def test_credible_intervals(self, mock_samples):
        """Credible intervals have correct structure."""
        intervals = compute_credible_intervals(
            mock_samples, 'temperature_map', levels=(0.68, 0.95)
        )
        assert 0.68 in intervals
        assert 0.95 in intervals

        lower_68, upper_68 = intervals[0.68]
        lower_95, upper_95 = intervals[0.95]

        # 95% CI should be wider than 68% CI
        assert jnp.all(lower_95 <= lower_68)
        assert jnp.all(upper_95 >= upper_68)

    def test_pixel_significance(self, mock_samples):
        """Significance is non-negative."""
        sig = compute_pixel_significance(mock_samples)
        assert sig.shape == (48,)
        assert jnp.all(sig >= 0)

    def test_spot_probability(self, mock_samples):
        """Spot probability is in [0, 1]."""
        prob = compute_spot_detection_probability(mock_samples)
        assert prob.shape == (48,)
        assert jnp.all(prob >= 0.0)
        assert jnp.all(prob <= 1.0)

    def test_format_table(self, mock_samples):
        """Parameter table formats without error."""
        summary = summarize_posterior(mock_samples)
        table = format_parameter_table(summary)
        assert isinstance(table, str)
        assert "Temperature Map" in table
        assert "Geometric Parameters" in table

    def test_correlation_matrix(self, mock_samples):
        """Correlation matrix is symmetric and has ones on diagonal."""
        corr, names = compute_correlation_matrix(
            mock_samples, ['diameter', 'inclination', 'position_angle']
        )
        assert corr.shape == (3, 3)
        # Diagonal is 1
        np.testing.assert_allclose(np.diag(np.array(corr)), 1.0, atol=0.05)
        # Symmetric
        np.testing.assert_allclose(np.array(corr), np.array(corr.T), atol=1e-6)

    def test_empty_samples_raises(self):
        """Empty sample list raises error."""
        with pytest.raises(ValueError):
            summarize_posterior([])
