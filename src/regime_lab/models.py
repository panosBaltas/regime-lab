"""Train-only PCA and Gaussian HMM; public-parameter forward filtering in log space."""

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .config import Config


def emission_log_likelihood(model: GaussianHMM, observations: np.ndarray) -> np.ndarray:
    """Full-covariance Gaussian log densities without hmmlearn private APIs."""
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 2 or not np.isfinite(observations).all():
        raise ValueError("Observations must be a finite matrix")
    if observations.shape[1] != model.means_.shape[1]:
        raise ValueError("Observation dimension does not match model")
    result = np.empty((len(observations), model.n_components))
    for state in range(model.n_components):
        chol = np.linalg.cholesky(model.covars_[state])
        residuals = solve_triangular(chol, (observations - model.means_[state]).T, lower=True)
        result[:, state] = -0.5 * (observations.shape[1] * np.log(2 * np.pi)
                                  + 2 * np.log(np.diag(chol)).sum()
                                  + np.square(residuals).sum(axis=0))
    return result


def filtered_probabilities(model: GaussianHMM, observations: np.ndarray,
                           previous: np.ndarray | None = None) -> np.ndarray:
    """P(S_t | x_1,...,x_t); previous is the posterior at the preceding session."""
    emissions = emission_log_likelihood(model, observations)
    with np.errstate(divide="ignore"):
        log_transition = np.log(model.transmat_)
        log_prior = np.log(model.startprob_ if previous is None else previous)
    result = np.empty_like(emissions)
    for row in range(len(emissions)):
        if row > 0 or previous is not None:
            log_prior = logsumexp(log_prior[:, None] + log_transition, axis=0)
        log_posterior = log_prior + emissions[row]
        normalizer = logsumexp(log_posterior)
        if not np.isfinite(normalizer):
            raise ValueError("HMM produced an impossible observation sequence")
        log_prior = log_posterior - normalizer
        result[row] = np.exp(log_prior)
    return result


class RegimeEncoder:
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=config.n_components, svd_solver="full", whiten=False)

    def fit(self, features: np.ndarray, target_daily_returns: np.ndarray):
        if self.config.n_components > features.shape[1]:
            raise ValueError("n_components exceeds the number of available features")
        factors = self.pca.fit_transform(self.scaler.fit_transform(features))
        self.candidates = []
        best_score = -np.inf
        best = None
        for restart in range(self.config.hmm_restarts):
            seed = self.config.seed + restart
            model = GaussianHMM(n_components=self.config.n_states, covariance_type="full",
                                n_iter=self.config.hmm_iterations, tol=1e-3,
                                min_covar=1e-3, random_state=seed, implementation="scaling",
                                transmat_prior=1.0 + 3.0 * np.eye(self.config.n_states))
            candidate = {"seed": seed}
            try:
                model.fit(factors)
                score = float(model.score(factors))
                if not np.isfinite(score):
                    raise ValueError("Nonfinite calibration likelihood")
                probabilities = filtered_probabilities(model, factors)
                occupied = bool((probabilities.mean(axis=0) >= 0.01).all())
                history = list(model.monitor_.history)
                delta = float(history[-1] - history[-2]) if len(history) >= 2 else None
                tolerance_reached = bool(delta is not None and -1e-6 <= delta < model.tol)
                candidate.update(log_likelihood=score, iterations=model.monitor_.iter,
                                 tolerance_reached=tolerance_reached, occupied=occupied)
                if occupied and score > best_score:
                    best_score = score
                    best = model
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                candidate["error"] = str(exc)
            self.candidates.append(candidate)
        if best is None:
            raise ValueError(f"No usable HMM fit (states must each occupy >=1% of calibration): {self.candidates}")
        self.model = best
        probabilities = filtered_probabilities(best, factors)
        risk = []
        for state in range(best.n_components):
            weights = probabilities[:, state]
            mean = np.average(target_daily_returns, weights=weights)
            variance = np.average(np.square(target_daily_returns - mean), weights=weights)
            risk.append(np.sqrt(variance * 252))
        self.order = np.argsort(risk, kind="stable")
        self.calibration_volatility = np.asarray(risk)[self.order]
        self.selected_seed = int(best.random_state)
        return self

    def transform(self, features: np.ndarray):
        return self.pca.transform(self.scaler.transform(features))

    def probabilities(self, features: np.ndarray):
        return filtered_probabilities(self.model, self.transform(features))[:, self.order]

    @property
    def transition_matrix(self):
        return self.model.transmat_[np.ix_(self.order, self.order)]


class ReturnModels:
    """Shared ridge slopes plus penalized, soft-probability regime deviations."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.factor_scaler = StandardScaler()

    def _augmented(self, x, probabilities):
        contrast = probabilities[:, :-1] - self.probability_center
        interactions = (contrast[:, :, None] * x[:, None, :]).reshape(len(x), -1)
        return np.column_stack([x, contrast, interactions])

    def fit(self, features, factors, probabilities, returns):
        x = self.scaler.fit_transform(features)
        z = self.factor_scaler.fit_transform(factors)
        self.probability_center = probabilities[:, :-1].mean(axis=0)
        self.mean = float(np.mean(returns))
        self.ridge = Ridge(alpha=self.alpha).fit(x, returns)
        self.pca_ridge = Ridge(alpha=self.alpha).fit(z, returns)
        self.probability_ridge = Ridge(alpha=self.alpha).fit(probabilities[:, :-1], returns)
        self.regime_ridge = Ridge(alpha=self.alpha).fit(self._augmented(x, probabilities), returns)
        return self

    def predict(self, features, factors, probabilities):
        x = self.scaler.transform(features)
        return {
            "historical_mean": np.full(len(x), self.mean),
            "ridge": self.ridge.predict(x),
            "pca_ridge": self.pca_ridge.predict(self.factor_scaler.transform(factors)),
            "regime_probabilities": self.probability_ridge.predict(probabilities[:, :-1]),
            "pca_hmm_ridge": self.regime_ridge.predict(self._augmented(x, probabilities)),
        }

