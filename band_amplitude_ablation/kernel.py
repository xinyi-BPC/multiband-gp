"""Band-specific covariance SDs, with nm inputs and a fixed Å wavelength scale."""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, Matern
from MOGP_model import DEFAULT_BAND_TO_WAVELENGTH


class BandAmplitudeMatern(Kernel):
    """D_sigma K_time K_wavelength D_sigma; analytic log-parameter gradients.

    Only the explicitly configured bands have parameters. Exact wavelength
    lookup makes row ordering irrelevant. No additional global amplitude exists.
    sigma is a vector of standard deviations, not covariance variances.
    """
    def __init__(self, bands=("i", "z"), wavelengths=(754.6, 869.1),
                 sigma=(1.0, 1.0), sigma_bounds=(1e-2, 1e2),
                 time_length_scale=0.3, time_length_scale_bounds=(0.05, 10.0),
                 wavelength_length_scale=6000.0):
        self.bands = bands
        self.wavelengths = wavelengths
        self.sigma = sigma
        self.sigma_bounds = sigma_bounds
        self.time_length_scale = time_length_scale
        self.time_length_scale_bounds = time_length_scale_bounds
        self.wavelength_length_scale = wavelength_length_scale

    @property
    def hyperparameter_sigma(self):
        return Hyperparameter("sigma", "numeric", self.sigma_bounds, len(self.bands))

    @property
    def hyperparameter_time_length_scale(self):
        return Hyperparameter("time_length_scale", "numeric", self.time_length_scale_bounds)

    def _indices(self, X):
        waves = np.asarray(self.wavelengths)
        sigmas = np.atleast_1d(self.sigma)
        if len(set(waves)) != len(waves) or len(waves) != len(self.bands) or len(sigmas) != len(waves):
            raise ValueError("Bands, unique wavelengths, and sigma must have equal length")
        if np.any(sigmas <= 0) or not np.all(np.isfinite(sigmas)):
            raise ValueError("Amplitudes must be positive and finite")
        matches = np.asarray(X)[:, 1, None] == waves[None, :]
        if not np.all(matches.sum(axis=1) == 1):
            raise ValueError("Input wavelength has no exact configured band mapping")
        return matches.argmax(axis=1)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X)
        if eval_gradient and Y is not None:
            raise ValueError("Gradient requires Y=None")
        ix = self._indices(X)
        iy = ix if Y is None else self._indices(Y)
        Yv = X if Y is None else np.asarray(Y)
        time = Matern(self.time_length_scale, self.time_length_scale_bounds, nu=1.5)
        if eval_gradient:
            kt, gt = time(X[:, [0]], eval_gradient=True)
        else:
            kt = time(X[:, [0]], Yv[:, [0]])
        kw = Matern(self.wavelength_length_scale, "fixed", nu=1.5)(
            X[:, [1]] * 10.0, Yv[:, [1]] * 10.0)
        s = np.atleast_1d(self.sigma)
        covariance = s[ix, None] * s[iy][None, :] * kt * kw
        if not eval_gradient:
            return covariance
        gradients = []
        if not self.hyperparameter_sigma.fixed:
            for j in range(len(self.bands)):
                gradients.append(covariance * ((ix == j)[:, None].astype(float) + (ix == j)[None, :]))
        if not self.hyperparameter_time_length_scale.fixed:
            gradients.append(s[ix, None] * s[ix][None, :] * kw * gt[:, :, 0])
        return covariance, np.stack(gradients, axis=2) if gradients else np.empty((*covariance.shape, 0))

    def diag(self, X):
        return np.atleast_1d(self.sigma)[self._indices(X)] ** 2

    def is_stationary(self):
        return False


def initialize_amplitudes(data, bands, bounds):
    """Training-only MAD; fallback to uncentered RMS/noise for tiny or flat sets.

    This GP has zero mean and no background subtraction. For one/two points or
    vanishing MAD, RMS provides a finite amplitude even for a constant signal.
    Out-of-bound estimates are reported as errors, never silently clipped.
    """
    estimates = []
    details = []
    for band in bands:
        mask = np.asarray(data['band']) == band
        y, err = np.asarray(data['y'])[mask], np.asarray(data['yerr'])[mask]
        mad = 1.4826 * np.median(np.abs(y - np.median(y)))
        method = 'MAD'
        scale = mad
        if len(y) < 3 or mad <= np.finfo(float).eps:
            scale = max(float(np.sqrt(np.mean(y ** 2))), float(np.median(err)), 1e-8)
            method = 'RMS/noise fallback'
        if not bounds[0] < scale < bounds[1]:
            raise ValueError(f'{band} training scale {scale} outside sigma bounds {bounds}; explicit adjustment required')
        estimates.append(float(scale))
        details.append(dict(band=band, n_train=len(y), initial_sigma=float(scale), method=method))
    return tuple(estimates), details


def fit_band_amplitude_gp(data, sigma_bounds=(1e-2, 1e2), **settings):
    bands = tuple(sorted(set(data['band'])))
    sigma, initialization = initialize_amplitudes(data, bands, sigma_bounds)
    kernel = BandAmplitudeMatern(bands=bands,
        wavelengths=tuple(DEFAULT_BAND_TO_WAVELENGTH[b] for b in bands), sigma=sigma,
        sigma_bounds=sigma_bounds, time_length_scale=settings.get('time_length_scale', 0.3),
        time_length_scale_bounds=settings.get('time_length_scale_bounds', (0.05, 10)),
        wavelength_length_scale=settings.get('wavelength_length_scale', 6000))
    gp = GaussianProcessRegressor(kernel=kernel,
        alpha=(settings.get('yerr_scale', 1) * data['yerr']) ** 2 + settings.get('noise_floor', 0) ** 2 + settings.get('jitter', 1e-8),
        normalize_y=False, n_restarts_optimizer=settings.get('n_restarts_optimizer', 5),
        random_state=settings.get('random_state', 1))
    gp.fit(data['X'], data['y'])
    return gp, initialization
