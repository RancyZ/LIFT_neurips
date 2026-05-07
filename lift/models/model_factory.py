"""
Model factory — instantiates any of the 9 candidate models with a uniform
fit/predict interface wrapping sklearn, bambi, and torch backends.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseModelWrapper(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# sklearn wrappers
# ---------------------------------------------------------------------------

class LogisticRegressionWrapper(BaseModelWrapper):
    def __init__(self, C: float = 1.0) -> None:
        from sklearn.linear_model import LogisticRegression
        self._model = LogisticRegression(
            C=C, solver="lbfgs", max_iter=1000, random_state=42
        )

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class KNNWrapper(BaseModelWrapper):
    def __init__(self, k: int = 5) -> None:
        from sklearn.neighbors import KNeighborsClassifier
        self._model = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="uniform")

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class SVMWrapper(BaseModelWrapper):
    def __init__(self, C: float = 1.0) -> None:
        from sklearn.svm import SVC
        self._model = SVC(C=C, kernel="rbf", gamma="scale", probability=True, random_state=42)

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class DecisionTreeWrapper(BaseModelWrapper):
    def __init__(self, max_depth: int = 3) -> None:
        from sklearn.tree import DecisionTreeClassifier
        self._model = DecisionTreeClassifier(
            max_depth=max_depth, criterion="gini", random_state=42
        )

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class RandomForestWrapper(BaseModelWrapper):
    def __init__(self, n_estimators: int = 100) -> None:
        from sklearn.ensemble import RandomForestClassifier
        self._model = RandomForestClassifier(
            n_estimators=n_estimators, criterion="gini",
            max_features=None,  # paper: all features considered at each split
            bootstrap=True, random_state=42, n_jobs=-1,
        )

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class MLPWrapper(BaseModelWrapper):
    def __init__(self, lr: float = 0.001) -> None:
        from sklearn.neural_network import MLPClassifier
        self._model = MLPClassifier(
            hidden_layer_sizes=(10, 5),
            activation="relu",
            max_iter=100,
            learning_rate_init=lr,
            random_state=42,
        )

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


# ---------------------------------------------------------------------------
# Bayesian Regression via bambi
# ---------------------------------------------------------------------------

class BayesianRegressionWrapper(BaseModelWrapper):
    """
    Logistic regression using bambi (PyMC backend).
    Uses MCMC sampling; inherently stochastic — set seeds externally.
    """
    def __init__(self, sigma: float = 5.0) -> None:
        self.sigma = sigma
        self._model = None
        self._idata = None
        self._feature_cols: list[str] = []
        self._outcome_col: str = "y"
        self._classes: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import bambi as bmb
        import warnings
        warnings.filterwarnings("ignore")

        self._feature_cols = list(X.columns)
        self._outcome_col = y.name or "y"
        self._classes = np.unique(y)

        data = X.copy()
        data[self._outcome_col] = y.values

        # Centered normal prior N(0, sigma) on all coefficients, per paper spec
        priors = {col: bmb.Prior("Normal", mu=0, sigma=self.sigma)
                  for col in self._feature_cols}
        formula = f"{self._outcome_col} ~ " + " + ".join(self._feature_cols)
        self._model = bmb.Model(formula, data, family="bernoulli", priors=priors)
        self._idata = self._model.fit(draws=50, tune=20, chains=1, progressbar=False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        import bambi as bmb
        data = X.copy()
        data[self._outcome_col] = 0  # placeholder required by bambi
        preds = self._model.predict(self._idata, data=data, kind="mean", inplace=False)
        mean_proba = preds.posterior["y_mean"].values.mean(axis=(0, 1))
        return np.column_stack([1 - mean_proba, mean_proba])


# ---------------------------------------------------------------------------
# PyTorch — ResNet10 classifier
# ---------------------------------------------------------------------------

class ResNetWrapper(BaseModelWrapper):
    """
    Simple 10-layer residual network for tabular data.
    BCE loss, Adam optimiser, batch=128, 100 epochs.
    """
    def __init__(self, lr: float = 0.001) -> None:
        self.lr = lr
        self._net = None
        self._n_features: int = 0
        self._device: str = "cpu"

    def _build_net(self, n_features: int):
        import torch
        import torch.nn as nn

        class ResBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc1 = nn.Linear(dim, dim)
                self.fc2 = nn.Linear(dim, dim)
                self.relu = nn.ReLU()
                self.bn1 = nn.BatchNorm1d(dim)
                self.bn2 = nn.BatchNorm1d(dim)

            def forward(self, x):
                return self.relu(self.bn2(self.fc2(self.relu(self.bn1(self.fc1(x))))) + x)

        class ResNet10(nn.Module):
            def __init__(self, n_features):
                super().__init__()
                hidden = 64
                self.input = nn.Linear(n_features, hidden)
                self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(4)])
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                x = torch.relu(self.input(x))
                x = self.blocks(x)
                return self.head(x).squeeze(-1)

        return ResNet10(n_features)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._n_features = X.shape[1]
        self._net = self._build_net(self._n_features).to(self._device)

        X_t = torch.tensor(X.values, dtype=torch.float32)
        y_t = torch.tensor(y.values, dtype=torch.float32)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=128, shuffle=True)

        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        self._net.train()
        for _ in range(100):
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                opt.zero_grad()
                loss_fn(self._net(xb), yb).backward()
                opt.step()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        X_t = torch.tensor(X.values, dtype=torch.float32).to(self._device)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(X_t).cpu().numpy()
        proba = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1 - proba, proba])


# ---------------------------------------------------------------------------
# PyTorch — VAE with classification head
# ---------------------------------------------------------------------------

class VAEWrapper(BaseModelWrapper):
    """
    Variational Autoencoder with a classification head.
    Loss = 2*BCE_rec + 0.7*L1_rec + 0.3*KL
    """
    def __init__(self, lr: float = 0.001) -> None:
        self.lr = lr
        self._net = None
        self._n_features: int = 0
        self._device: str = "cpu"

    def _build_vae(self, n_features: int):
        import torch
        import torch.nn as nn

        class VAEClassifier(nn.Module):
            def __init__(self, n_features, latent_dim=8):
                super().__init__()
                self.encoder_mu = nn.Sequential(
                    nn.Linear(n_features, 32), nn.ReLU(), nn.Linear(32, latent_dim)
                )
                self.encoder_logvar = nn.Sequential(
                    nn.Linear(n_features, 32), nn.ReLU(), nn.Linear(32, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, n_features)
                )
                self.classifier = nn.Linear(latent_dim, 1)

            def reparameterise(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                return mu + std * torch.randn_like(std)

            def forward(self, x):
                mu = self.encoder_mu(x)
                logvar = self.encoder_logvar(x)
                z = self.reparameterise(mu, logvar)
                x_hat = self.decoder(z)
                logit = self.classifier(mu).squeeze(-1)
                return logit, x_hat, mu, logvar

        return VAEClassifier(n_features)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._n_features = X.shape[1]
        self._net = self._build_vae(self._n_features).to(self._device)

        X_t = torch.tensor(X.values, dtype=torch.float32)
        y_t = torch.tensor(y.values, dtype=torch.float32)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=128, shuffle=True)

        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        bce_loss     = nn.BCEWithLogitsLoss()
        bce_rec_loss = nn.BCEWithLogitsLoss()  # reconstruction BCE (paper: weight 2)
        l1_loss      = nn.L1Loss()             # reconstruction L1  (paper: weight 0.7)

        # Normalise X to [0, 1] per feature for BCE reconstruction target
        X_min = X_t.min(dim=0).values
        X_max = X_t.max(dim=0).values
        X_range = (X_max - X_min).clamp(min=1e-8)

        self._net.train()
        for _ in range(100):
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                # Normalise this batch to [0, 1] to serve as BCE target
                xb_norm = (xb - X_min.to(self._device)) / X_range.to(self._device)
                opt.zero_grad()
                logit, x_hat, mu, logvar = self._net(xb)
                # Loss = clf_BCE + 2·rec_BCE + 0.7·rec_L1 + 0.3·KL  (paper spec)
                rec_bce = bce_rec_loss(x_hat, xb_norm)
                rec_l1  = l1_loss(torch.sigmoid(x_hat), xb_norm)
                kl      = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                clf_loss = bce_loss(logit, yb)
                loss = clf_loss + 2.0 * rec_bce + 0.7 * rec_l1 + 0.3 * kl
                loss.backward()
                opt.step()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        X_t = torch.tensor(X.values, dtype=torch.float32).to(self._device)
        self._net.eval()
        with torch.no_grad():
            logits, _, _, _ = self._net(X_t)
            logits = logits.cpu().numpy()
        proba = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1 - proba, proba])


# ---------------------------------------------------------------------------
# Registry + factory function
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: Dict[str, type] = {
    "LR": LogisticRegressionWrapper,
    "BR": BayesianRegressionWrapper,
    "KNN": KNNWrapper,
    "SVM": SVMWrapper,
    "DT": DecisionTreeWrapper,
    "RF": RandomForestWrapper,
    "MLP": MLPWrapper,
    "RN": ResNetWrapper,
    "VAE": VAEWrapper,
}


def get_model(
    model_id: str,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> BaseModelWrapper:
    """Returns an instantiated model with optional hyperparams."""
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{model_id}'. Supported: {list(SUPPORTED_MODELS)}")
    cls = SUPPORTED_MODELS[model_id]
    params = hyperparams or {}
    # Map tuner param names to constructor param names
    param_map = {
        "k": "k",
    }
    mapped = {param_map.get(k, k): v for k, v in params.items()}
    return cls(**mapped)
