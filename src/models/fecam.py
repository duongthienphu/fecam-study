"""First per-class FeCAM implementation; see docs/method.md.

Input: raw frozen-backbone features [N, d], global int64 labels [N].
Call update ONCE with all training features of the new classes in a task.
No images, exemplars, optimizer, or backbone are stored here.

Choices made explicit for this first implementation:
* Prototype = transform(mean(raw_features)); covariance = cov(transformed_features).
* Sample covariance uses n - 1, computed from the Tukey-transformed raw features.
* L2-normalize prototypes after Tukey transform. For queries, normalize before
  Tukey transform and again before distance scoring, matching the authors' code.
* Eq. (8) shrinkage precedes Eq. (7) correlation normalization.
* Eq. (10) distance-only score: no Gaussian log-determinant or prior term.
* Cholesky factors replace explicit inverses. Optional jitter is recorded.
* covariance_mode="classwise" keeps one FeCAM covariance per class.
* covariance_mode="shared" updates one class-averaged covariance across tasks.
* covariance_mode="none" uses I implicitly: squared Euclidean distance after
  the same feature/prototype transform, with no covariance factors stored.
  This is an ablation, not necessarily the project's NCM baseline.
  No eigenvalue/eigenvector compression is implemented.

The L2-normalization order follows the authors' released implementation.
Other implementation details remain independently implemented.

Example:
    classifier = FeCAM(transform="sqrt", gamma1=1.0, gamma2=1.0)
    classifier.update(train_features, train_labels)
    predictions = classifier.predict(test_features)  # global class IDs

For signed ViT features, use transform="none" and protocol-specific gammas.
Requires PyTorch with torch.linalg.cholesky_ex and solve_triangular.
"""

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class _Config:
    mode: str
    covariance_mode: str
    transform: str
    normalize_features: bool
    normalize_prototypes: bool
    gamma1: float
    gamma2: float
    correlation_normalization: bool
    jitter_schedule: tuple[float, ...]


class FeCAM:
    """Incremental classifier with no, shared, or per-class covariance.

    Statistics and returned distances use float64 on ``device`` (CPU default).
    Returned labels/class_ids use int64 on that same device. Inputs are detached
    and moved there automatically. Only new classes are accepted by update.
    Configuration is fixed at construction so old/new statistics stay compatible.
    """

    def __init__(
        self,
        *,
        mode: str = "full",
        covariance_mode: str | None = None,
        transform: str = "sqrt",
        normalize_features: bool = True,
        normalize_prototypes: bool = True,
        gamma1: float = 1.0,
        gamma2: float = 1.0,
        correlation_normalization: bool = True,
        jitter_schedule: Sequence[float] = (0.0, 1e-8, 1e-6, 1e-4, 1e-2),
        device: str | torch.device = "cpu",
    ) -> None:
        if mode not in ("full", "identity"):
            raise ValueError("mode must be 'full' or 'identity'.")
        if covariance_mode is None:
            covariance_mode = "classwise" if mode == "full" else "none"
        if covariance_mode not in ("none", "shared", "classwise"):
            raise ValueError("covariance_mode must be 'none', 'shared', or 'classwise'.")
        if covariance_mode == "none" and mode == "full":
            mode = "identity"
        elif covariance_mode != "none":
            mode = "full"
        if transform not in ("none", "sqrt"):
            raise ValueError("transform must be 'none' or 'sqrt'.")
        gamma1, gamma2 = float(gamma1), float(gamma2)
        if any(not math.isfinite(g) or g < 0 for g in (gamma1, gamma2)):
            raise ValueError("gamma1 and gamma2 must be finite and nonnegative.")
        schedule = tuple(float(v) for v in jitter_schedule)
        if (not schedule or schedule[0] != 0.0
                or any(not math.isfinite(v) or v < 0 for v in schedule)
                or any(a >= b for a, b in zip(schedule, schedule[1:]))):
            raise ValueError("jitter_schedule must start at 0 and strictly increase.")
        self._config = _Config(
            mode, covariance_mode, transform, bool(normalize_features),
            bool(normalize_prototypes), gamma1, gamma2,
            bool(correlation_normalization), schedule,
        )
        self._device = torch.device(device)
        self._d: int | None = None
        self.class_ids = torch.empty(0, dtype=torch.int64, device=self._device)
        self.counts = torch.empty(0, dtype=torch.int64, device=self._device)
        self.raw_prototypes = torch.empty((0, 0), dtype=torch.float64, device=self._device)
        self.score_prototypes = self.raw_prototypes.clone()
        self.factors = torch.empty((0, 0, 0), dtype=torch.float64, device=self._device)
        self.jitter_used = torch.empty(0, dtype=torch.float64, device=self._device)
        self._shared_covariance = torch.empty((0, 0), dtype=torch.float64, device=self._device)

    @property
    def mode(self) -> str:
        """Return the fixed covariance mode: full or identity."""
        return self._config.mode

    @property
    def covariance_mode(self) -> str:
        """Return none, shared, or classwise covariance behavior."""
        return self._config.covariance_mode

    @property
    def num_classes(self) -> int:
        return self.class_ids.numel()

    @property
    def feature_dim(self) -> int | None:
        return self._d

    @property
    def id_to_column(self) -> dict[int, int]:
        return {label: column for column, label in enumerate(self.class_ids.tolist())}

    def _features(self, features: Tensor, *, allow_empty: bool) -> Tensor:
        if not isinstance(features, Tensor) or not features.is_floating_point():
            raise TypeError("features must be a floating-point torch.Tensor.")
        if features.ndim != 2 or features.shape[1] == 0:
            raise ValueError("features must have shape [N, d] with d > 0.")
        if not allow_empty and features.shape[0] == 0:
            raise ValueError("update requires at least one sample.")
        if self._d is not None and features.shape[1] != self._d:
            raise ValueError(f"Expected {self._d} features, got {features.shape[1]}.")
        h = features.detach().to(device=self._device, dtype=torch.float64)
        if not torch.isfinite(h).all().item():
            raise ValueError("features contain NaN or infinity.")
        if self._config.transform == "sqrt" and (h < 0).any().item():
            raise ValueError("sqrt Tukey requires nonnegative features; use 'none' for signed features.")
        return h

    def _transform(self, h: Tensor) -> Tensor:
        return h.sqrt() if self._config.transform == "sqrt" else h

    def _build_factor(self, covariance: Tensor, class_id: int) -> tuple[Tensor, float]:
        """Shrink, optionally correlation-normalize, then factor the matrix."""
        d = covariance.shape[0]
        covariance = (covariance + covariance.T) / 2
        eye = torch.eye(d, dtype=covariance.dtype, device=covariance.device)
        diagonal = covariance.diagonal()
        v1 = diagonal.mean()
        v2 = ((covariance.sum() - diagonal.sum()) / (d * (d - 1))
              if d > 1 else covariance.new_zeros(()))
        shrunk = (covariance + self._config.gamma1 * v1 * eye
                  + self._config.gamma2 * v2 * (torch.ones_like(eye) - eye))
        if (not torch.isfinite(shrunk).all().item()
                or (shrunk.diagonal() <= 0).any().item()):
            raise ValueError(
                f"Class {class_id}: invalid shrunk covariance (nonfinite or nonpositive diagonal). "
                "Check constant features, sample counts and shrinkage parameters."
            )
        if self._config.correlation_normalization:
            std = shrunk.diagonal().sqrt()
            shrunk = shrunk / torch.outer(std, std)
            shrunk = (shrunk + shrunk.T) / 2
            if not torch.isfinite(shrunk).all().item():
                raise ValueError(f"Class {class_id}: correlation normalization overflowed.")
        for jitter in self._config.jitter_schedule:
            factor, info = torch.linalg.cholesky_ex(shrunk + jitter * eye)
            if info.item() == 0 and torch.isfinite(factor).all().item():
                return factor, jitter
        raise ValueError(
            f"Class {class_id}: covariance is not positive definite after jitter schedule "
            f"{self._config.jitter_schedule}. No state was updated."
        )

    @torch.no_grad()
    def update(self, features: Tensor, labels: Tensor) -> None:
        """Add complete new classes; preserve old statistics on success or failure.

        Each class must have >= 2 training samples. Repeated class IDs are rejected.
        New IDs are sorted within this call and appended after existing IDs.
        Both modes require >= 2 samples for comparable input validation.
        Identity mode skips covariance/shrinkage/correlation-normalization/
        factorization: factors stays empty and jitter_used records zero per class.
        """
        h = self._features(features, allow_empty=False)
        if not isinstance(labels, Tensor) or labels.dtype != torch.int64:
            raise TypeError("labels must be a torch.int64 tensor of global class IDs.")
        if labels.ndim != 1 or labels.shape[0] != h.shape[0]:
            raise ValueError("labels must have shape [N], matching features.")
        labels = labels.detach().to(self._device)
        new_ids = torch.unique(labels, sorted=True)
        overlap = set(new_ids.tolist()).intersection(self.class_ids.tolist())
        if overlap:
            raise ValueError(f"Classes already learned: {sorted(overlap)}. Pass only new classes.")

        raw_means, score_means, factors, covariances, counts, jitters = [], [], [], [], [], []
        for class_id in new_ids.tolist():
            h_y = h[labels == class_id]
            n = h_y.shape[0]
            if n < 2:
                raise ValueError(f"Class {class_id}: need at least 2 samples, got {n}.")
            mu = h_y.mean(dim=0)
            jitter = 0.0
            if self.covariance_mode != "none":
                z_y = self._transform(h_y)
                centered = z_y - z_y.mean(dim=0)
                covariance = centered.T @ centered / (n - 1)
                covariances.append(covariance)
                if self.covariance_mode == "classwise":
                    factor, jitter = self._build_factor(covariance, class_id)
                    factors.append(factor)
            score_mu = self._transform(mu)
            if self._config.normalize_prototypes:
                score_mu = torch.nn.functional.normalize(score_mu, p=2, dim=0)
            if not torch.isfinite(score_mu).all().item():
                raise ValueError(f"Class {class_id}: prototype overflowed.")
            raw_means.append(mu)
            score_means.append(score_mu)
            counts.append(n)
            jitters.append(jitter)

        # Build every new state tensor BEFORE assigning any persistent attribute.
        def append(old: Tensor, new: Tensor) -> Tensor:
            return torch.cat((old, new), dim=0) if self.num_classes else new

        if self.covariance_mode == "shared":
            # Paper Eq. (6): average the task covariance by the number of classes.
            # CIFAR-100 has the same number of examples per class, so this is also
            # the pooled within-class covariance for the configured protocol.
            current_covariance = torch.stack(covariances).mean(dim=0)
            old_classes = self.num_classes
            new_classes = len(covariances)
            if old_classes:
                shared_covariance = (
                    self._shared_covariance * old_classes
                    + current_covariance * new_classes
                ) / (old_classes + new_classes)
            else:
                shared_covariance = current_covariance
            shared_factor, shared_jitter = self._build_factor(shared_covariance, -1)
            factor_state = shared_factor.unsqueeze(0)
            jitter_state = torch.tensor([shared_jitter], dtype=torch.float64, device=self._device)
        elif self.covariance_mode == "classwise":
            shared_covariance = self._shared_covariance
            factor_state = append(self.factors, torch.stack(factors))
            jitter_state = append(
                self.jitter_used,
                torch.tensor(jitters, dtype=torch.float64, device=self._device),
            )
        else:
            shared_covariance = self._shared_covariance
            factor_state = self.factors
            jitter_state = append(
                self.jitter_used,
                torch.tensor(jitters, dtype=torch.float64, device=self._device),
            )

        state = (
            append(self.class_ids, new_ids),
            append(self.raw_prototypes, torch.stack(raw_means)),
            append(self.score_prototypes, torch.stack(score_means)),
            factor_state,
            append(self.counts, torch.tensor(counts, dtype=torch.int64, device=self._device)),
            jitter_state,
        )
        (self.class_ids, self.raw_prototypes, self.score_prototypes,
         self.factors, self.counts, self.jitter_used) = state
        self._shared_covariance = shared_covariance
        self._d = h.shape[1]

    @torch.no_grad()
    def distances(self, features: Tensor) -> Tensor:
        """Return squared Mahalanobis distances [B, C]; smaller is better.

        In identity mode these are squared Euclidean distances between the
        normalized transformed query and normalized transformed prototypes.
        Column j corresponds to class_ids[j]. These values are not probabilities.
        Query batches may be empty. Batch large query sets in the caller.
        """
        if not self.num_classes:
            raise RuntimeError("Call update before prediction.")
        h = self._features(features, allow_empty=True)
        if self._config.normalize_features:
            h = torch.nn.functional.normalize(h, p=2, dim=1)
        z = self._transform(h)
        if self._config.normalize_features:
            z = torch.nn.functional.normalize(z, p=2, dim=1)
        result = z.new_empty((z.shape[0], self.num_classes))
        if z.shape[0] == 0:
            return result
        for column in range(self.num_classes):
            residual = z - self.score_prototypes[column]
            if self.covariance_mode == "none":
                # I is implicit; no d x d tensor or inverse is needed.
                result[:, column] = residual.square().sum(dim=1)
            else:
                # R = L L^T; v^T R^-1 v = ||L^-1 v||^2.
                factor_index = column if self.covariance_mode == "classwise" else 0
                whitened = torch.linalg.solve_triangular(
                    self.factors[factor_index], residual.T, upper=False
                )
                result[:, column] = whitened.square().sum(dim=0)
        if not torch.isfinite(result).all().item():
            raise FloatingPointError("Nonfinite distances; check feature scale and conditioning.")
        return result

    @torch.no_grad()
    def predict(self, features: Tensor) -> Tensor:
        """Return global class IDs [B]; ties select the first stored class."""
        distances = self.distances(features)
        if distances.shape[0] == 0:
            return self.class_ids.new_empty((0,))
        return self.class_ids[distances.argmin(dim=1)]


def _toy_example() -> None:
    """Run with: python src/models/fecam.py"""
    features = torch.tensor([[1., 1.], [1., 2.], [2., 1.],
                             [5., 5.], [5., 6.], [6., 5.]])
    labels = torch.tensor([7, 7, 7, 42, 42, 42], dtype=torch.int64)
    model = FeCAM(transform="none", gamma1=1, gamma2=0)
    model.update(features, labels)
    query = torch.tensor([[1., 1.], [5., 5.]])
    print("Class IDs:", model.class_ids.tolist())
    print("Squared distances:\n", model.distances(query))
    print("Predictions:", model.predict(query).tolist())
    print("Jitter used:", model.jitter_used.tolist())


if __name__ == "__main__":
    _toy_example()
