import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression


class ICEImputer:
    """
    Feature-by-feature iterative imputer matching the CAFE paper (Algorithm 1).

    Imputation model  : Ridge regression  (paper §IV: "we use ridge regression
                        for imputation")
    Mechanism model   : Logistic regression on the missingness indicator
                        (paper Algorithm 1 line 11: Logit-Reg.fit(R_f^k, X̃^k))

    Note on categorical / binary columns
    -------------------------------------
    The original CAFE code uses Logistic Regression for categorical columns and
    Ridge for numeric ones (distributed_imputer.py).  Here we auto-detect binary
    columns (exactly 2 unique observed values) during fit and apply the same
    split: LogisticRegression for binary, Ridge for everything else.
    Multi-class integer-coded categoricals still use Ridge + rounding.

    Parameters
    ----------
    missing_mask : bool ndarray, shape (n_samples, n_features)
        True where a value is missing.  Fixed for the lifetime of the imputer.
    cat_col_indices : set of int, optional
        Column indices that are categorical (integer-coded 0 … n_classes-1).
        Predictions for these columns are rounded to the nearest integer.
    clip : bool
        Whether to clip imputed values to [features_min, features_max].
    """

    def __init__(self, missing_mask, cat_col_indices=None, clip=True):
        self.missing_mask = missing_mask
        self.cat_col_indices = set(cat_col_indices or [])
        self.clip = clip
        self._use_logistic = {}    # feature_idx -> bool, set by fit_feature
        self._binary_classes = {}  # feature_idx -> (class0, class1) for logistic cols
        self.features_min = None   # set externally before transform_feature
        self.features_max = None   # set externally before transform_feature
        self.missing_info = {
            j: self._compute_missing_info(j)
            for j in range(missing_mask.shape[1])
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initial_impute(self, X, global_stats=None, columns=None):
        """Fill NaN with global mean (numeric) or global mode (categorical)."""
        X_filled = X.copy()
        for j in range(X.shape[1]):
            col_mask = self.missing_mask[:, j]
            if col_mask.sum() == 0:
                continue
            if global_stats is not None and columns is not None:
                col_name = columns[j]
                gs = global_stats[col_name]
                if gs["categorical"]:
                    fill = float(gs["mode"])
                else:
                    fill = (gs["mean"] - gs["min"]) / gs["scale"]
            else:
                observed = X[~col_mask, j]
                observed = observed[~np.isnan(observed)]
                if j in self.cat_col_indices:
                    if len(observed) == 0:
                        fill = 0.0
                    else:
                        vals, counts = np.unique(observed, return_counts=True)
                        fill = float(vals[np.argmax(counts)])
                else:
                    fill = float(np.mean(observed)) if len(observed) > 0 else 0.0
            X_filled[col_mask, j] = fill
        return X_filled

    def fit_feature(self, X, feature_idx):
        """
        Fit a Ridge model to predict feature_idx from all other features
        (paper Algorithm 1, line 10: theta_f^k = Linear-Reg.fit(A_f^k, A_{-f}^k)).

        Returns
        -------
        model_weights : 1-D ndarray  [coef | intercept], shape (n_features,)
        missing_info  : dict         statistics used by CAFE aggregation
        ms_coef       : 1-D ndarray  logistic mechanism model params,
                        shape (n_features + 1,)

        Returns (None, None, None) when the feature is entirely missing for
        this client (server will use other clients' weights instead).
        """
        row_mask = self.missing_mask[:, feature_idx]
        if row_mask.all():
            return None, None, None

        other_cols = np.arange(X.shape[1]) != feature_idx
        X_train = X[~row_mask][:, other_cols]
        y_train = X[~row_mask][:, feature_idx]

        # --- imputation model: Logistic for binary cols, Ridge otherwise ---
        unique_vals = np.unique(y_train)
        if len(unique_vals) == 2:
            est = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
            est.fit(X_train, y_train)
            model_weights = np.concatenate([est.coef_[0], np.atleast_1d(est.intercept_)])
            self._use_logistic[feature_idx] = True
            self._binary_classes[feature_idx] = unique_vals
        else:
            est = Ridge(alpha=1.0)
            est.fit(X_train, y_train)
            model_weights = np.concatenate([est.coef_, np.atleast_1d(est.intercept_)])
            self._use_logistic[feature_idx] = False

        # --- mechanism model: logistic on missingness indicator ---
        ms_coef = self._compute_ms_coef(X, row_mask)

        return model_weights, self.missing_info[feature_idx], ms_coef

    def transform_feature(self, X, feature_idx, global_weights):
        """
        Impute missing values in feature_idx using global_weights = [coef | intercept]
        (paper Algorithm 1, line 16: B_f^k = Linear-Reg.predict(theta_tilde_f^k, B_{-f}^k)).
        """
        row_mask = self.missing_mask[:, feature_idx]
        if row_mask.sum() == 0:
            return X

        other_cols = np.arange(X.shape[1]) != feature_idx
        X_test = X[row_mask][:, other_cols]

        coef = global_weights[:-1]
        intercept = global_weights[-1]

        if self._use_logistic.get(feature_idx, False):
            logit = X_test @ coef + intercept
            prob = 1.0 / (1.0 + np.exp(-logit))
            classes = self._binary_classes.get(feature_idx, np.array([0.0, 1.0]))
            imputed = np.where(prob > 0.5, classes[1], classes[0])
        else:
            imputed = X_test @ coef + intercept
            if self.clip and self.features_min is not None:
                imputed = np.clip(
                    imputed,
                    self.features_min[feature_idx],
                    self.features_max[feature_idx],
                )
            if feature_idx in self.cat_col_indices:
                imputed = np.round(imputed)

        X_out = X.copy()
        X_out[row_mask, feature_idx] = imputed
        return X_out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_missing_info(self, col_idx):
        row_mask = self.missing_mask[:, col_idx]
        other_cols = np.arange(self.missing_mask.shape[1]) != col_idx
        rest_mask = self.missing_mask[~row_mask][:, other_cols]
        sample_size = rest_mask.shape[0]
        n_total = self.missing_mask.shape[0]
        if sample_size == 0:
            return {
                "sample_size": 0,
                "sample_row_pct": 0.0,
                "missing_cell_pct": 1.0,
                "missing_row_pct": 1.0,
            }
        return {
            "sample_size": sample_size,                              # N_f^k (paper)
            "sample_row_pct": sample_size / n_total,
            "missing_cell_pct": rest_mask.sum() / max(rest_mask.size, 1),
            "missing_row_pct": rest_mask.any(axis=1).sum() / sample_size,
        }

    def _compute_ms_coef(self, X, row_mask):
        """
        Fit logistic regression on the missingness indicator R_f^k to obtain
        the mechanism model xi_f^k (paper Algorithm 1, line 11).

        Returns [coef | intercept] of shape (n_features + 1,).
        """
        n_missing = int(row_mask.sum())
        n_observed = int((~row_mask).sum())
        n_features = X.shape[1]

        if n_missing < 3 or n_observed < 3:
            return np.full(n_features + 1, 0.001)

        try:
            lr = LogisticRegression(
                C=0.1, random_state=0, max_iter=500,
                class_weight="balanced", solver="lbfgs",
            )
            lr.fit(X, row_mask)
            return np.concatenate([lr.coef_[0], lr.intercept_])
        except Exception:
            return np.full(n_features + 1, 0.001)
