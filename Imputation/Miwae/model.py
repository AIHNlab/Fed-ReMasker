from typing import List, Dict, Tuple
import numpy as np
import torch
from torch import nn
import torch.distributions as td
import random
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True)


def weights_init(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.orthogonal_(layer.weight)


class MIWAE(nn.Module):
    """
    MIWAE: Deep Generative Modelling and Imputation of Incomplete Data.
    Reference: Mattei & Frellsen, ICML 2019.

    Encoder: x -> q(z | x_obs)  (Gaussian)
    Decoder: z -> p(x | z)       (Student-T per feature, for robustness to outliers)

    K  — importance-weighted samples during training
    L  — samples used at imputation time
    """

    def __init__(self, num_features: int, K: int = 50, L: int = 10000,
                 n_hidden: int = 256, latent_size: int = 20, seed: int = 0) -> None:
        super().__init__()
        set_seed(seed)

        self.num_features = num_features
        self.n_hidden = n_hidden        # Fed-MIWAE paper Table S3: 256
        self.latent_size = latent_size  # Fed-MIWAE paper Table S3: 20
        self.K = K
        self.L = L

        self.encoder = nn.Sequential(
            nn.Linear(num_features, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, 2 * self.latent_size),
        ).to(DEVICE)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, 3 * num_features),  # mean, scale, df per feature
        ).to(DEVICE)

        self.p_z = td.Independent(
            td.Normal(
                loc=torch.zeros(self.latent_size).to(DEVICE),
                scale=torch.ones(self.latent_size).to(DEVICE),
            ), 1
        )

    def init(self, seed: int) -> None:
        set_seed(seed)
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

    def compute_loss(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        x, mask = inputs  # mask: 1 = observed, 0 = missing
        batch_size = x.shape[0]

        out_encoder = self.encoder(x)
        mu = out_encoder[..., :self.latent_size]
        logvar = out_encoder[..., self.latent_size:]

        q_zgivenxobs = td.Independent(td.Normal(loc=mu, scale=nn.Softplus()(logvar) + 1e-6), 1)
        zgivenx = q_zgivenxobs.rsample([self.K])           # (K, B, latent)
        zgivenx_flat = zgivenx.reshape([self.K * batch_size, self.latent_size])

        out_decoder = self.decoder(zgivenx_flat)
        recon_means = out_decoder[..., :self.num_features]
        recon_scale = nn.Softplus()(out_decoder[..., self.num_features:2 * self.num_features]) + 0.001
        recon_df = nn.Softplus()(out_decoder[..., 2 * self.num_features:]) + 3  # df > 3 → finite variance

        data_flat = torch.Tensor.repeat(x, [self.K, 1]).reshape([-1, 1]).to(DEVICE)
        tiled_mask = torch.Tensor.repeat(mask, [self.K, 1]).to(DEVICE)

        log_pxgivenz_flat = td.StudentT(
            loc=recon_means.reshape([-1, 1]),
            scale=recon_scale.reshape([-1, 1]),
            df=recon_df.reshape([-1, 1]),
        ).log_prob(data_flat)

        log_pxgivenz = log_pxgivenz_flat.reshape([self.K * batch_size, self.num_features])
        logpxobsgivenz = torch.sum(log_pxgivenz * tiled_mask, 1).reshape([self.K, batch_size])
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq, 0))
        return neg_bound, {}

    def impute(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """mask: 1 = observed, 0 = missing"""
        L = self.L
        batch_size, p = x.shape

        out_encoder = self.encoder(x)
        mu = out_encoder[..., :self.latent_size]
        logvar = nn.Softplus()(out_encoder[..., self.latent_size:]) + 1e-6

        q_zgivenxobs = td.Independent(td.Normal(loc=mu, scale=logvar), 1)
        zgivenx = q_zgivenxobs.rsample([L])
        zgivenx_flat = zgivenx.reshape([L * batch_size, self.latent_size])

        out_decoder = self.decoder(zgivenx_flat)
        recon_means = out_decoder[..., :p]
        recon_scale = nn.Softplus()(out_decoder[..., p:2 * p]) + 0.001
        recon_df = nn.Softplus()(out_decoder[..., 2 * p:]) + 3

        data_flat = torch.Tensor.repeat(x, [L, 1]).reshape([-1, 1]).to(DEVICE)
        tiled_mask = torch.Tensor.repeat(mask, [L, 1]).to(DEVICE)

        log_pxgivenz_flat = td.StudentT(
            loc=recon_means.reshape([-1, 1]),
            scale=recon_scale.reshape([-1, 1]),
            df=recon_df.reshape([-1, 1]),
        ).log_prob(data_flat)

        log_pxgivenz = log_pxgivenz_flat.reshape([L * batch_size, p])
        logpxobsgivenz = torch.sum(log_pxgivenz * tiled_mask, 1).reshape([L, batch_size])
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        xgivenz = td.Independent(
            td.StudentT(loc=recon_means, scale=recon_scale, df=recon_df), 1
        )
        imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq, 0)  # (L, B)
        xms = xgivenz.sample().reshape([L, batch_size, p])
        xm = torch.einsum("ki,kij->ij", imp_weights, xms)

        xhat = torch.clone(x)
        xhat[~mask.bool()] = xm[~mask.bool()]
        return xhat
