# stdlib
# third party
import numpy as np
import pandas as pd
import torch
from torch import nn
from functools import partial
from Remasker.utils import MAEDataset, adjust_learning_rate
from .model_mae import MaskedAutoencoder
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils import parameters_to_vector

eps = 1e-8
MODEL_DTYPE = torch.float32
LAYERNORM_EPS = 1e-5


class Flare_Remasker(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.accum_iter = args.accum_iter
        self.min_lr = args.min_lr
        self.norm_field_loss = args.norm_field_loss
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.blr = args.blr
        self.warmup_epochs = args.warmup_epochs
        self.model = None
        self.norm_parameters = None

        self.embed_dim = args.embed_dim
        self.depth = args.depth
        self.decoder_depth = args.decoder_depth
        self.num_heads = args.num_heads
        self.mlp_ratio = args.mlp_ratio
        self.max_epochs = args.global_rounds
        self.mask_ratio = args.mask_ratio
        self.encode_func = args.encode_func
        self.categorical_vocab_sizes = args.categorical_vocab_sizes
        self.cat_features = args.cat_features
        self.local_epochs = args.local_epochs
        self.lambda_cat = args.cat_features
        self.optimizer = None
        self.device = args.device
        self.data_dim = args.data_dim
        # prox_mu only takes effect when strategy="fedprox" — zero otherwise
        # keeps the FedProx check below (`if self.prox_mu > 0`) as the single
        # source of truth, with no separate strategy check needed at use sites.
        self.prox_mu = args.prox_mu if args.strategy == "fedprox" else 0
        self.prepare_model()

    def prepare_model(self):
        self.model = MaskedAutoencoder(
            rec_len=self.data_dim,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=LAYERNORM_EPS),
            norm_field_loss=self.norm_field_loss,
            encode_func=self.encode_func,
            categorical_vocab_sizes=self.categorical_vocab_sizes,
            cat_features=self.cat_features,
            lambda_cat=self.lambda_cat
        )
        self.to(self.device)
        self.model.to(self.device).to(MODEL_DTYPE)
        for idx, _ in enumerate(self.cat_features):
            self.model.decoder_pred_cat[idx].to(self.device)

        # set optimizers
        # param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
        eff_batch_size = self.batch_size * self.accum_iter
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / 64
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.95),
            weight_decay=self.weight_decay)

    def prepare_data(self, X_raw: pd.DataFrame):
        X = X_raw.clone().cpu()
        M = 1 - (1 * (np.isnan(X)))
        M = M.float()
        X = torch.nan_to_num(X).float()
        # drop rows with fewer than 2 observed features — too sparse for attention
        enough_obs = M.sum(dim=1) >= 2
        if enough_obs.sum() == 0:
            return
        X = X[enough_obs]
        M = M[enough_obs]
        dataset = MAEDataset(X, M)
        self.dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset),
            batch_size=self.batch_size,
        )

    def local_train(self, global_vec, epoch):
        self.model.train()
        for local_epoch in range(self.local_epochs):
            self.optimizer.zero_grad()
            for it, (samples, masks) in enumerate(self.dataloader):
                frac = (local_epoch + it / len(self.dataloader)) / self.local_epochs
                adjust_learning_rate(self.optimizer, epoch - 1 + frac, self.lr,
                                     self.min_lr, self.max_epochs, self.warmup_epochs)

                samples = samples.unsqueeze(dim=1).to(self.device)
                masks = masks.to(self.device)

                loss, _, _, _ = self.model(samples, masks, mask_ratio=self.mask_ratio)
                if self.prox_mu > 0:
                    local_vec = parameters_to_vector(self.model.parameters())
                    # ||w - w^t||^2 (squared L2 norm, i.e. sum not mean) — matches
                    # the FedProx objective exactly: F_k(w) + (mu/2)||w - w^t||^2
                    # (Li et al., 2020)
                    proximal_term = torch.square(local_vec - global_vec).sum()
                    loss += (self.prox_mu / 2) * proximal_term
                loss = loss / self.accum_iter
                loss.backward()
                if (it + 1) % self.accum_iter == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

    def transform(self, X_raw: torch.Tensor, batch_size: int = 512):

        X = X_raw.clone()
        no, dim = X.shape
        X = X.cpu()

        # Set missing
        M = 1 - (1 * (np.isnan(X)))
        X = np.nan_to_num(X)

        X = torch.from_numpy(X).to(dtype=MODEL_DTYPE)
        M = M.to(dtype=MODEL_DTYPE)

        self.model.eval()

        num_features = self.model.num_features
        n_num = len(num_features)
        vocab = self.model.categorical_vocab_sizes

        imputed_data = torch.zeros((no, dim), dtype=MODEL_DTYPE)
        with torch.no_grad():
            for start_row in range(0, no, batch_size):
                end_row = min(start_row + batch_size, no)
                samples = X[start_row:end_row].unsqueeze(dim=1).to(self.device)
                masks = M[start_row:end_row].to(self.device)

                _, pred, _, _ = self.model(samples, masks)
                pred = pred.squeeze(dim=2)

                # scatter predictions back to their ORIGINAL column positions
                out = torch.zeros((end_row - start_row, dim), dtype=pred.dtype,
                                  device=pred.device)
                out[:, num_features] = pred[:, :n_num]
                start = n_num
                for idx, cat_feat_idx in enumerate(self.cat_features):
                    if idx > 0:
                        start += vocab[idx - 1]
                    pred_categorical = pred[:, start:start + vocab[idx]]
                    softmax_pred = torch.softmax(pred_categorical, dim=1)
                    out[:, cat_feat_idx] = torch.argmax(softmax_pred, dim=1).to(pred.dtype)

                imputed_data[start_row:end_row] = out.to(MODEL_DTYPE).cpu()

        if np.all(np.isnan(imputed_data.numpy())):
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            raise RuntimeError(err)

        M = M.cpu()
        # print('imputed', imputed_data, M)
        # print('imputed', M * np.nan_to_num(X_raw.cpu()) + (1 - M) * imputed_data)
        return M * X.cpu() + (1 - M) * imputed_data
