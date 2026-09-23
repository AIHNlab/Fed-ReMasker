import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from Remasker.utils import MaskEmbed, get_1d_sincos_pos_embed, ActiveEmbed, CategoricalEmbed


LAYERNORM_EPS = 1e-5
VAR_EPS = 1e-6  # for norm_loss


class MaskedAutoencoder(nn.Module):

    """ Masked Autoencoder with Transformer backbone
    """

    def __init__(self, rec_len=25, embed_dim=64, depth=4, num_heads=4,
                 decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_field_loss=False, encode_func='linear',
                 categorical_vocab_sizes=[], cat_features=None, lambda_cat=1.0):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.cat_features = cat_features
        self.categorical_vocab_sizes = categorical_vocab_sizes
        # Fix 5: compute once here so forward passes never mutate model state
        self.num_features = [i for i in range(rec_len) if i not in (cat_features or [])]

        if encode_func == 'active':
            self.mask_embed = ActiveEmbed(rec_len, embed_dim)
        else:
            self.mask_embed = MaskEmbed(rec_len, embed_dim)

        self.cat_embeddings = nn.ModuleList([
            CategoricalEmbed(vocab_size, embed_dim) for vocab_size in self.categorical_vocab_sizes
        ])
        self.num_categorical = len(self.categorical_vocab_sizes)

        self.rec_len = rec_len
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, rec_len + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, rec_len + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 1, bias=True)  # decoder to patch

        self.decoder_pred_cat = nn.ModuleList([
            nn.Linear(decoder_embed_dim, self.categorical_vocab_sizes[idx])
            for idx, cat_feature in enumerate(self.cat_features)])
        # --------------------------------------------------------------------------

        self.norm_field_loss = norm_field_loss
        self.lambda_cat = lambda_cat

        self.register_buffer("num_idx", torch.tensor(self.num_features, dtype=torch.long),
                             persistent=False)
        self.register_buffer("cat_idx",
                             torch.tensor(list(self.cat_features or []), dtype=torch.long),
                             persistent=False)
        self.register_buffer("perm",
                             torch.tensor(self.num_features + list(self.cat_features or []),
                                          dtype=torch.long),
                             persistent=False)

        self.initialize_weights()

    def initialize_weights(self):

        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.mask_embed.rec_len, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.mask_embed.rec_len, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.mask_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, m, mask_ratio=0.5):

        # embed patches
        x_cat = x.index_select(2, self.cat_idx)

        parts = []
        if len(self.num_features) > 0:
            parts.append(self.mask_embed(x.index_select(2, self.num_idx)))
        for idx, cat_embedding in enumerate(self.cat_embeddings):
            parts.append(cat_embedding(x_cat[:, :, idx].unsqueeze(-1).long()))
        x_num = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

        # x_num is now in "reordered" space: numerical tokens first, then categorical.
        N, L, D = x_num.shape

        m_reordered = m.index_select(1, self.perm)
        observed = m_reordered > 0.5

        if self.training:
            noise = torch.rand((N, L), device=x_num.device)
            noise[~observed] = 2.0                      # missing positions sort last
            rank = torch.argsort(torch.argsort(noise, dim=1), dim=1)
            n_hide = (observed.sum(dim=1, keepdim=True).to(noise.dtype) * mask_ratio).floor().long()
            hide = observed & (rank < n_hide)
        else:
            hide = torch.zeros_like(observed)

        known = observed & ~hide

        assert self.mask_token.shape[-1] == D, (
            "mask_token is shared with the encoder, so decoder_embed_dim must equal "
            f"embed_dim (got {self.mask_token.shape[-1]} vs {D})")
        x_num = torch.where(known.unsqueeze(-1), x_num, self.mask_token.to(x_num.dtype))

        # add pos embed w/o cls token
        x = x_num + self.pos_embed[:, 1:, :]

        mask = hide.to(x.dtype)
        nask = known.to(x.dtype)
        ids_restore = torch.arange(L, device=x_num.device).unsqueeze(0).expand(N, -1)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x, mask, nask, ids_restore

    def forward_decoder(self, x, ids_restore):

        # embed tokens
        x = self.decoder_embed(x)

        n_masked = ids_restore.shape[1] + 1 - x.shape[1]
        if n_masked > 0:
            mask_tokens = self.mask_token.repeat(x.shape[0], n_masked, 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = x[:, 1:, :]  # [N, rec_len, D] in reordered space: num tokens first, then cat

        n_num = len(self.num_features)
        x_num = self.decoder_pred(x[:, :n_num])
        x_num = torch.tanh(x_num) / 2 + 0.5  # Scale to [0, 1]

        parts = [x_num]
        for idx, head in enumerate(self.decoder_pred_cat):
            parts.append(head(x[:, n_num + idx]).unsqueeze(-1))
        return parts[0] if len(parts) == 1 else torch.cat(parts, 1)

    def forward_loss(self, data, pred, mask, nask):
        """
        data: [N, 1, L]
        pred: [N, L]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        N, _, L = data.shape
        # target = self.patchify(data)
        target = data.squeeze(dim=1)
        if self.norm_field_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + VAR_EPS)**.5

        n_num = len(self.num_features)
        pred_flat = pred.squeeze(dim=2)

        loss_numerical = (pred_flat[:, :n_num] - target[:, self.num_features]) ** 2

        cat_parts = []
        start = n_num
        for idx, cat_feature in enumerate(self.cat_features):
            if idx > 0:
                start += self.categorical_vocab_sizes[idx - 1]
            vocab = self.categorical_vocab_sizes[idx]
            pred_logits = pred_flat[:, start:start + vocab]
            loss_categorical = F.cross_entropy(pred_logits,
                                               target[:, cat_feature].long(),
                                               reduction='none')
            cat_parts.append((loss_categorical / math.log(max(vocab, 2))).unsqueeze(-1))

        def masked_mean(values, weights):
            total = weights.sum()
            if total == 0:
                return values.sum() * 0.0
            return (values * weights).sum() / total

        loss = (masked_mean(loss_numerical, mask[:, :n_num]) + masked_mean(loss_numerical, nask[:, :n_num]))

        if cat_parts:
            loss_cat = torch.cat(cat_parts, dim=1)
            loss = loss + self.lambda_cat * (
                masked_mean(loss_cat, mask[:, n_num:]) + masked_mean(loss_cat, nask[:, n_num:]))

        return loss

    def forward(self, data, miss_idx, mask_ratio=0.5):

        latent, mask, nask, ids_restore = self.forward_encoder(data, miss_idx, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(data, pred, mask, nask)
        return loss, pred, mask, nask
