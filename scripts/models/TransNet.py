import copy
import math
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import Tensor
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from NeuralNetBase import NeuralNetBase, GEGLU


class MultiHeadAttention(nn.Module):
    def __init__(self, in_feature, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.scale = in_feature ** (-0.5)
        self.in_feature = in_feature
        self.num_heads = num_heads
        self.head_dim = in_feature // num_heads

        assert self.head_dim * num_heads == in_feature, "in_feature must be divisible by num_heads"

        self.scaling = math.sqrt(float(in_feature))
        # self.dropout = nn.Dropout(dropout)
        self.qkv_proj = nn.Linear(in_feature, 3 * in_feature, bias=False)
        self.out_proj = nn.Linear(in_feature, in_feature)
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        self.score = nn.Softmax(dim=-1)  # nn.Softmax(dim=-1) Sparsemax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [N, var, embedding dim]
        h = self.num_heads
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.score(sim)
        attn = self.dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    def __init__(self, in_feature, num_heads=8, dim_feedforward=512,
                 dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadAttention(in_feature, num_heads, dropout)

        self.layer_attn = nn.Sequential(
            nn.Identity(),
            MultiHeadAttention(in_feature, num_heads, dropout),
        )
        self.layer_feedforward = nn.Sequential(
            nn.Identity(),  # RMSNorm(in_feature), nn.LayerNorm(in_feature),
            nn.Linear(in_feature, dim_feedforward * 2),
            GEGLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, in_feature))

    def forward(self, src):
        src2 = self.layer_attn(src)
        src = src + src2

        src2 = self.layer_feedforward(src)
        src = src + src2
        return src


class Transformer(nn.Module):
    def __init__(self, in_feature, num_heads, dim_feedforward,
                 attn_dropout=0.1, num_transformer_layers=4):
        super(Transformer, self).__init__()
        encoder_layer = TransformerLayer(in_feature=in_feature,
                                         num_heads=num_heads,
                                         dim_feedforward=dim_feedforward,
                                         dropout=attn_dropout)
        self.num_layers = num_transformer_layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(self.num_layers)])
        self.norm = nn.Identity()  # RMSNorm(in_feature)  # nn.LayerNorm(in_feature)
        self.apply(self.init_weights)

    def forward(self, x):
        out = x
        for mod in self.layers:
            out = mod(out)
        out = self.norm(out)
        return out

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
            temp_state_dict = m.state_dict()
            temp_state_dict['weight'] *= 0.67 * self.num_layers ** (-0.25)
            m.load_state_dict(temp_state_dict)


class ResBasicBlock(nn.Module):

    def __init__(self, in_feature, activation, dropout=0.1):
        super(ResBasicBlock, self).__init__()
        # BN -> ReLu -> Weight -> BN -> ReLu -> Weight
        self.layers = nn.Sequential(
            nn.BatchNorm1d(num_features=in_feature),
            activation,
            nn.Linear(in_features=in_feature, out_features=in_feature),
            nn.BatchNorm1d(num_features=in_feature),
            activation,
            nn.Linear(in_features=in_feature, out_features=in_feature),
            nn.Dropout(dropout),

        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)

        return out + x


class TransTab(NeuralNetBase):
    # standard resnet
    def __init__(self,
                 num_continuous_vars: int,
                 nunique_categorical_vars: list,
                 dim_embedding: int,
                 dim_feedforward: int,
                 num_attn_head: int,
                 num_transformer_layers: int,
                 lr: float,
                 weight_decay: float = 0.0,
                 dropout_trans: float = 0.1,

                 ):
        super(TransTab, self).__init__()
        self.embedding_list = nn.ModuleList(
            [nn.Embedding(num_embeddings=i,
                          embedding_dim=dim_embedding) for i in nunique_categorical_vars])
        for i in self.embedding_list:
            self.init_embedding_layers(i, dim_embedding, num_transformer_layers)

        self.transformer = Transformer(in_feature=dim_embedding,
                                       num_heads=num_attn_head,
                                       dim_feedforward=dim_feedforward,
                                       attn_dropout=dropout_trans,
                                       num_transformer_layers=num_transformer_layers)

        # numeric variables --> same dimensions of categorical
        if num_continuous_vars > 0:
            self.numeric_embedding = nn.ModuleList([nn.Linear(1, dim_embedding) for _ in range(num_continuous_vars)])

            for i in self.numeric_embedding:
                self.init_embedding_layers(i, dim_embedding, num_transformer_layers)
        else:
            self.numeric_embedding = None
        # shrink var dimension to 1 (to reduce variables)
        self.post_embedding = nn.Sequential(
            nn.Conv1d(num_continuous_vars + len(nunique_categorical_vars), 1, 1, bias=False),
            nn.SiLU(),
        )
        self.init_linear_layers(self.post_embedding[0])

        self.out_layer = nn.Linear(dim_embedding, 1)
        self.init_linear_layers(self.out_layer)

        self.weight_decay = weight_decay
        self.lr = lr

    @staticmethod
    def init_linear_layers(layer):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)

    @staticmethod
    def init_embedding_layers(layer, embedding_dim, num_transformer_layers):
        nn.init.normal_(layer.weight, mean=0, std=embedding_dim ** (-0.5))
        temp_state_dict = layer.state_dict()
        temp_state_dict['weight'] *= (9 * num_transformer_layers) ** (-1 / 4)
        layer.load_state_dict(temp_state_dict)

    @staticmethod
    def _make_layer(in_dim, layers_dim, activation, dropout):
        layers = []
        in_feat = in_dim
        for i in layers_dim:
            layers += [nn.Linear(in_features=in_feat, out_features=i),
                       ResBasicBlock(i, activation=activation, dropout=dropout)]
            in_feat = i

        return nn.Sequential(*layers)

    def forward(self, x) -> Tensor:
        x_continuous, x_categorical = x
        out = torch.stack([self.embedding_list[i](x_categorical[:, i]) for i in range(x_categorical.shape[1])], dim=1)
        # ++++ embedding part ++++
        if x_continuous.shape[1] > 0:
            x_cont = torch.stack([self.numeric_embedding[i](x_continuous[:, i].view(-1, 1))
                                  for i in range(x_continuous.shape[1])], dim=1)

            out = torch.cat([out, x_cont], dim=1)

        out = self.transformer(out)
        out = self.post_embedding(out).squeeze(1)  # --> [N, embedding]

        # ++++ mlp part ++++
        # out = self.res_mlp(out)
        # out = self.act(out)
        out = self.out_layer(out)
        return out

    def _one_step(self, x_continuous, x_categorical, y, weight=None, log_name='None'):
        y_hat = self.forward((x_continuous, x_categorical))
        loss = self.criteria_weighted(y_hat, y, weight)
        self.log(log_name, loss, prog_bar=True, )
        return loss

    @staticmethod
    def early_stopping(patience, *args, **kwargs):
        return pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', patience=patience, verbose=True, *args,
                                                         **kwargs)

    def training_step(self, batch, batch_idx):
        return self._one_step(*batch, log_name='train_loss')

    def validation_step(self, batch, batch_idx):
        return self._one_step(*batch, log_name='val_loss')

    def test_step(self, batch, batch_idx):
        return self._one_step(*batch, log_name='test_loss')

    def configure_optimizers(self):

        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = GradualWarmupScheduler(optim, multiplier=1.0, total_epoch=self.warmup_epochs, )
        # assert isinstance(scheduler, ReduceLROnPlateau)
        scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.8, patience=5,
                                      threshold=0.0001, min_lr=1e-5, verbose=True)

        return {'optimizer': optim,
                "lr_scheduler": {'scheduler': scheduler, "monitor": 'val_loss'}}
