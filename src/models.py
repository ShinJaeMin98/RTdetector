import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from src.constants import *
torch.manual_seed(1)

class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)          # B x 1 x E
        x = torch.cat([x, stats], dim=1) # B x 2 x E
        x = x.view(batch_size, -1) # B x 2E

        y = self.backbone(x)       # B x O

        return y

class RTdetector_Normalization(nn.Module):
    def __init__(self, feats):
        super(RTdetector_Normalization, self).__init__()
        self.name = 'RTdetector'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.std_enc = torch.randn((self.batch,1, 2*feats))
        self.mean_enc = torch.randn((self.batch,1, 2*feats))
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
        self.tau_learner   = Projector(enc_in=2*feats, seq_len=self.n_window, hidden_dims=[16], hidden_layers=1, output_dim=1)
        self.delta_learner = Projector(enc_in=2*feats, seq_len=self.n_window, hidden_dims=[16], hidden_layers=1, output_dim=self.n_window)

    def encode(self, src,c,tgt):
        src = torch.cat((src, c), dim=2)
        src = src.permute(1, 0, 2)
        mean_enc = src.mean(1, keepdim=True).detach()
        src = src - mean_enc
        std_enc = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        self.std_enc = std_enc
        self.mean_enc = mean_enc
        src = src/std_enc
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2
	
class RTdetector_Destation(nn.Module):
    def __init__(self, feats):
        super(RTdetector_Destation, self).__init__()
        self.name = 'RTdetector'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.std_enc = torch.randn((self.batch,1, 2*feats))
        self.mean_enc = torch.randn((self.batch,1, 2*feats))
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
        self.tau_learner   = Projector(enc_in=2*feats, seq_len=self.n_window, hidden_dims=[16], hidden_layers=1, output_dim=1)
        self.delta_learner = Projector(enc_in=2*feats, seq_len=self.n_window, hidden_dims=[16], hidden_layers=1, output_dim=self.n_window)

    def encode(self, src,c,tgt):
        src = torch.cat((src, c), dim=2)
        src = src.permute(1, 0, 2)
        mean_enc = src.mean(1, keepdim=True).detach()
        src = src - mean_enc
        std_enc = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        self.std_enc = std_enc
        self.mean_enc = mean_enc

        src = src/std_enc
        tau = self.tau_learner(src,std_enc).exp()
        delta = self.delta_learner(src, mean_enc)

        tau = 1.0 if tau is None else tau.unsqueeze(1)
        delta = 0.0 if delta is None else delta.unsqueeze(1)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        memory = memory.permute(1, 0, 2)
        delta = delta.permute(0, 2, 1)
        memory = memory*tau + delta
        memory = memory.permute(1, 0, 2)
        tgt = tgt.repeat(1, 1, 2)
        result = torch.mean(delta, axis=1, keepdims=True)
        tgt = tgt.permute(1, 0, 2)
        tgt = tgt*tau+result
        tgt = tgt.permute(1, 0, 2)
        return tgt, memory

    def forward(self, src, tgt):
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2

class RTdetector_DeNormalization(nn.Module):
    def __init__(self, feats):
        super(RTdetector_DeNormalization, self).__init__()
        self.name = 'RTdetector'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.std_enc = torch.randn((self.batch,1, 2*feats))
        self.mean_enc = torch.randn((self.batch,1, 2*feats))
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
        self.tau_learner   = Projector(enc_in=2*feats, seq_len=self.n_window, hidden_dims=[16], hidden_layers=1, output_dim=1)
        self.delta_learner = Projector(enc_in=2*feats, seq_len=self.n_window, hidden_dims=[16], hidden_layers=1, output_dim=self.n_window)

    def encode(self, src,c,tgt):
        src = torch.cat((src, c), dim=2)
        src = src.permute(1, 0, 2)
        mean_enc = src.mean(1, keepdim=True).detach()
        src = src - mean_enc
        std_enc = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        self.std_enc = std_enc
        self.mean_enc = mean_enc
        src = src/std_enc
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        decoder_out = self.transformer_decoder1(*self.encode(src, c, tgt)).squeeze(0)
        std_enc = self.std_enc.squeeze(1)
        mean_enc = self.mean_enc.squeeze(1)
        decoder_out = decoder_out*std_enc+mean_enc
        decoder_out = decoder_out.unsqueeze(0)
        x1 = self.fcn(decoder_out)
        c = (x1 - src) ** 2
        decoder_out = self.transformer_decoder1(*self.encode(src, c, tgt)).squeeze(0)
        std_enc = self.std_enc.squeeze(1)
        mean_enc = self.mean_enc.squeeze(1)
        decoder_out = decoder_out*std_enc+mean_enc
        decoder_out = decoder_out.unsqueeze(0)
        x2 = self.fcn(decoder_out)
        return x1, x2
	
class RTdetector(nn.Module):
    def __init__(self, feats):
        super(RTdetector, self).__init__()
        self.name = 'RTdetector'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.std_enc = torch.randn((self.batch,1, 2*feats))
        self.mean_enc = torch.randn((self.batch,1, 2*feats))
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
        self.tau_learner   = Projector(enc_in=2*feats, seq_len=self.n_window, hidden_dims=[16], hidden_layers=1, output_dim=1)
        self.delta_learner = Projector(enc_in=2*feats, seq_len=self.n_window, hidden_dims=[16], hidden_layers=1, output_dim=self.n_window)

    def encode(self, src,c,tgt):
        src = torch.cat((src, c), dim=2)
        src = src.permute(1, 0, 2)
        mean_enc = src.mean(1, keepdim=True).detach()
        src = src - mean_enc
        std_enc = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        self.std_enc = std_enc
        self.mean_enc = mean_enc
        src = src/std_enc
        tau = self.tau_learner(src,std_enc).exp()
        delta = self.delta_learner(src, mean_enc)
        tau = 1.0 if tau is None else tau.unsqueeze(1)
        delta = 0.0 if delta is None else delta.unsqueeze(1)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        memory = memory.permute(1, 0, 2)
        delta = delta.permute(0, 2, 1)
        memory = memory*tau + delta
        memory = memory.permute(1, 0, 2)
        tgt = tgt.repeat(1, 1, 2)
        result = torch.mean(delta, axis=1, keepdims=True)
        tgt = tgt.permute(1, 0, 2)
        tgt = tgt*tau+result
        tgt = tgt.permute(1, 0, 2)
        return tgt, memory

    def forward(self, src, tgt):
        c = torch.zeros_like(src)
        decoder_out = self.transformer_decoder1(*self.encode(src, c, tgt)).squeeze(0)
        std_enc = self.std_enc.squeeze(1)
        mean_enc = self.mean_enc.squeeze(1)
        decoder_out = decoder_out*std_enc+mean_enc
        decoder_out = decoder_out.unsqueeze(0)
        x1 = self.fcn(decoder_out)
        c = (x1 - src) ** 2
        decoder_out = self.transformer_decoder2(*self.encode(src, c, tgt)).squeeze(0)
        std_enc = self.std_enc.squeeze(1)
        mean_enc = self.mean_enc.squeeze(1)
        decoder_out = decoder_out*std_enc+mean_enc
        decoder_out = decoder_out.unsqueeze(0)
        x2 = self.fcn(decoder_out)
        return x1, x2
