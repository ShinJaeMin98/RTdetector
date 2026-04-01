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
        self.use_rt_attention_logits = True

    def _rt_clear_encoder(self):
        for layer in self.transformer_encoder.layers:
            layer._rt_tau = None
            layer._rt_delta = None

    def _rt_set_encoder(self, tau, delta):
        for layer in self.transformer_encoder.layers:
            layer._rt_tau = tau
            layer._rt_delta = delta

    def _rt_apply_decoder(self, decoder):
        for layer in decoder.layers:
            layer._rt_tau = self._rt_tau
            layer._rt_delta = self._rt_delta
            layer._rt_delta_self = getattr(self, '_rt_delta_self', None)
            layer._rt_delta_cross = getattr(self, '_rt_delta_cross', None)

    def _rt_clear_decoder(self, decoder):
        for layer in decoder.layers:
            layer._rt_tau = None
            layer._rt_delta = None
            layer._rt_delta_self = None
            layer._rt_delta_cross = None

    def encode(self, src,c,tgt):
        src = torch.cat((src, c), dim=2)
        src = src.permute(1, 0, 2)
        mean_enc = src.mean(1, keepdim=True).detach()
        src = src - mean_enc
        std_enc = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        self.std_enc = std_enc
        self.mean_enc = mean_enc
        src = src/std_enc
        # After cat + permute, src is (B, W, 2F); mean/std are (B, 1, 2F) (dim 1 = time).
        src_bw = src.contiguous()
        std_enc_bp = std_enc
        mean_enc_bp = mean_enc
        tau = self.tau_learner(src_bw, std_enc_bp).exp()
        delta = self.delta_learner(src_bw, mean_enc_bp)
        Bsz, Wsz = src.shape[0], src.shape[1]
        if tau is None:
            tau = torch.ones(Bsz, 1, 1, device=src.device, dtype=src.dtype)
        else:
            tau = tau.unsqueeze(1)
        if delta is None:
            delta = torch.zeros(Bsz, 1, Wsz, device=src.device, dtype=src.dtype)
        else:
            delta = delta.unsqueeze(1)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        if self.use_rt_attention_logits:
            self._rt_tau = tau
            self._rt_delta = delta
            self._rt_delta_self = delta.mean(dim=2, keepdim=True)
            self._rt_delta_cross = delta
            self._rt_set_encoder(tau, delta)
            memory = self.transformer_encoder(src)
            self._rt_clear_encoder()
            tgt = tgt.repeat(1, 1, 2)
            return tgt, memory
        memory = self.transformer_encoder(src)
        memory = memory.permute(1, 0, 2)
        delta_legacy = delta.permute(0, 2, 1)
        memory = memory*tau + delta_legacy
        memory = memory.permute(1, 0, 2)
        tgt = tgt.repeat(1, 1, 2)
        result = torch.mean(delta_legacy, dim=1, keepdim=True).permute(1, 0, 2)
        tau_b = tau.permute(1, 0, 2)
        tgt = tgt * tau_b + result
        return tgt, memory

    def forward(self, src, tgt):
        # Algorithm 1 requires three decoder outputs:
        # - O1  : Decoder1 with c = 0
        # - O2  : Decoder2 with c = 0
        # - Ô2  : Decoder2 with c = ||O1 - W||^2  (focus-based)
        c0 = torch.zeros_like(src)

        # O1 (Decoder1, c=0)
        em = self.encode(src, c0, tgt)
        if self.use_rt_attention_logits:
            self._rt_apply_decoder(self.transformer_decoder1)
        decoder_out = self.transformer_decoder1(*em).squeeze(0)
        if self.use_rt_attention_logits:
            self._rt_clear_decoder(self.transformer_decoder1)
        std_enc = self.std_enc.squeeze(1)
        mean_enc = self.mean_enc.squeeze(1)
        decoder_out = decoder_out * std_enc + mean_enc
        decoder_out = decoder_out.unsqueeze(0)
        O1 = self.fcn(decoder_out)

        # O2 (Decoder2, c=0)
        em = self.encode(src, c0, tgt)
        if self.use_rt_attention_logits:
            self._rt_apply_decoder(self.transformer_decoder2)
        decoder_out = self.transformer_decoder2(*em).squeeze(0)
        if self.use_rt_attention_logits:
            self._rt_clear_decoder(self.transformer_decoder2)
        std_enc = self.std_enc.squeeze(1)
        mean_enc = self.mean_enc.squeeze(1)
        decoder_out = decoder_out * std_enc + mean_enc
        decoder_out = decoder_out.unsqueeze(0)
        O2 = self.fcn(decoder_out)

        # Ô2 (Decoder2, c = (O1 - W)^2)
        c_focus = (O1 - src) ** 2
        em = self.encode(src, c_focus, tgt)
        if self.use_rt_attention_logits:
            self._rt_apply_decoder(self.transformer_decoder2)
        decoder_out = self.transformer_decoder2(*em).squeeze(0)
        if self.use_rt_attention_logits:
            self._rt_clear_decoder(self.transformer_decoder2)
        std_enc = self.std_enc.squeeze(1)
        mean_enc = self.mean_enc.squeeze(1)
        decoder_out = decoder_out * std_enc + mean_enc
        decoder_out = decoder_out.unsqueeze(0)
        Oh2 = self.fcn(decoder_out)

        # Keep ordering: (x1, x2_hat, x2_base) so existing inference code using [0],[1] still works.
        return O1, Oh2, O2
