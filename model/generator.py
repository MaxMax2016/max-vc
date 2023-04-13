import torch
import torch.nn as nn
import numpy as np

from torch.nn import Conv1d
from torch.nn import ConvTranspose1d
from torch.nn.utils import weight_norm
from torch.nn.utils import remove_weight_norm

from .bigv import init_weights, SnakeBeta, AMPBlock
from .alias import Activation1d


class SpeakerAdapter(nn.Module):

    def __init__(self,
                 speaker_dim,
                 adapter_dim,
                 epsilon=1e-5
                 ):
        super(SpeakerAdapter, self).__init__()
        self.speaker_dim = speaker_dim
        self.adapter_dim = adapter_dim
        self.epsilon = epsilon
        self.W_scale = nn.Linear(self.speaker_dim, self.adapter_dim)
        self.W_bias = nn.Linear(self.speaker_dim, self.adapter_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)

    def forward(self, x, speaker_embedding):
        x = x.transpose(1, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= scale.unsqueeze(1)
        y += bias.unsqueeze(1)
        y = y.transpose(1, -1)
        return y


class Generator(torch.nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def __init__(self, hp):
        super(Generator, self).__init__()
        self.hp = hp
        self.num_kernels = len(hp.gen.resblock_kernel_sizes)
        self.num_upsamples = len(hp.gen.upsample_rates)
        # speaker adaper, 256 should change by what speaker encoder you use
        self.adapter = nn.ModuleList()
        # 1024 should change by your whisper out
        self.cond_pit = nn.Linear(2048, hp.audio.n_mel_channels)
        self.cond_pre = nn.Linear(1024, hp.audio.n_mel_channels)
        self.cond_pos = nn.Embedding(3, hp.audio.n_mel_channels)
        # pre conv
        self.conv_pit = nn.utils.weight_norm(
            Conv1d(hp.audio.n_mel_channels, hp.gen.upsample_initial_channel, 7, 1, padding=3))
        self.conv_pre = nn.utils.weight_norm(
            Conv1d(hp.audio.n_mel_channels, hp.gen.upsample_initial_channel, 7, 1, padding=3))
        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        self.upp = nn.ModuleList()
        for i, (u, k) in enumerate(zip(hp.gen.upsample_rates, hp.gen.upsample_kernel_sizes)):
            # spk
            self.adapter.append(SpeakerAdapter(
                256, hp.gen.upsample_initial_channel // (2 ** (i + 1))))
            # print(f'ups: {i} {k}, {u}, {(k - u) // 2}')
            # base
            self.ups.append(nn.ModuleList([
                weight_norm(ConvTranspose1d(hp.gen.upsample_initial_channel // (2 ** i),
                                            hp.gen.upsample_initial_channel // (
                                                2 ** (i + 1)),
                                            k, u, padding=(k - u) // 2))
            ]))
        for i, (u, k) in enumerate(zip(hp.gen.upsample_rates, hp.gen.upsample_kernel_sizes_p)):
            # print(f'upp: {i} {k}, {u}, {(k - u) // 2}')
            # pit
            self.upp.append(nn.ModuleList([
                weight_norm(ConvTranspose1d(hp.gen.upsample_initial_channel // (2 ** i),
                                            hp.gen.upsample_initial_channel // (
                                                2 ** (i + 1)),
                                            k, u, padding=(k - u) // 2))
            ]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = hp.gen.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(hp.gen.resblock_kernel_sizes, hp.gen.resblock_dilation_sizes):
                self.resblocks.append(AMPBlock(hp, ch, k, d))

        # post conv
        activation_post = SnakeBeta(ch, alpha_logscale=True)
        self.activation_post = Activation1d(activation=activation_post)
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        for i in range(len(self.upp)):
            self.upp[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, spk, x, pos, f0):
        # pit conv
        f = self.cond_pit(f0)               # [B, L, D]
        f = torch.transpose(f, 1, -1)       # [B, D, L]
        f = self.conv_pit(f)
        # pre conv
        x = self.cond_pre(x)                # [B, L, D]
        p = self.cond_pos(pos)
        x = x + p
        x = torch.transpose(x, 1, -1)       # [B, D, L]
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # adapter
            x = self.adapter[i](x, spk)
            # upsampling
            for i_up in range(len(self.upp[i])):
                f = self.upp[i][i_up](f)
            x = x + f
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.upp:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def inference(self, spk, ppg, pos, f0):
        MAX_WAV_VALUE = 32768.0
        audio = self.forward(spk, ppg, pos, f0)
        audio = audio.squeeze()  # collapse all dimension except time axis
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        return audio
