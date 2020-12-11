"""Module with helper functions for DC-CRN module.
Modified from https://github.com/huyanxin/DeepComplexCRN,
by 'huyanxin', author of: https://arxiv.org/abs/2008.00264

Modifications include:
- refractoring code
- adapting to pytorch lightning format
- switched to yaml based hyperparameter loading/handling
- removing unused code
- other minor changes

Original license notice:
Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
# pylint: disable=arguments-differ, no-member, invalid-name, too-many-arguments
import numpy as np
from scipy.signal import get_window

import torch
import torch.nn as nn
import torch.nn.functional as F


class NavieComplexLSTM(nn.Module):
    """Complex LSTM layer"""

    def __init__(
        self,
        input_size,
        hidden_size,
        projection_dim=None,
        bidirectional=False,
        batch_first=False,
    ):
        super().__init__()

        self.input_dim = input_size // 2
        self.rnn_units = hidden_size // 2
        self.real_lstm = nn.LSTM(
            self.input_dim,
            self.rnn_units,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )
        self.imag_lstm = nn.LSTM(
            self.input_dim,
            self.rnn_units,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )
        if bidirectional:
            bidirectional = 2
        else:
            bidirectional = 1
        if projection_dim is not None:
            self.projection_dim = projection_dim // 2
            self.r_trans = nn.Linear(
                self.rnn_units * bidirectional, self.projection_dim
            )
            self.i_trans = nn.Linear(
                self.rnn_units * bidirectional, self.projection_dim
            )
        else:
            self.projection_dim = None

    def forward(self, x):
        if isinstance(x, list):
            real, imag = x
        elif isinstance(x, torch.Tensor):
            real, imag = torch.chunk(x, -1)
        real_out = (
            self.real_lstm(real)[0] - self.imag_lstm(imag)[0]
        )  # r2r_out - i2i_out
        imag_out = (
            self.real_lstm(imag)[0] + self.imag_lstm(real)[0]
        )  # i2r_out + r2i_out
        if self.projection_dim is not None:
            real_out = self.r_trans(real_out)
            imag_out = self.i_trans(imag_out)
        return [real_out, imag_out]

    def flatten_parameters(self):
        """flatten imaginary and real parameters"""
        self.imag_lstm.flatten_parameters()
        self.real_lstm.flatten_parameters()


def complex_cat(inputs, axis):
    """cat real and imaginary tensors"""
    real, imag = [], []
    for data in inputs:
        r, i = torch.chunk(data, 2, axis)
        real.append(r)
        imag.append(i)
    real = torch.cat(real, axis)
    imag = torch.cat(imag, axis)
    outputs = torch.cat([real, imag], axis)
    return outputs


class ComplexConv2d(nn.Module):
    """Complex conv2d layer"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=1,
        groups=1,
        causal=True,
        complex_axis=1,
    ):
        """
        in_channels: real+imag
        out_channels: real+imag
        kernel_size : input [B,C,D,T] kernel size in [D,T]
        padding : input [B,C,D,T] padding in [D,T]
        causal: if causal, will padding time dimension's left side,
                otherwise both
        """
        super().__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.complex_axis = complex_axis
        self.real_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=[self.padding[0], 0],
            dilation=self.dilation,
            groups=self.groups,
        )
        self.imag_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=[self.padding[0], 0],
            dilation=self.dilation,
            groups=self.groups,
        )

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.0)
        nn.init.constant_(self.imag_conv.bias, 0.0)

    def forward(self, x):
        if self.padding[1] != 0 and self.causal:
            x = F.pad(x, [self.padding[1], 0, 0, 0])
        else:
            x = F.pad(x, [self.padding[1], self.padding[1], 0, 0])

        if self.complex_axis == 0:
            real = self.real_conv(x)
            imag = self.imag_conv(x)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            # if isinstance(x, torch.Tensor):
            real, imag = torch.chunk(x, 2, self.complex_axis)

            real2real = self.real_conv(real)
            imag2imag = self.imag_conv(imag)
            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)

        return out


class ComplexConvTranspose2d(nn.Module):
    """ Complex convtranspose2d"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        output_padding=(0, 0),
        causal=False,
        complex_axis=1,
        groups=1,
    ):
        """
        in_channels: real+imag
        out_channels: real+imag
        """
        super().__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.real_conv = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=self.padding,
            output_padding=output_padding,
            groups=self.groups,
        )
        self.imag_conv = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=self.padding,
            output_padding=output_padding,
            groups=self.groups,
        )
        self.complex_axis = complex_axis

        nn.init.normal_(self.real_conv.weight, std=0.05)
        nn.init.normal_(self.imag_conv.weight, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.0)
        nn.init.constant_(self.imag_conv.bias, 0.0)

    def forward(self, x):

        if isinstance(x, torch.Tensor):
            real, imag = torch.chunk(x, 2, self.complex_axis)
        elif isinstance(x, tuple) or isinstance(x, list):
            real = x[0]
            imag = x[1]
        if self.complex_axis == 0:
            real = self.real_conv(x)
            imag = self.imag_conv(x)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)
        else:
            if isinstance(x, torch.Tensor):
                real, imag = torch.chunk(x, 2, self.complex_axis)

            real2real = self.real_conv(real)
            imag2imag = self.imag_conv(imag)
            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)

        return out


class ComplexBatchNorm(torch.nn.Module):
    """Complex BatchNorm
    Source: https://github.com/ChihebTrabelsi/deep_complex_networks/tree/pytorch
    from https://github.com/IMLHF/SE_DCUNet/blob/f28bf1661121c8901ad38149ea827693f1830715/models/layers/complexnn.py#L55
    """

    # pylint: disable=invalid-name
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        complex_axis=1,
    ):
        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        super().__init__()
        self.num_features = num_features // 2
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter("Wrr", None)
            self.register_parameter("Wri", None)
            self.register_parameter("Wii", None)
            self.register_parameter("Br", None)
            self.register_parameter("Bi", None)

        if self.track_running_stats:
            self.register_buffer("RMr", torch.zeros(self.num_features))
            self.register_buffer("RMi", torch.zeros(self.num_features))
            self.register_buffer("RVrr", torch.ones(self.num_features))
            self.register_buffer("RVri", torch.zeros(self.num_features))
            self.register_buffer("RVii", torch.ones(self.num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("RMr", None)
            self.register_parameter("RMi", None)
            self.register_parameter("RVrr", None)
            self.register_parameter("RVri", None)
            self.register_parameter("RVii", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-0.9, +0.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert xr.shape == xi.shape
        assert xr.size(1) == self.num_features

    def forward(self, x):
        # self._check_input_dim(xr, xi)

        xr, xi = torch.chunk(x, 2, axis=self.complex_axis)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)

        # Mean M Computation and Centering
        # Includes running mean update if training and running.
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr - Mr, xi - Mi

        # Variance Matrix V Computation
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr + self.eps
        Vri = Vri
        Vii = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, Vri, Vri, value=-1)
        # delta = Vrr * Vii - 1 * Vri * Vri
        # assert delta1 == delta

        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst = (s * t).reciprocal()
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (-Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = (
                self.Wrr.view(vdim),
                self.Wri.view(vdim),
                self.Wii.view(vdim),
            )
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        outputs = torch.cat([yr, yi], self.complex_axis)
        return outputs

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )


def init_kernels(win_len, fft_len, win_type=None, invers=False):
    if win_type == "None" or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)  # **0.5

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return (
        torch.from_numpy(kernel.astype(np.float32)),
        torch.from_numpy(window[None, :, None].astype(np.float32)),
    )


class ConvSTFT(nn.Module):
    def __init__(
        self,
        win_len,
        win_inc,
        fft_len=None,
        win_type="hamming",
        feature_type="real",
    ):
        super().__init__()

        if fft_len is None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, self.fft_len, win_type)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer("weight", kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, x):
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        x = F.pad(x, [self.win_len - self.stride, self.win_len - self.stride])
        outputs = F.conv1d(x, self.weight, stride=self.stride)

        if self.feature_type == "complex":
            return outputs

        dim = self.dim // 2 + 1
        real = outputs[:, :dim, :]
        imag = outputs[:, dim:, :]
        mags = torch.sqrt(real ** 2 + imag ** 2)
        phase = torch.atan2(imag, real)
        return mags, phase


class ConviSTFT(nn.Module):
    def __init__(
        self,
        win_len,
        win_inc,
        fft_len=None,
        win_type="hamming",
        feature_type="real",
    ):
        super().__init__()
        if fft_len is None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, self.fft_len, win_type, invers=True)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer("weight", kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer("window", window)
        self.register_buffer("enframe", torch.eye(win_len)[:, None, :])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """

        if phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)
            inputs = torch.cat([real, imag], 1)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs / (coff + 1e-8)
        # outputs = torch.where(coff == 0, outputs, outputs/coff)
        outputs = outputs[
            ..., self.win_len - self.stride : -(self.win_len - self.stride)
        ]

        return outputs


class Enhancer(nn.Module):
    def __init__(self, fft_len, hidden_layers_n, kernel_num, bidirectional, rnn_units):
        super().__init__()
        hidden_dim = fft_len // (2 ** (len(kernel_num)))
        fac = 2 if bidirectional else 1
        self.enhance = nn.LSTM(
            input_size=hidden_dim * kernel_num[-1],
            hidden_size=rnn_units,
            num_layers=hidden_layers_n,
            dropout=0.0,
            bidirectional=bidirectional,
            batch_first=False,
        )
        self.tranform = nn.Linear(rnn_units * fac, hidden_dim * kernel_num[-1])

    def forward(self, out):
        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)  # .T

        out = torch.reshape(out, [lengths, batch_size, channels * dims])
        out, _ = self.enhance(out)  # lstm
        out = self.tranform(out)  # linear
        out = torch.reshape(out, [lengths, batch_size, channels, dims])

        out = out.permute(1, 2, 3, 0)  # # .T
        return out


class ComplexEnhancer(nn.Module):
    def __init__(self, fft_len, hidden_layers_n, kernel_num, bidirectional, rnn_units):
        super().__init__()
        hidden_dim = fft_len // (2 ** (len(kernel_num)))

        rnns = []
        for idx in range(hidden_layers_n):
            rnns.append(
                NavieComplexLSTM(
                    input_size=hidden_dim * kernel_num[-1] if idx == 0 else rnn_units,
                    hidden_size=rnn_units,
                    bidirectional=bidirectional,
                    batch_first=False,
                    projection_dim=hidden_dim * kernel_num[-1]
                    if idx == hidden_layers_n - 1
                    else None,
                )
            )
            self.enhance = nn.Sequential(*rnns)

    def forward(self, out):
        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)  # .T
        out = self.forward_complex_lstm(out, channels, lengths, batch_size, dims)
        out = out.permute(1, 2, 3, 0)  # # .T
        return out

    def forward_complex_lstm(self, out, channels, lengths, batch_size, dims):
        r_rnn_in = torch.reshape(
            out[:, :, : channels // 2],
            [lengths, batch_size, channels // 2 * dims],
        )
        i_rnn_in = torch.reshape(
            out[:, :, channels // 2 :],
            [lengths, batch_size, channels // 2 * dims],
        )

        r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

        r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2, dims])
        i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2, dims])
        return torch.cat([r_rnn_in, i_rnn_in], 2)


class Decoder(nn.Module):
    def __init__(
        self,
        activation_function,
        kernel_num,
        kernel_size,
        use_cbn,
    ):
        super().__init__()
        # add decoder layers
        self.decoder = nn.ModuleList()
        for idx in range(len(kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    ComplexDecoderLayer(
                        activation_function, kernel_num, kernel_size, use_cbn, idx
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            kernel_num[idx] * 2,
                            kernel_num[idx - 1],
                            kernel_size=(kernel_size, 2),
                            stride=(2, 1),
                            padding=((kernel_size - 1) // 2, 0),
                            output_padding=(1, 0),
                        ),
                    )
                )

    def forward(self, out, encoder_out):
        # pass through decoder
        for idx in range(len(self.decoder)):
            out = complex_cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
            out = out[..., 1:]
        return out


class Encoder(nn.Module):
    """(Complex) encoder class"""

    def __init__(self, activation_function, kernel_num, kernel_size, use_cbn):
        super().__init__()
        self.encoder = nn.ModuleList()
        for idx in range(len(kernel_num) - 1):
            self.encoder.append(
                ComplexEncoderLayer(
                    activation_function, kernel_num, kernel_size, use_cbn, idx
                )
            )

    def forward(self, out):
        encoder_out = []
        for layer in self.encoder:
            out = layer(out)
            encoder_out.append(out)
        return encoder_out, out


class ComplexEncoderLayer(nn.Module):
    """(Complex) encoder layer class"""

    def __init__(self, activation_function, kernel_num, kernel_size, use_cbn, idx):
        super().__init__()
        self.layer = nn.Sequential(
            # nn.ConstantPad2d([0, 0, 0, 0], 0),
            ComplexConv2d(
                kernel_num[idx],
                kernel_num[idx + 1],
                kernel_size=(kernel_size, 2),
                stride=(2, 1),
                padding=((kernel_size - 1) // 2, 1),
            ),
            nn.BatchNorm2d(kernel_num[idx + 1])
            if not use_cbn
            else ComplexBatchNorm(kernel_num[idx + 1]),
            activation_function,
        )

    def forward(self, x):
        return self.layer(x)


class ComplexDecoderLayer(nn.Module):
    """(Complex) encoder layer class"""

    def __init__(self, activation_function, kernel_num, kernel_size, use_cbn, idx):
        super().__init__()
        self.layer = nn.Sequential(
            ComplexConvTranspose2d(
                kernel_num[idx] * 2,
                kernel_num[idx - 1],
                kernel_size=(kernel_size, 2),
                stride=(2, 1),
                padding=((kernel_size - 1) // 2, 0),
                output_padding=(1, 0),
            ),
            nn.BatchNorm2d(kernel_num[idx - 1])
            if not use_cbn
            else ComplexBatchNorm(kernel_num[idx - 1]),
            # nn.ELU()
            activation_function,
        )

    def forward(self, x):
        return self.layer(x)


class Mask(nn.Module):
    """Mask layer class"""

    def __init__(self, masking_mode):
        super().__init__()
        self.masking_mode = masking_mode

    def forward(self, out, real, imag):
        """Mask according to chosen strategy"""
        mask_real = F.pad(out[:, 0], [0, 0, 1, 0])
        mask_imag = F.pad(out[:, 1], [0, 0, 1, 0])
        if self.masking_mode == "E":
            mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
            mask_phase = torch.atan2(
                mask_imag / (mask_mags + 1e-8), mask_real / (mask_mags + 1e-8)
            )  # atan2(imag_phase, real_phase)

            est_mags = torch.tanh(mask_mags) * (
                torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
            )  # mask mags * spec_mags
            est_phase = torch.atan2(imag, real) + mask_phase
            real = est_mags * torch.cos(est_phase)
            imag = est_mags * torch.sin(est_phase)
        elif self.masking_mode == "C":
            real, imag = (
                real * mask_real - imag * mask_imag,
                real * mask_imag + imag * mask_real,
            )
        elif self.masking_mode == "R":
            real, imag = real * mask_real, imag * mask_imag
        return real, imag
