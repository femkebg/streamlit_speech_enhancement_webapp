""""Implementation of DC-CRN
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

from os.path import join

import yaml
import pytorch_lightning as pl
import torch


from data.model.utils import (
    Decoder,
    ConviSTFT,
    ConvSTFT,
    Encoder,
    Enhancer,
    ComplexEnhancer,
    Mask,
)


class DCCRN(pl.LightningModule):
    """Torch model with DCCRN structure"""

    def __init__(self, model_param):

        super().__init__()
        # set hyperparameters
        assert model_param["model_type"] == "DCCRN"
        self.model_param = model_param
        self.kernel_num = [2] + list(model_param["kernel"]["num"])
        self.forward_output_is_spec = False
        self.fft_len = model_param["fft"]["length"]

        # init stft
        self.stft = ConvSTFT(
            model_param["fft"]["window_length"],
            model_param["fft"]["window_increment"],
            model_param["fft"]["length"],
            model_param["fft"]["window_type"],
            "complex",
        )

        # init encoder layers
        self.encoder = Encoder(
            self.get_activation(model_param["endecoder_layers"]["activation"]),
            self.kernel_num,
            model_param["kernel"]["size"],
            model_param["endecoder_layers"]["complex_batch_normalization"],
        )

        # init enhance layers, complex or not, depending on config
        if model_param["hidden_layers"]["complex"]:
            self.enhance = ComplexEnhancer(
                model_param["fft"]["length"],
                model_param["hidden_layers"]["n"],
                self.kernel_num,
                model_param["hidden_layers"]["bidirectional"],
                model_param["hidden_layers"]["nodes"],
            )
        else:
            self.enhance = Enhancer(
                model_param["fft"]["length"],
                model_param["hidden_layers"]["n"],
                self.kernel_num,
                model_param["hidden_layers"]["bidirectional"],
                model_param["hidden_layers"]["nodes"],
            )

        # init decoder layers
        self.decoder = Decoder(
            self.get_activation(model_param["endecoder_layers"]["activation"]),
            self.kernel_num,
            model_param["kernel"]["size"],
            model_param["endecoder_layers"]["complex_batch_normalization"],
        )

        # init masking
        self.mask = Mask(model_param["masking_mode"])

        # init istft
        self.istft = ConviSTFT(
            model_param["fft"]["window_length"],
            model_param["fft"]["window_increment"],
            model_param["fft"]["length"],
            model_param["fft"]["window_type"],
            "complex",
        )

        # add example input for tensorboard graph creation
        self.example_input_array = torch.randn([16000]).clamp_(-1, 1)

    def forward(self, x):
        """Forward pass through network."""
        # turn input into stacked  real and imaginary spectograms
        specs = self.stft(x)
        real = specs[:, : self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1 :]
        out = torch.stack([real, imag], 1)[:, :, 1:]

        # pass through encoder
        encoder_out, out = self.encoder(out)

        # hidden layers
        out = self.enhance(out)

        # pass through the decoder
        out = self.decoder(out, encoder_out)

        # mask
        real, imag = self.mask(out, real, imag)

        # prepare output (time or freq domain)
        y_pred = torch.cat([real, imag], 1)
        if not self.forward_output_is_spec:
            y_pred = self.istft(y_pred)
            y_pred = torch.squeeze(y_pred, 1)
            y_pred = torch.clamp_(y_pred, -1, 1)
        return y_pred

    @classmethod
    def load(cls):
        """Load model and weights"""
        with open(join("data", "model", "params.yaml"), "r") as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)

        return DCCRN.load_from_checkpoint(
            join("data", "model", "best_model-v0.ckpt"),
            model_param=config["model_param"],
        )

    @staticmethod
    def get_activation(activation_string):
        """Get activation based on name."""
        activation_function = getattr(torch.nn, activation_string)
        return activation_function()
