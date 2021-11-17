import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
import pickle
import os

import ai8x.ai8x as ai8x

#model relu that USES ai8x modules
def _conv_stack(dilations, in_channels, out_channels, kernel_size,bias=True,**kwargs):
    """
    Create stack of dilated convolutional layers, outlined in WaveNet paper:
    https://arxiv.org/pdf/1609.03499.pdf
    """
    return nn.ModuleList(
        [
            ai8x.FusedConv1dReLU(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                dilation=d,
                bias=bias,
                **kwargs
            )
            for i, d in enumerate(dilations)
        ]
    )

class WaveNet(nn.Module):
    """
    1D TCN
    """
    def __init__(self, num_channels=12,
                dilation_depth=10, num_repeat=1,
                kernel_size=3,dilation_power=2,bias=True,
                in_bit_depth='None'):
        """
        num_channels: num INPUT channels
        """
        super().__init__()

        #num_classes HAVE TO LEAVE IT HERE
        # Limits
        #assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
        
        #bias=True #get bias from parameters

        #num_hidden_channels = 64
        #self.num_channels = num_channels
        num_hidden_channels = num_channels
        num_channels = 1
        kwargs = {'weight_bits': None, 'bias_bits': None, 'quantize_activation': False}

        ai8x.set_device(device =85, simulate= False, round_avg=False)

        dilations = [dilation_power ** d for d in range(dilation_depth)] * num_repeat

        #create dilated conv stack
        self.hidden = _conv_stack(dilations, num_hidden_channels, num_hidden_channels, kernel_size,bias=bias)
        self.residuals = _conv_stack(dilations, num_hidden_channels, num_hidden_channels, 1,bias=bias)

        #self.input_layer = ai8x.FusedConv1dReLU(
        # for first layer NO nonlinearity: simply linar mix
        self.input_layer = ai8x.Conv1d(
                in_channels=num_channels,#input channels
                out_channels=num_hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                bias=bias,
                **kwargs 
            )

        self.linear_mix = ai8x.Conv1d(
            in_channels=num_hidden_channels*dilation_depth*num_repeat, #no skips * dilation_depth * num_repeat,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True, #force no bias false for the last layer
            wide=True,
            #wide=True, #32 bit output!
            **kwargs
        )

        #init weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                pass
                #m.weight.data[:,:,:] = torch.tensor(0)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""

        self.outs = []
        
        skips = [] #stores skip connections

        out = self.input_layer(x)
        self.outs.append(out)

        for hidden, residual in zip(self.hidden, self.residuals):
            res = out
            out = hidden(out)

            skips.append(out) #append skip connections

            out = residual(out)

            out = out + res[:, :, -out.size(2) :]

            self.outs.append(out)

        out = torch.cat([s[:, :, -out.size(2) :] for s in skips], dim=1)
        out = self.linear_mix(out)

        #change linear mix SIZE!!!

        return out


def error_to_signal(y, y_pred):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    """
    y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
    return (y - y_pred).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + 1e-10)


def pre_emphasis_filter(x, coeff=0.95):
    return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)


class PedalNet(pl.LightningModule):
    def __init__(self, hparams):
        super(PedalNet, self).__init__()
        self.wavenet = WaveNet(
            num_channels=hparams["num_channels"],
            dilation_depth=hparams["dilation_depth"],
            num_repeat=hparams["num_repeat"],
            kernel_size=hparams["kernel_size"],
        )
        self.hparams = hparams

    def prepare_data(self):
        ds = lambda x, y: TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        data = pickle.load(open(os.path.dirname(self.hparams.model) + "/data.pickle", "rb"))
        self.train_ds = ds(data["x_train"], data["y_train"])
        self.valid_ds = ds(data["x_valid"], data["y_valid"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.wavenet.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams.batch_size, num_workers=4)

    def forward(self, x):
        return self.wavenet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = error_to_signal(y[:, :, -y_pred.size(2) :], y_pred).mean()
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = error_to_signal(y[:, :, -y_pred.size(2) :], y_pred).mean()
        return {"val_loss": loss}

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}
