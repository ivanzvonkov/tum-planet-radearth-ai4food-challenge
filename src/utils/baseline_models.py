"""
For the further information about the baseline model please refer to
Kondmann, Lukas, et al. (2021),
[DENETHOR: The DynamicEarthNET dataset for Harmonized, inter-Operable, analysis-Ready, daily crop monitoring](https://openreview.net/pdf?id=uUa4jNMLjrL)
"""

import breizhcrops as bzh
import numpy as np
from torchvision import models
import pdb
import torch
from torch import nn
from src.utils.ltae import LTAE

from src.utils.pse import PixelSetEncoder


SUPPORTED_TEMPORAL_MODELS = [
    "inceptiontime",
    "lstm",
    "msresnet",
    "starrnn",
    "tempcnn",
    "transformermodel",
    "ltae",
]
SUPPORTED_SPATIAL_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnext50_32x4d",
    "resnext50_32x4d",
    "wide_resnet50_2",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
    "alexnet",
    "squeezenet1_0",
    "pixelsetencoder",
]


class SpatiotemporalModel(nn.Module):
    """
    A wrapper around torchvision (spatial) and breizhcrops models (temporal)
    """

    def __init__(
        self,
        spatial_backbone="mobilenet_v3_small",
        temporal_backbone="LSTM",
        input_dim=4,
        num_classes=9,
        sequencelength=365,
        pretrained_spatial=True,
        ta_model_path=None,
        ta_probability=0.0,
        ta_perturb_amount=1e-4,
        device="cpu",
    ):
        super(SpatiotemporalModel, self).__init__()

        if spatial_backbone not in ["none", "mean_pixel", "median_pixel", "random_pixel", "stats"]:
            self.spatial_encoder = SpatialEncoder(
                backbone=spatial_backbone, input_dim=input_dim, pretrained=pretrained_spatial
            )
            output_dim = self.spatial_encoder.output_dim
        else:
            output_dim = input_dim
        self.temporal_encoder = TemporalEncoder(
            backbone=temporal_backbone,
            input_dim=output_dim,
            num_classes=num_classes,
            sequencelength=sequencelength,
            ta_model_path=ta_model_path,
            ta_probability=ta_probability,
            ta_perturb_amount=ta_perturb_amount,
            device=device,
        )

        self.modelname = f"{spatial_backbone}_{temporal_backbone}"

        self.to(device)
        print("INFO: model initialized with name:{}".format(self.modelname))

    def forward(self, x):
        if hasattr(self, "spatial_encoder"):
            x = self.spatial_encoder(x)
        x = self.temporal_encoder(x)
        return x


class SpatialEncoder(torch.nn.Module):
    def __init__(self, backbone, input_dim=4, pretrained=False):
        super(SpatialEncoder, self).__init__()
        """
        A wrapper around torchvision models with some minor modifications for >3 input dimensions and features
        """
        assert (
            backbone in SUPPORTED_SPATIAL_MODELS
        ), f"spatial backbone model must be a supported torchvision model {SUPPORTED_SPATIAL_MODELS}"
        if "resnet" in backbone or "resnext" in backbone:
            self.model = models.__dict__[backbone](pretrained=pretrained)

            self.output_dim = self.model.fc.in_features

            # replace first conv layer to accomodate more spectral bands
            self.model.conv1 = nn.Conv2d(
                input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

            # remove last layer to get features instead of class scores
            modules = list(self.model.children())[:-1]
            self.model = nn.Sequential(*modules)

        elif "mobilenet_v3" in backbone:
            cnn = models.__dict__[backbone](pretrained=pretrained).features
            cnn[0][0] = nn.Conv2d(
                input_dim, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
            self.model = nn.Sequential(cnn, nn.AdaptiveAvgPool2d((1, 1)))
            self.output_dim = cnn[-1][0].out_channels
        elif "vgg" in backbone:
            self.model = models.__dict__[backbone](pretrained=pretrained)
            self.model.features[0] = nn.Conv2d(
                input_dim, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )
            self.output_dim = self.model.classifier[-1].out_features
        elif "alexnet" in backbone:
            self.model = models.__dict__[backbone](pretrained=pretrained)
            self.model.features[0] = nn.Conv2d(
                input_dim, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)
            )
            self.output_dim = self.model.classifier[-1].out_features
        elif "squeezenet" in backbone:
            self.model = models.__dict__[backbone](pretrained=pretrained)
            self.model.features[0] = nn.Conv2d(input_dim, 96, kernel_size=(7, 7), stride=(2, 2))
            self.output_dim = self.model.classifier[1].out_channels

        elif "pixelsetencoder" in backbone:
            self.model = PixelSetEncoder(
                input_dim=input_dim, mlp1=[input_dim, 32, 64], mlp2=[128, 128], with_extra=False
            )
            self.output_dim = self.model.output_dim

        self.modelname = backbone.replace("_", "-")

    def forward(self, x):
        x, mask = x
        if self.modelname == "pixelsetencoder":
            # Pixel-Set : Batch_size x (Sequence length) x Channel x Number of pixels
            # Pixel-Mask : Batch_size x (Sequence length) x Number of pixels
            x = self.model((x, mask))
            return x
        else:
            N, T, D, H, W = x.shape
            x = self.model(x.view(N * T, D, H, W))
            return x.view(N, T, x.shape[1])


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        backbone,
        input_dim,
        num_classes,
        sequencelength,
        ta_model_path,
        device,
        ta_probability=0.0,
        ta_perturb_amount=1e-4,
    ):
        super(TemporalEncoder, self).__init__()
        """
        A wrapper around Breizhcrops models for time series classification
        """
        backbone = backbone.lower()  # make case insensitive
        assert (
            backbone in SUPPORTED_TEMPORAL_MODELS
        ), f"temporal backbone model must be a supported breizhcrops model {SUPPORTED_TEMPORAL_MODELS}"

        if backbone == "lstm":
            self.model = bzh.models.LSTM(input_dim=input_dim, num_classes=num_classes)
        if backbone == "inceptiontime":
            self.model = bzh.models.InceptionTime(
                input_dim=input_dim, num_classes=num_classes, device=device
            )
        if backbone == "msresnet":
            self.model = bzh.models.MSResNet(input_dim=input_dim, num_classes=num_classes)
        if backbone == "starrnn":
            self.model = bzh.models.StarRNN(
                input_dim=input_dim, num_classes=num_classes, device=device
            )
        if backbone == "tempcnn":
            self.model = bzh.models.TempCNN(
                input_dim=input_dim, num_classes=num_classes, sequencelength=sequencelength
            )
        if backbone == "transformermodel":
            self.model = bzh.models.TransformerModel(input_dim=input_dim, num_classes=num_classes)

        if backbone == "ltae":
            self.model = LTAE(
                in_channels=input_dim,
                n_head=16,
                d_k=8,
                d_model=256,
                n_neurons=[256, 128],
                dropout=0.2,
                T=1000,
                len_max_seq=sequencelength,
                positions=None,
                return_att=False,
                mlp4=[128, 64, 32, num_classes],
            )

        self.modelname = backbone

        self.temporal_augmentation_model = None
        self.ta_probability = ta_probability

        if ta_model_path:
            from src.temporal_augmentor import TemporalAugmentor

            saved = torch.load(ta_model_path)
            config = saved["config"]
            self.temporal_augmentation_model = TemporalAugmentor(
                num_bands=config["input_dim"],
                hidden_size=config["lstm_hidden_size"],
                dropout=config["lstm_dropout"],
                input_timesteps=config["input_timesteps"],
                output_timesteps=config["output_timesteps"],
                gp_inference_indexes=[],
                device=device,
                gp_enabled=False,
                teacher_forcing=config["teacher_forcing"],
                lstm_layers=config["lstm_layers"],
                lstm_type=config["lstm_type"],
                perturb_h_indexes=[10, 20],
            )
            self.temporal_augmentation_model.load_state_dict(saved["model_state"])
            self.temporal_augmentation_model.eval()
            self.temporal_augmentation_model.perturb_amount = ta_perturb_amount
            print("\u2713 Model loaded")
        else:
            self.temporal_augmentation_model = None

    def forward(self, x):
        if type(x) == tuple:
            x, _ = x
        if (
            self.temporal_augmentation_model
            and self.model.training
            and (self.ta_probability > np.random.rand())
        ):
            with torch.no_grad():
                self.temporal_augmentation_model.lstm.train()
                x = self.temporal_augmentation_model(x, training=False)

        return self.model(x)


if __name__ == "__main__":
    model = SpatiotemporalModel()

    X = torch.ones([12, 365, 4, 32, 32])
    y_pred = model(X)
    print(y_pred)
