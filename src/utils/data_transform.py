"""
This code is generated by Ridvan Salih KUZU @DLR
LAST EDITED:  14.09.2021
ABOUT SCRIPT:
It defines a sample Data Transformer for augmentation
"""

import numpy as np
import torch
import pdb


class EOTransformer:
    """
    THIS CLASS DEFINE A SAMPLE TRANSFORMER FOR DATA AUGMENTATION IN THE TRAINING, VALIDATION, AND TEST DATA LOADING
    """

    def __init__(
        self,
        spatial_backbone="none",
        normalize=True,
        image_size=32,
        pse_sample_size=64,
        is_train=True,
    ):
        """
        THIS FUNCTION INITIALIZES THE DATA TRANSFORMER.
        :param spatial_backbone: It determine if spatial information will be exploited or not. It should be determined in line with the training model.
        :param normalize: It determine if the data to be normalized or not. Default is TRUE
        :param image_size: It determine how the data is partitioned into the NxN windows. Default is 32x32
        :return: None
        """
        self.spatial_backbone = spatial_backbone
        self.image_size = image_size
        self.normalize = normalize
        self.pse_sample_size = pse_sample_size
        self.is_train = is_train

    def normalize_and_torchify(self, image_stack, mask=None):
        # image_stack = image_stack * 1e-4

        # z-normalize
        if self.normalize:
            image_stack = image_stack * 1e-4
            # image_stack -= 0.1014 + np.random.normal(scale=0.01)
            # image_stack /= 0.1171 + np.random.normal(scale=0.01)

        return torch.from_numpy(np.ascontiguousarray(image_stack)).float(), torch.from_numpy(
            np.ascontiguousarray(mask)
        )

    def torchify(self, image_stack, mask=None):
        return torch.from_numpy(np.ascontiguousarray(image_stack)).float(), torch.from_numpy(
            np.ascontiguousarray(mask)
        )

    def transform(self, image_stack, mask=None, return_unnormalized_numpy=False):
        """
        THIS FUNCTION INITIALIZES THE DATA TRANSFORMER.
        :param image_stack: If it is spatial data, it is in size [Time Stamp, Image Dimension (Channel), Height, Width],
                            If it is not spatial data, it is in size [Time Stamp, Image Dimension (Channel)]
        :param mask: It is spatial mask of the image, to filter out uninterested areas. It is not required in case of having non-spatial data
        :return: image_stack, mask
        """
        assert mask is None or (mask > 0).any(), "mask all 0s"

        if self.spatial_backbone == "stats":
            mask = -1

        elif (
            self.spatial_backbone == "mean_pixel" or self.spatial_backbone == "none"
        ):  # average over field mask: T, D = image_stack.shape
            image_stack = np.mean(image_stack[:, :, mask > 0], axis=2)
            mask = -1  # mask is meaningless now but needs to be constant size for batching
        elif (
            self.spatial_backbone == "median_pixel"
        ):  # average over field mask: T, D = image_stack.shape
            image_stack = np.median(image_stack[:, :, mask > 0], axis=2)
            mask = -1  # mask is meaningless now but needs to be constant size for batching
        elif self.spatial_backbone == "pixelsetencoder":
            # Sample S pixels from image
            image_stack, mask = random_pixel_set(
                image_stack, mask, sample_size=self.pse_sample_size
            )
        elif self.spatial_backbone == "random_pixel":
            if self.is_train:
                masked_image = image_stack[:, :, mask > 0]
                idx = np.random.randint(0, masked_image.shape[2] - 1)
                image_stack = masked_image[:, :, idx]
            else:
                image_stack = image_stack[:, :, mask > 0]
            mask = -1

        elif self.spatial_backbone == "as_is":
            return image_stack, mask

        else:  # crop/pad image to fixed size + augmentations: T, D, H, W = image_stack.shape
            if image_stack.shape[2] >= self.image_size and image_stack.shape[3] >= self.image_size:
                image_stack, mask = random_crop(image_stack, mask, self.image_size)

            image_stack, mask = crop_or_pad_to_size(image_stack, mask, self.image_size)

            # rotations
            rot = np.random.choice([0, 1, 2, 3])
            image_stack = np.rot90(image_stack, rot, [2, 3])
            mask = np.rot90(mask, rot)

            # flip up down
            # if np.random.rand() < 0.5:
            #     image_stack = np.flipud(image_stack)
            #     mask = np.flipud(mask)

            # flip left right
            # if np.random.rand() < 0.5:
            #     image_stack = np.fliplr(image_stack)
            #     mask = np.fliplr(mask)

        # if self.is_train and self.temporal_augmentation and np.random.rand() < 0.8:
        #     assert (
        #         len(image_stack.shape) == 2
        #     ), f"Expecting an image stack with shape (Temporal, Bands) but got shape {image_stack.shape}"

        #     pdb.set_trace()
        #     synthetic_image_stack = self.temporal_augmentation_model(image_stack)

        #     return synthetic_image_stack, mask

        if return_unnormalized_numpy:
            return image_stack, mask

        image_stack, mask = self.normalize_and_torchify(image_stack, mask)
        return image_stack, mask


class PlanetTransform(EOTransformer):
    """
    THIS CLASS INHERITS EOTRANSFORMER FOR DATA AUGMENTATION IN THE PLANET DATA
    """

    # South Africa values
    per_band_mean = np.array([580.4186, 852.98376, 1136.9423, 2761.0286])
    per_band_std = np.array([179.95744, 209.66647, 384.34073, 476.9446])

    def __init__(
        self,
        include_bands=True,
        include_ndvi=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_bands = include_bands
        self.include_ndvi = include_ndvi

    def transform(self, image_stack, mask=None):

        image_stack, mask = super().transform(image_stack, mask, return_unnormalized_numpy=False)

        # if self.normalize:
        #     image_stack = (image_stack - self.per_band_mean) / self.per_band_std

        if self.include_ndvi:
            red = image_stack[:, 2]
            nir = image_stack[:, 3]
            ndvi = (nir - red) / (nir + red)
            ndvi[nir == red] = 0
            assert np.isnan(ndvi).sum() == 0, "NDVI contains NaN"
            ndvi = ndvi[:, np.newaxis]

        if self.include_bands and self.include_ndvi:
            image_stack = np.concatenate((image_stack, ndvi), axis=1)
        elif not self.include_bands and self.include_ndvi:
            image_stack = ndvi

        return self.torchify(image_stack, mask)


class Sentinel1Transform(EOTransformer):
    """
    THIS CLASS INHERITS EOTRANSFORMER FOR DATA AUGMENTATION IN THE SENTINEL-1 DATA
    """

    def __init__(
        self,
        include_bands=True,
        include_rvi=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_bands = include_bands
        self.include_rvi = include_rvi

    def transform(self, image_stack, mask=None):

        image_stack, mask = super().transform(image_stack, mask, return_unnormalized_numpy=True)

        if self.include_rvi:
            VV = image_stack[:, 0]
            VH = image_stack[:, 1]
            dop = VV / (VV + VH)
            radar_vegetation_index = (np.sqrt(dop)) * ((4 * (VH)) / (VV + VH))
            assert np.isnan(radar_vegetation_index).sum() == 0, "RVI contains NaN"
            radar_vegetation_index = radar_vegetation_index[:, np.newaxis]

        if self.include_bands and self.include_rvi:
            image_stack = np.concatenate((image_stack, radar_vegetation_index), axis=1)
        elif not self.include_bands and self.include_rvi:
            image_stack = radar_vegetation_index

        return self.normalize_and_torchify(image_stack, mask)


class Sentinel2Transform(EOTransformer):
    """
    THIS CLASS INHERITS EOTRANSFORMER FOR DATA AUGMENTATION IN THE SENTINEL-2 DATA
    """

    def __init__(
        self,
        include_bands=True,
        include_cloud=True,
        include_ndvi=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_bands = include_bands
        self.include_cloud = include_cloud
        self.include_ndvi = include_ndvi

    def transform(self, image_stack, mask=None):

        image_stack, mask = super().transform(image_stack, mask, return_unnormalized_numpy=True)
        if self.include_ndvi:
            nir = image_stack[:, 7]
            red = image_stack[:, 3]
            ndvi = (nir - red) / (nir + red)
            ndvi[nir == red] = 0
            assert np.isnan(ndvi).sum() == 0, "NDVI contains NaN"
            ndvi = ndvi[:, np.newaxis]

        if self.include_bands and self.include_cloud and self.include_ndvi:
            image_stack = np.concatenate((image_stack, ndvi), axis=1)
        elif self.include_bands and not self.include_cloud and self.include_ndvi:
            image_stack = np.concatenate((image_stack[:, :-1], ndvi), axis=1)
        elif not self.include_bands and self.include_cloud and self.include_ndvi:
            image_stack = np.concatenate((image_stack[:, -1:], ndvi), axis=1)
        elif not self.include_bands and not self.include_cloud and self.include_ndvi:
            image_stack = ndvi

        return self.normalize_and_torchify(image_stack, mask)


def random_crop(image_stack, mask, image_size):
    """
    THIS FUNCTION DEFINES RANDOM IMAGE CROPPING.
     :param image_stack: input image in size [Time Stamp, Image Dimension (Channel), Height, Width]
    :param mask: input mask of the image, to filter out uninterested areas [Height, Width]
    :param image_size: It determine how the data is partitioned into the NxN windows
    :return: image_stack, mask
    """

    H, W = image_stack.shape[2:]

    # skip random crop is image smaller than crop size
    if H - image_size // 2 <= image_size:
        return image_stack, mask
    if W - image_size // 2 <= image_size:
        return image_stack, mask

    h = np.random.randint(image_size, H - image_size // 2)
    w = np.random.randint(image_size, W - image_size // 2)

    image_stack = image_stack[
        :,
        :,
        h - int(np.floor(image_size // 2)) : int(np.ceil(h + image_size // 2)),
        w - int(np.floor(image_size // 2)) : int(np.ceil(w + image_size // 2)),
    ]
    mask = mask[
        h - int(np.floor(image_size // 2)) : int(np.ceil(h + image_size // 2)),
        w - int(np.floor(image_size // 2)) : int(np.ceil(w + image_size // 2)),
    ]

    return image_stack, mask


def crop_or_pad_to_size(image_stack, mask, image_size):
    """
    THIS FUNCTION DETERMINES IF IMAGE TO BE CROPPED OR PADDED TO THE GIVEN SIZE.
     :param image_stack: input image in size [Time Stamp, Image Dimension (Channel), Height, Width]
    :param mask: input mask of the image, to filter out uninterested areas [Height, Width]
    :param image_size: It determine how the data is cropped or padded into the NxN windows.
                       If the size of input image is larger than the given image size, it will be cropped, otherwise padded.
    :return: image_stack, mask
    """
    T, D, H, W = image_stack.shape
    hpad = image_size - H
    wpad = image_size - W

    # local flooring and ceiling helper functions to save some space
    def f(x):
        return int(np.floor(x))

    def c(x):
        return int(np.ceil(x))

    # crop image if image_size < H,W
    if hpad < 0:
        image_stack = image_stack[:, :, -c(hpad) // 2 : f(hpad) // 2, :]
        mask = mask[-c(hpad) // 2 : f(hpad) // 2, :]
    if wpad < 0:
        image_stack = image_stack[:, :, :, -c(wpad) // 2 : f(wpad) // 2]
        mask = mask[:, -c(wpad) // 2 : f(wpad) // 2]
    # pad image if image_size > H, W
    if hpad > 0:
        padding = (f(hpad / 2), c(hpad / 2))
        image_stack = np.pad(image_stack, ((0, 0), (0, 0), padding, (0, 0)))
        mask = np.pad(mask, (padding, (0, 0)))
    if wpad > 0:
        padding = (f(wpad / 2), c(wpad / 2))
        image_stack = np.pad(image_stack, ((0, 0), (0, 0), (0, 0), padding))
        mask = np.pad(mask, ((0, 0), padding))
    return image_stack, mask


def random_pixel_set(image_stack, mask, sample_size=64, jitter=(0.01, 0.05)):
    """
    Author: Vivien Sainte Fare Garnot
    https://github.com/VSainteuf/lightweight-temporal-attention-pytorch/blob/master/models/pse.py
    """

    field_pixels = image_stack[:, :, mask > 0]
    field_pixel_amount = int(mask.sum())
    if sample_size < field_pixel_amount:
        idx = np.random.choice(list(range(field_pixel_amount)), size=sample_size, replace=False)
        random_pixels = field_pixels[:, :, idx]
        S_mask = np.ones(sample_size, dtype=int)

    elif sample_size > field_pixel_amount:
        random_pixels = np.zeros((*field_pixels.shape[:2], sample_size))
        random_pixels[:, :, :field_pixel_amount] = field_pixels
        random_pixels[:, :, field_pixel_amount:] = np.stack(
            [random_pixels[:, :, 0]] * (sample_size - field_pixel_amount), axis=-1
        )

        S_mask = np.zeros(sample_size, dtype=int)
        S_mask[:field_pixel_amount] = 1

    else:
        random_pixels = field_pixels
        S_mask = np.ones(sample_size, dtype=int)

    S_mask = np.stack([S_mask] * field_pixels.shape[0], axis=0)  # Add temporal dimension to mask

    if jitter is not None:
        sigma, clip = jitter
        random_pixels = random_pixels + np.clip(
            sigma * np.random.randn(*random_pixels.shape), -1 * clip, clip
        )

    return random_pixels, S_mask
