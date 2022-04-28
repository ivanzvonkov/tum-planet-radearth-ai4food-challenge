import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch
import wandb

from typing import List

stats_columns = [
    "blue_mean",
    "blue_std",
    "blue_min",
    "blue_max",
    "blue_median",
    "green_mean",
    "green_std",
    "green_min",
    "green_max",
    "green_median",
    "red_mean",
    "red_std",
    "red_min",
    "red_max",
    "red_median",
    "nir_mean",
    "nir_std",
    "nir_min",
    "nir_max",
    "nir_median",
    "ndvi_mean",
    "ndvi_std",
    "ndvi_min",
    "ndvi_max",
    "ndvi_median",
]

mean_min_max = ["mean", "min", "max"]


def get_ndvi(image_stack):
    if image_stack.shape[1] == 4:
        red = image_stack[:, [2]]
        nir = image_stack[:, [3]]
    elif image_stack.shape[1] == 13:
        red = image_stack[:, [3]]
        nir = image_stack[:, [7]]
    elif image_stack.shape[1] == 25:
        cols = [stats_columns.index(f"ndvi_{col}") for col in mean_min_max]
        return image_stack[:, cols]

    ndvi = (nir - red) / (nir + red)
    ndvi[nir == red] = 0
    assert np.isnan(ndvi).sum() == 0, "NDVI contains NaN"
    return ndvi


def get_nir(image_stack):
    if image_stack.shape[1] == 4:
        return image_stack[:, [3]]
    elif image_stack.shape[1] == 13:
        return image_stack[:, [7]]
    elif image_stack.shape[1] == 25:
        cols = [stats_columns.index(f"nir_{col}") for col in mean_min_max]
        return image_stack[:, cols]


def plot_preds(
    title: str,
    model,
    x: torch.Tensor,
    gp_indexes: List[int] = [],
    perturb_h_indexes: List[int] = [],
    perturb_amount: float = 1e-4,
    preds_with_dropout: int = 0,
    return_wandb_image: bool = True,
    predict_amount: int = 1,
):
    model.gp_inference_indexes = gp_indexes
    model.perturb_h_indexes = perturb_h_indexes
    model.perturb_amount = perturb_amount
    seq_length = x.shape[1]

    preds = []

    with torch.no_grad():
        x_np = x.cpu().numpy()
        for i in range(predict_amount):
            pred = model(x, training=False)
            pred_np = pred.cpu().numpy()
            preds.append(pred_np)

        if preds_with_dropout > 0:

            # Enable dropout
            if model.lstm_type == "unrolled" or model.lstm_type == "simple":
                print("Enabling dropout between LSTM cells")
                model.lstm.train()
            elif model.lstm_type == "pytorch":
                raise NotImplementedError("Enabling Dropout not implemented for pytorch lstm ")

            # Compute preds with dropout
            for i in range(preds_with_dropout):
                pred_np = model(x, training=False).cpu().numpy()
                preds.append(pred_np)

            if model.lstm_type == "unrolled":
                model.lstm.eval()

    input_timesteps = model.input_timesteps

    fig, axes = plt.subplots(nrows=x.shape[0], ncols=2, figsize=(20, x.shape[0] * 7))
    for i in range(x.shape[0]):
        ax = axes[i]
        actual_nir = get_nir(x_np[i])
        actual_ndvi = get_ndvi(x_np[i])

        is_mean_min_max = actual_nir.shape[1] == 3
        x = list(range(actual_nir.shape[0]))

        ax[0].plot(actual_nir[:, 0], label="Actual NIR", color="blue")
        ax[1].plot(actual_ndvi[:, 0], label="Actual NDVI", color="blue")
        if is_mean_min_max:
            ax[0].fill_between(
                x=x, y1=actual_nir[:, 1], y2=actual_nir[:, 2], color="blue", alpha=0.1
            )
            ax[1].fill_between(
                x=x, y1=actual_ndvi[:, 1], y2=actual_ndvi[:, 2], color="blue", alpha=0.1
            )

        pred_colors = ["orange", "green", "red", "purple"]
        for j, pred_np in enumerate(preds):

            pred_nir = get_nir(pred_np[i])
            pred_ndvi = get_ndvi(pred_np[i])

            ax[0].plot(pred_nir[:, 0], label=f"Generated NIR {j}", color=pred_colors[j])
            ax[1].plot(pred_ndvi[:, 0], label=f"Generated NDVI {j}", color=pred_colors[j])
            if is_mean_min_max:
                ax[0].fill_between(
                    x=x, y1=pred_nir[:, 1], y2=pred_nir[:, 2], color=pred_colors[j], alpha=0.1
                )
                ax[1].fill_between(
                    x=x, y1=pred_ndvi[:, 1], y2=pred_ndvi[:, 2], color=pred_colors[j], alpha=0.1
                )

        ax[0].axvline(x=input_timesteps, label="Predictions start", linestyle="--", color="gray")
        ax[1].axvline(x=input_timesteps, label="Predictions start", linestyle="--", color="gray")

        timesteps = [input_timesteps + j for j in gp_indexes if (input_timesteps + j) < seq_length]
        if len(gp_indexes) > 0:
            ax[0].plot(
                timesteps, pred_nir[timesteps, 0], "ro", color="red", label="GP used", markersize=2
            )
            ax[1].plot(
                timesteps, pred_ndvi[timesteps, 0], "ro", color="red", label="GP used", markersize=2
            )

        ax[0].set_title("NIR")
        ax[0].set_ylabel("NIR")
        ax[0].set_xlabel("Time interval")
        ax[1].set_title("NDVI")
        ax[1].set_ylabel("NDVI")
        ax[1].set_xlabel("Time interval")
        ax[0].legend()
        ax[1].legend()

    plt.suptitle(title, size=24)
    fig.subplots_adjust(top=0.2)
    fig.tight_layout()
    if return_wandb_image:
        return wandb.Image(fig)
    else:
        return fig
