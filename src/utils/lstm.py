import torch
from torch import nn

from typing import Tuple, List, Optional

from src.utils.unrolled_lstm import UnrolledLSTM


class LSTM(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, batch_first: bool) -> None:
        super().__init__()
        
        # Dim (Batch, timesteps, channel)

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(
        self,
        image_stack: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        mask: Optional[Tuple[torch.Tensor, torch.Tensor]],
        gp_affected_timesteps: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        : image_stack: [Time Stamp, Channel, 1x1 Pixelvalue sample count]
        : mask: [time stamp, 1x1 pixelvalue sample count]
        : gp_affected_timesteps: The last N timesteps will be influenced by GP Trigger. GP Trigger will be applied at (T - N + 1)th time series.
        """
        assert len(image_stack.shape) == 3

        ### Setup
        input_timesteps, channels, pixel_sample_size = image_stack.shape
        output_timesteps = input_timesteps - gp_affected_timesteps

        image_stack = image_stack.reshape((*image_stack.shape, 1, 1)).transpose(2, 0, 1, 3, 4)
        image_stack = torch.tensor(image_stack).to('cuda')

        lstm = UnrolledLSTM(
            input_size=channels,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            batch_first=True,
        )

        hidden_tuple: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        assert input_timesteps >= 1

        predicted_output: List[torch.Tensor] = []

        for i in range(input_timesteps):
            input = image_stack[:, i : i + 1, :]
            """
            forget_state = self.forget_gate(torch.cat((x, hidden), dim=-1))
            RuntimeError: Tensors must have same number of dimensions: got 3 and 5
            """
            output, hidden_tuple = lstm(input, hidden_tuple)
            predicted_output.append(output)

        # we have already predicted the first output timestep (the last
        # output of the loop above)
        for i in range(output_timesteps - 1):
            output, hidden_tuple = lstm(output, hidden_tuple)
            predicted_output.append(output)

        return torch.cat(predicted_output, dim=1)
