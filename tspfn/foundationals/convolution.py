import torch
import torch.nn as nn


class TimeSeriesConvolutionTokenizer(nn.Module):
    def __init__(self, ts_size: int = 1000, ts_num_channels: int = 16):
        super(TimeSeriesConvolutionTokenizer, self).__init__()
        self.ts_size = ts_size
        self.ts_num_channels = ts_num_channels

        C = ts_num_channels  # Number of input channels
        # self.tokenizer = nn.Sequential(
        #     # LAYER 1: Downsize 1000 -> 200 (Stride 5)
        #     nn.Conv1d(C, C*2, kernel_size=11, stride=5, padding=5, groups=C),
        #     nn.BatchNorm1d(C*2), nn.ReLU(inplace=True),

        #     # LAYER 2: Refine (Stride 1)
        #     nn.Conv1d(C*2, C*2, kernel_size=3, stride=1, padding=1, groups=C*2),
        #     nn.BatchNorm1d(C*2), nn.ReLU(inplace=True),

        #     # LAYER 3: Refine (Stride 1)
        #     nn.Conv1d(C*2, C*2, kernel_size=3, stride=1, padding=1, groups=C*2),
        #     nn.BatchNorm1d(C*2), nn.ReLU(inplace=True),

        #     # LAYER 4: Downsize 200 -> 40 (Stride 5)
        #     nn.Conv1d(C*2, C*4, kernel_size=5, stride=5, padding=0, groups=C*2),
        #     nn.BatchNorm1d(C*4), nn.ReLU(inplace=True),

        #     # LAYER 5: Refine (Stride 1)
        #     nn.Conv1d(C*4, C*4, kernel_size=3, stride=1, padding=1, groups=C*4),
        #     nn.BatchNorm1d(C*4), nn.ReLU(inplace=True),

        #     # LAYER 6: Refine (Stride 1)
        #     nn.Conv1d(C*4, C*4, kernel_size=3, stride=1, padding=1, groups=C*4),
        #     nn.BatchNorm1d(C*4), nn.ReLU(inplace=True),

        #     # LAYER 7: Downsize 40 -> 20 (Stride 2)
        #     nn.Conv1d(C*4, C*2, kernel_size=3, stride=2, padding=1, groups=C*2),
        #     nn.BatchNorm1d(C*2), nn.ReLU(inplace=True),

        #     # LAYER 8: Refine (Stride 1)
        #     nn.Conv1d(C*2, C*2, kernel_size=3, stride=1, padding=1, groups=C*2),
        #     nn.BatchNorm1d(C*2), nn.ReLU(inplace=True),

        #     # LAYER 9: Refine (Stride 1)
        #     nn.Conv1d(C*2, C*2, kernel_size=3, stride=1, padding=1, groups=C*2),
        #     nn.BatchNorm1d(C*2), nn.ReLU(inplace=True),

        #     # LAYER 10: Final projection back to original channel count (Stride 1)
        #     # To stay independent, we group by the original C=16.
        #     # This collapses the 2 filters per channel back into 1.
        #     nn.Conv1d(C*2, C, kernel_size=1, stride=1, groups=C),
        #     nn.BatchNorm1d(C)
        # )

        self.tokenizer = nn.Sequential(
            nn.Conv1d(
                in_channels=C,
                out_channels=C,
                kernel_size=5,
                stride=5,
                groups=C,
            ),  # T = 200 -> 40
            nn.ReLU(),
            nn.Conv1d(
                in_channels=C,
                out_channels=C,
                kernel_size=2,
                stride=2,
                groups=C,
            ),  # T = 40 -> 20
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) Time series input.
        Returns:
            embed: (B, C*T') Token embeddings.
        """
        embed = self.tokenizer(x)  # (B, C, T') where T' = 20
        # Flatten on channels and time
        embed = embed.flatten(start_dim=1)  # (B, C*T')
        return embed
