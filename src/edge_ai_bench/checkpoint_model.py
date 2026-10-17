"""ResNet-Transformer model matching the downloaded PTB-XL checkpoint."""

from pathlib import Path
from typing import Any


def _torch_modules() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("Install the project torch dependency to use the checkpoint") from error
    return torch, nn, functional


def build_resnet_transformer(input_dim: int = 12, num_classes: int = 5, d_model: int = 256, nhead: int = 8, num_layers: int = 3, dropout: float = 0.5) -> Any:
    """Build the upstream ResNet-Transformer architecture used by Exp 12."""
    torch, nn, functional = _torch_modules()

    class ResBlock1D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels),
                )

        def forward(self, values: Any) -> Any:
            output = functional.relu(self.bn1(self.conv1(values)))
            output = self.bn2(self.conv2(output))
            output = output + self.shortcut(values)
            return functional.relu(output)

    class ResNetTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                ResBlock1D(64, 128, stride=2),
                nn.Dropout(0.2),
                ResBlock1D(128, d_model),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )

        def forward(self, values: Any) -> Any:
            values = self.cnn(values)
            values = values.transpose(1, 2)
            values = self.transformer(values)
            values = values.transpose(1, 2)
            return self.head(values)

    return ResNetTransformer()


def load_checkpoint(path: Path, device: str = "cpu") -> Any:
    """Strictly load the downloaded Exp 12 checkpoint."""
    torch, _, _ = _torch_modules()
    model = build_resnet_transformer()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def predict_logits(model: Any, signals: Any) -> Any:
    """Run deterministic checkpoint inference on channel-first ECG tensors."""
    torch, _, _ = _torch_modules()
    with torch.inference_mode():
        return model(signals)
