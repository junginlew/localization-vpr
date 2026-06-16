from pathlib import Path

import cv2
import numpy as np
import torch

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_rgb(image):
    """경로면 RGB uint8 (H, W, 3)로 읽고, 이미 배열이면 그대로 둠."""
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return image


class GlobalExtractor:
    """CosPlace 전역 descriptor 추출기. (D,) L2 정규화 벡터 (지도, 쿼리 공용)."""

    def __init__(self, dim=512, device=_DEVICE):
        self.model = torch.hub.load(
            "gmberton/cosplace", "get_trained_model",
            backbone="ResNet18", fc_output_dim=dim, trust_repo=True,
        ).to(device).eval()
        self.device = device
        self._mean = _IMAGENET_MEAN.to(device)
        self._std = _IMAGENET_STD.to(device)

    @torch.no_grad()
    def __call__(self, image):
        rgb = load_rgb(image)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        t = (t - self._mean) / self._std
        return self.model(t)[0].cpu().numpy()


class LocalExtractor:
    """XFeat 로컬 피처 추출기. (keypoints (N, 2), descriptors (N, 64)) 반환."""

    def __init__(self, top_k=4096, device=_DEVICE):
        self.model = torch.hub.load(
            "verlab/accelerated_features", "XFeat",
            pretrained=True, top_k=top_k, trust_repo=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        rgb = load_rgb(image)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)
        out = self.model.detectAndCompute(t)[0]
        return out["keypoints"].cpu().numpy(), out["descriptors"].cpu().numpy()
