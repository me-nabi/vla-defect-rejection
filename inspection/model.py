"""Bottle inspection classifier: ResNet-18, shared over the front and back views."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from libero.libero.envs.inspection_defects import DEFECT_TYPES

CLASSES = ("good",) + DEFECT_TYPES

# Why a bottle is rejected, in plain words, per predicted defect type.
REASONS = {
    "crack": "Crack in the glass",
    "chip": "Chipped glass",
    "missing_cap": "Cap missing",
    "tilted_cap": "Cap not seated (tilted)",
    "low_fill": "Liquid below fill line",
    "particle": "Foreign particle in the drink",
    "dented": "Dented / crushed bottle",
}
IMAGE_SIZE = 320
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class InspectionNet(nn.Module):
    """Classifies a bottle from two views into "good" or one of the defect types."""

    def __init__(self, n_classes=len(CLASSES), pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = torchvision.models.resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * 512, n_classes))

    def forward(self, front, back):
        feats = self.backbone(torch.cat([front, back], dim=0))
        b = front.shape[0]
        return self.head(torch.cat([feats[:b], feats[b:]], dim=1))


def to_tensor(img):
    """uint8 HxWx3 image -> normalized float tensor 3xHxW."""
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    return (t - _MEAN) / _STD


class InspectionClassifier:
    """Loads a trained checkpoint and classifies (front, back) image pairs."""

    def __init__(self, path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.classes = tuple(ckpt["classes"])
        self.threshold = float(ckpt.get("threshold", 0.5))
        self.net = InspectionNet(len(self.classes), pretrained=False)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(self.device).eval()

    @torch.inference_mode()
    def probabilities(self, fronts, backs):
        f = torch.stack([to_tensor(x) for x in fronts]).to(self.device)
        b = torch.stack([to_tensor(x) for x in backs]).to(self.device)
        return torch.softmax(self.net(f, b), dim=1).cpu().numpy()

    def explain(self, front, back, defect):
        """Grad-CAM: where in each photo the evidence for `defect` is.

        Returns (front_heat, back_heat), each HxW in [0, 1], normalized jointly so a view
        that doesn't show the defect stays dark.
        """
        acts = {}

        def keep(_module, _inp, out):
            out.retain_grad()
            acts["a"] = out

        hook = self.net.backbone.layer4.register_forward_hook(keep)
        try:
            with torch.enable_grad():
                f = to_tensor(front)[None].to(self.device)
                b = to_tensor(back)[None].to(self.device)
                logits = self.net(f, b)
                self.net.zero_grad(set_to_none=True)
                logits[0, self.classes.index(defect)].backward()
        finally:
            hook.remove()
        a = acts["a"]                                   # (2 views, 512, 7, 7)
        weights = a.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * a).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=front.shape[:2], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / cam.max().clamp_min(1e-8)
        cam = cam.detach().cpu().numpy()
        return cam[0], cam[1]

    def __call__(self, front, back):
        """-> (is_defective, p_defect, predicted defect type or None)."""
        p = self.probabilities([front], [back])[0]
        p_defect = float(1.0 - p[self.classes.index("good")])
        is_defective = p_defect >= self.threshold
        defect = self.classes[1 + int(np.argmax(p[1:]))] if is_defective else None
        return is_defective, p_defect, defect
