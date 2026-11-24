import torch
import torch.nn as nn
import torch.nn.functional as F

class AccidentNet(nn.Module):
    def __init__(self):
        super(AccidentNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Pool spatial features instead of flattening 128x128 into a very large
        # dense layer. This keeps CPU training practical and reduces overfit.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.global_pool(x).flatten(1)
        return self.fc(x)


class ResNet18AccidentNet(nn.Module):
    """ImageNet-pretrained ResNet-18 with a fresh binary classification head."""

    def __init__(self, pretrained_backbone=False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, images):
        return self.backbone(images)


class VideoAccidentNet(nn.Module):
    """ImageNet feature extractor with a bidirectional temporal GRU head."""

    def __init__(self, pretrained_backbone=False):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
        backbone = mobilenet_v3_small(weights=weights)
        self.frame_encoder = backbone.features
        self.frame_pool = backbone.avgpool
        self.temporal = nn.GRU(
            input_size=576,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(256, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(512, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def encode_frames(self, frames):
        features = self.frame_encoder(frames)
        features = self.frame_pool(features).flatten(1)
        return features

    def classify_embeddings(self, embeddings):
        sequence, _ = self.temporal(embeddings)
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
        attended = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        average = sequence.mean(dim=1)
        return self.classifier(torch.cat((attended, average), dim=1))

    def forward(self, clips):
        """Accept RGB clips shaped (batch, time, channels, height, width)."""
        batch, frames, channels, height, width = clips.shape
        flat = clips.reshape(batch * frames, channels, height, width)
        embeddings = self.encode_frames(flat).reshape(batch, frames, -1)
        return self.classify_embeddings(embeddings)


class R3D18AccidentNet(nn.Module):
    """Kinetics-pretrained 3D ResNet with a binary accident classification head."""

    def __init__(self, pretrained_backbone=False):
        super().__init__()
        from torchvision.models.video import R3D_18_Weights, r3d_18

        weights = R3D_18_Weights.KINETICS400_V1 if pretrained_backbone else None
        self.backbone = r3d_18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def encode_clips(self, clips):
        """Accept transformed clips shaped (batch, channels, time, height, width)."""
        return self.backbone(clips)

    def forward(self, clips):
        return self.classifier(self.encode_clips(clips))
