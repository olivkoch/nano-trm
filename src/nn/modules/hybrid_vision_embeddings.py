import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridDINOEmbedding(nn.Module):
    """
    Concatenate learned color embeddings with DINOv2 visual features.
    DINOv2 might be better for geometric patterns.
    """

    def __init__(
        self,
        num_colors: int = 11,
        hidden_size: int = 512,
        dino_model_name: str = "facebook/dinov2-base",
        freeze_dino: bool = True,
    ):
        super().__init__()

        # Learned color embedding
        self.color_embedding = nn.Embedding(num_colors, hidden_size // 2)

        # DINOv2 vision encoder
        from transformers import AutoImageProcessor, AutoModel

        self.dino = AutoModel.from_pretrained(dino_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(dino_model_name)
        self.dino_hidden_size = self.dino.config.hidden_size  # 768 for base

        # Get image size from processor (handle different formats)
        if hasattr(self.image_processor, "size"):
            if isinstance(self.image_processor.size, dict):
                # Could be {'height': 224, 'width': 224} or {'shortest_edge': 224}
                if "height" in self.image_processor.size:
                    self.image_size = self.image_processor.size["height"]
                elif "shortest_edge" in self.image_processor.size:
                    self.image_size = self.image_processor.size["shortest_edge"]
                else:
                    self.image_size = 224  # Default
            else:
                self.image_size = self.image_processor.size
        else:
            self.image_size = 224  # Default for DINOv2

        # Calculate patch grid size
        self.patch_size = self.dino.config.patch_size  # 14 for dinov2-base
        self.num_patches_per_side = self.image_size // self.patch_size  # 224/14 = 16
        self.num_patches = self.num_patches_per_side**2  # 256

        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False

        # Project DINOv2 features
        self.dino_projection = nn.Linear(self.dino_hidden_size, hidden_size // 2)

        # Store ARC colors
        self.register_buffer("arc_colors_rgb", self._get_arc_colors_rgb())

    def _get_arc_colors_rgb(self) -> torch.Tensor:
        """Get ARC color palette as RGB tensor."""
        arc_colors = [
            "#000000",  # 0: Black
            "#0074D9",  # 1: Blue
            "#FF4136",  # 2: Red
            "#2ECC40",  # 3: Green
            "#FFDC00",  # 4: Yellow
            "#AAAAAA",  # 5: Gray
            "#F012BE",  # 6: Magenta
            "#FF851B",  # 7: Orange
            "#7FDBFF",  # 8: Sky blue
            "#870C25",  # 9: Maroon
            "#FFFFFF",  # 10: Padding (white)
        ]

        rgb_values = []
        for hex_color in arc_colors:
            hex_color = hex_color.lstrip("#")
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            rgb_values.append([r, g, b])

        return torch.tensor(rgb_values, dtype=torch.float32)

    def grid_to_rgb_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert integer grid to RGB image.

        Args:
            x: [batch, height, width] - integer values 0-10
        Returns:
            [batch, 3, image_size, image_size] - RGB image
        """
        batch, height, width = x.shape
        device = x.device

        # Map each integer to RGB
        rgb_image = self.arc_colors_rgb.to(device)[x]  # [B, H, W, 3]

        # Permute to [B, 3, H, W]
        rgb_image = rgb_image.permute(0, 3, 1, 2)

        # Resize to DINOv2 input size
        rgb_image = F.interpolate(
            rgb_image, size=(self.image_size, self.image_size), mode="nearest"
        )

        return rgb_image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, height, width]
        Returns:
            [batch, height*width, hidden_size]
        """
        batch, height, width = x.shape
        seq_len = height * width

        # 1. Learned embeddings
        color_emb = self.color_embedding(x).view(batch, seq_len, -1)

        # 2. DINOv2 features
        rgb_image = self.grid_to_rgb_image(x)

        # Use torch.no_grad() when frozen
        context = torch.no_grad() if not self.dino.training else torch.enable_grad()

        with context:
            dino_output = self.dino(pixel_values=rgb_image)
            # Remove CLS token: [B, num_patches+1, hidden] → [B, num_patches, hidden]
            dino_patches = dino_output.last_hidden_state[:, 1:, :]

        # Project
        dino_features = self.dino_projection(dino_patches)  # [B, num_patches, hidden_size/2]

        # Reshape to spatial grid
        dino_features_spatial = dino_features.view(
            batch, self.num_patches_per_side, self.num_patches_per_side, -1
        )  # [B, 16, 16, hidden_size/2]

        # Permute for interpolation: [B, hidden_size/2, 16, 16]
        dino_features_spatial = dino_features_spatial.permute(0, 3, 1, 2)

        # Interpolate to grid size
        dino_features_resized = F.interpolate(
            dino_features_spatial, size=(height, width), mode="bilinear", align_corners=False
        )  # [B, hidden_size/2, H, W]

        # Back to [B, H, W, hidden_size/2]
        dino_features_resized = dino_features_resized.permute(0, 2, 3, 1)
        dino_features_resized = dino_features_resized.view(batch, seq_len, -1)

        # 3. Concatenate
        combined = torch.cat([color_emb, dino_features_resized], dim=-1)

        return combined
