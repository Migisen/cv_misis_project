import torch
from torch.nn.functional import upsample
from transformers import Swinv2Model

from cv_misis_project.models.csrnet import CSRNet


class CSRNetTransformers(CSRNet):
    def __init__(self, lr: float = 1e-6, pretrained_model: str = 'microsoft/swinv2-tiny-patch4-window8-256'):
        super().__init__(lr=lr)
        self.front_end = Swinv2Model.from_pretrained(pretrained_model, output_hidden_states=True)
        back_end_config = [256, 128, 64]
        # self.upscaling_layers = nn.ModuleList([
        #     nn.ConvTranspose2d(
        #         in_channels=in_channels,
        #         out_channels=in_channels,
        #         kernel_size=kernel_size,  # Увеличение разрешения в 2 раза
        #         stride=kernel_size
        #     )
        #     for in_channels, kernel_size in zip([96, 192, 384], [1, 2, 4])
        #     # Каналы для каждого уровня SwinTransformer
        # ])
        self.back_end = self.make_layers(back_end_config, in_channels=2208, dilation=False)

    def forward(self, x):
        x = self.front_end(x).reshaped_hidden_states

        resized_features = []
        target_size = 64
        for layer_idx, features_output in enumerate(x):
            resized_features.append(upsample(features_output, size=(target_size, target_size), mode='bilinear'))

        x = torch.cat(resized_features, dim=1)
        x = self.back_end(x)  # (1, 64, 64, 64)
        x = self.output_layer(x)
        return x
