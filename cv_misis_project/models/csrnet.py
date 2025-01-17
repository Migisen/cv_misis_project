import pytorch_lightning as pl
import torch
import wandb
from matplotlib import pyplot as plt
from torch import nn
from torchmetrics import MeanAbsoluteError
from torchvision import models


class CSRNet(pl.LightningModule):
    def __init__(self, lr: float = 1e-6):
        super().__init__()
        self.save_hyperparameters(ignore=['lr'])

        # Используем обученную VGG-16 как в оригинальной статье для front-end
        # Также в оригинальной статье авторы убирают классифицирующий часть VGG-16
        self.front_end = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features[:23]

        # В качестве бэкэнда используется dilated convolution layer,
        # он нужен, что бы не уменьшать размер выхода нашего front-end, при этом извлекая значимую информацию
        back_end_config = [512, 512, 512, 256, 128, 64]
        self.back_end = self.make_layers(back_end_config, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.loss_fn = nn.MSELoss()
        self.mae = MeanAbsoluteError()
        # В оригинальной статье 1e-6
        self.lr = lr

    def forward(self, x):
        x = self.front_end(x)
        x = self.back_end(x)
        x = self.output_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        image, density_map = batch
        pred = self(image)
        loss = self.loss_fn(pred, density_map)
        self.log('train_mse_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, density_map = batch
        pred = self(image)

        # Подсчёт количества людей
        pred_count = pred.sum()
        true_count = density_map.sum()
        self.mae.update(pred_count, true_count)

        loss = self.loss_fn(pred, density_map)
        self.log('val_mse_loss', loss, prog_bar=True)

        if batch_idx == 0:
            self.visualize_results(image, density_map, pred)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log('val_epoch_mae', self.mae.compute(), prog_bar=True)
        self.mae.reset()

    def test_step(self, batch, batch_idx):
        # Этап тестирования
        image, density_map = batch
        pred = self(image)

        # Подсчёт количества людей на тестовом наборе
        pred_count = pred.sum()
        true_count = density_map.sum()
        self.mae.update(pred_count, true_count)

        loss = self.loss_fn(pred, density_map)
        self.log('test_mse_loss', loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log('test_mae', self.mae.compute(), prog_bar=True)
        self.mae.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def visualize_results(self, images_batch: torch.Tensor, gt_batch: torch.Tensor, pred_batch: torch.Tensor):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        batch_size = images_batch.size(0)
        fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))

        if batch_size == 1:
            axes = [axes]  # Чтобы унифицировать обработку батча из одной строки

        for i, (img, gt_map, pred_map) in enumerate(zip(images_batch, gt_batch, pred_batch)):
            # Преобразование изображения в исходный вид
            img = img.cpu() * std[:, None, None] + mean[:, None, None]
            img = img.clamp(0, 1).permute(1, 2, 0).numpy()

            # Преобразование карт плотности
            gt_map = gt_map.squeeze().cpu().clamp(0, 1).numpy()
            pred_map = pred_map.squeeze().cpu().clamp(0, 1).numpy()

            # Подсчёт количества людей
            true_count = gt_map.sum()
            pred_count = pred_map.sum()

            # Отображаем изображения
            axes[i][0].imshow(img, interpolation='nearest')
            axes[i][0].set_title("Исходное изображение")
            axes[i][0].axis('off')

            axes[i][1].imshow(gt_map, cmap='viridis', interpolation='nearest')
            axes[i][1].set_title(f"Истинная плотность: {true_count:.2f}")
            axes[i][1].axis('off')

            axes[i][2].imshow(pred_map, cmap='viridis', interpolation='nearest')
            axes[i][2].set_title(f"Предсказанная плотность: {pred_count:.2f}")
            axes[i][2].axis('off')

        plt.tight_layout()

        # Логируем результат в wandb
        self.logger.experiment.log({
            'generated_density_map': [wandb.Image(fig, caption='Validation image snapshot')],
        })
        plt.close(fig)

    @staticmethod
    def make_layers(cfg, in_channels: int = 3, batch_norm: bool = False, dilation: bool = False):
        # Источник: https://github.com/leeyeehoo/CSRNet-pytorch/blob/master/model.py
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
