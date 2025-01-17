from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
from transformers import AutoImageProcessor

from cv_misis_project.datasets.shanghai_dataset import ShanghaiDataModule
from cv_misis_project.models.csrnet_transformers import CSRNetTransformers

torch.set_float32_matmul_precision('medium')

# Конфиг
BATCH_SIZE = 8
TARGET_RESIZE = (256, 256)
RELOAD_DENSITY_MAPS = False
ENABLE_LOGGER = True

# CSRNET / Default
# TRANSFORM_PIPELINE = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize(TARGET_RESIZE),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

TRANSFORM_PIPELINE = AutoImageProcessor.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256')

DENSITY_TRANSFORM_PIPELINE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((TARGET_RESIZE[0] // 4, TARGET_RESIZE[1] // 4),
                      interpolation=transforms.InterpolationMode.BICUBIC),
])

EARLY_STOPPING = EarlyStopping(monitor="val_mse_loss", mode="min", patience=5)
CHECKPOINT_CALLBACK = ModelCheckpoint(monitor="val_mse_loss", mode="min", filename='result_model', save_top_k=1,
                                      verbose=True)
LOGGER = WandbLogger(log_model="all", project='cv-csrnet') if ENABLE_LOGGER else None

# Инициализация
trainer = Trainer(
    max_epochs=300,
    accelerator="gpu",
    devices=1,
    callbacks=[EARLY_STOPPING, CHECKPOINT_CALLBACK],
    logger=LOGGER,
    check_val_every_n_epoch=2
)

model = CSRNetTransformers(lr=1e-5)

data_module = ShanghaiDataModule(
    path_to_data=Path(__file__).parent.parent / 'dataset' / 'ShanghaiTech_Crowd_Counting_Dataset' / 'part_B_final',
    transform_func=TRANSFORM_PIPELINE,
    density_transform_func=DENSITY_TRANSFORM_PIPELINE,
    reload_density_maps=RELOAD_DENSITY_MAPS,
    batch_size=BATCH_SIZE
)

# Обучение
trainer.fit(model, datamodule=data_module)

# Тестирование
test_model = CSRNetTransformers.load_from_checkpoint(CHECKPOINT_CALLBACK.best_model_path)
trainer.test(test_model, datamodule=data_module)
