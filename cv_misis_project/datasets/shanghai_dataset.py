from logging import Logger
from pathlib import Path
from typing import Callable

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.transforms import transforms

from cv_misis_project.datasets.image_data import ImageData
from cv_misis_project.utils.image_utils import load_image, get_density_map

logger = Logger(__name__, level='DEBUG')


class ShanghaiDataset(Dataset):
    def __init__(self,
                 img_ground_truth_mapping: dict[Path, Path],
                 transform: Callable | None = None,
                 target_transform: Callable | None = None):
        self.images = list(img_ground_truth_mapping.keys())
        self.ground_truths = list(img_ground_truth_mapping.values())
        self.transform_func = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> ImageData:
        x = load_image(self.images[idx])
        if self.transform_func:
            x = self.transform_func(x)
        y = torch.load(self.ground_truths[idx], weights_only=True)
        return x.pixel_values[0], y


class ShanghaiDataModule(pl.LightningDataModule):
    def __init__(self,
                 path_to_data: Path,
                 batch_size: int = 32,
                 stages_dirs: list[str] | None = None,
                 img_folder_name: str = 'images',
                 ground_truth_folder_name: str = 'ground_truth',
                 density_map_folder_name: str = 'density_maps',
                 transform_func: Callable | None = None,
                 density_transform_func: Callable | None = None,
                 reload_density_maps: bool = False, ):
        super().__init__()
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        if stages_dirs:
            self.stages_dirs = stages_dirs
        else:
            self.stages_dir = ['train_data', 'test_data']
        self.img_folder_name = img_folder_name
        self.ground_truth_folder_name = ground_truth_folder_name
        self.density_map_folder_name = density_map_folder_name
        self.transform_func = transform_func
        self.density_transform_func = density_transform_func
        self.reload_density_maps = reload_density_maps

    def prepare_data(self):
        # Создаем карты плотности для обучающих и тестовых изображений
        for stage_dir in self.stages_dir:
            stage_path = self.path_to_data / stage_dir
            images_path = stage_path / self.img_folder_name
            ground_truth_path = stage_path / self.ground_truth_folder_name
            density_folder_path = stage_path / self.density_map_folder_name

            if not density_folder_path.exists():
                density_folder_path.mkdir()

            for image_path in images_path.iterdir():
                ground_truth_file_path = ground_truth_path / f'GT_{image_path.stem}.mat'
                save_path = density_folder_path / f'{image_path.stem}.pt'

                # Проверяем, есть ли в кэше и не нужно ли перезагружать
                if save_path.exists() and not self.reload_density_maps:
                    logger.info(f'{save_path} карта плотности уже существует, пропускаем...')
                    continue
                elif save_path.exists() and self.reload_density_maps:
                    logger.info(f'Перезагружаем карту плотности для {image_path}')
                    save_path.unlink()
                logger.info(f'Создаем карту плотности для {image_path}')
                try:
                    density_map = get_density_map(image_path, ground_truth_file_path)
                    if self.density_transform_func:
                        # Корректируем плотность
                        resized_density_map = self.density_transform_func(density_map)
                        original_sum = density_map.sum()
                        resized_sum = resized_density_map.sum()
                        scale_factor = original_sum / resized_sum
                        resized_density_map *= scale_factor
                        density_map = resized_density_map
                except Exception as e:
                    logger.warning(
                        f'Произошла ошибка при создании карты плотности для {image_path}: {e}, пропускаем...')
                    continue
                with open(save_path, 'wb') as f:
                    # noinspection PyTypeChecker
                    torch.save(f=f, obj=density_map)

    def setup(self, stage: str):
        if stage == 'fit':
            train_stage_path = self.path_to_data / self.stages_dir[0]
            img_ground_truth_mapping = self._get_img_ground_truth_mapping(train_stage_path)
            shanghai_full = ShanghaiDataset(img_ground_truth_mapping, transform=self.transform_func)
            train_split, val_split = random_split(shanghai_full, [0.9, 0.1],
                                                  generator=torch.Generator().manual_seed(42))
            self.schanghai_train = train_split
            self.schanghai_val = val_split
        if stage == 'test':
            test_stage_path = self.path_to_data / self.stages_dir[1]
            img_ground_truth_mapping = self._get_img_ground_truth_mapping(test_stage_path)
            self.schanghai_test = ShanghaiDataset(img_ground_truth_mapping, transform=self.transform_func)

    def train_dataloader(self):
        return DataLoader(self.schanghai_train, batch_size=self.batch_size, shuffle=True, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.schanghai_val, batch_size=self.batch_size, shuffle=False, num_workers=15)

    def test_dataloader(self):
        return DataLoader(self.schanghai_test, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def _get_img_ground_truth_mapping(self, stage_path: Path) -> dict[Path, Path]:
        img_ground_truth_mapping = {}
        images_path = stage_path / self.img_folder_name
        density_path = stage_path / self.density_map_folder_name
        for image_path in images_path.iterdir():
            ground_truth_path = density_path / f'{image_path.stem}.pt'
            if not ground_truth_path.exists():
                logger.warning(f'Карта плотности для {image_path} не найдена, пропускаем...')
                continue
            img_ground_truth_mapping[image_path] = ground_truth_path
        return img_ground_truth_mapping


if __name__ == '__main__':
    TARGET_RESIZE = (256, 256)
    RELOAD_DENSITY_MAPS = False

    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(TARGET_RESIZE),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    density_transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(TARGET_RESIZE),
    ])
    data_module = ShanghaiDataModule(
        path_to_data=Path(
            __file__).parent.parent.parent / 'dataset' / 'ShanghaiTech_Crowd_Counting_Dataset' / 'part_A_final',
        transform_func=transform_pipeline,
        density_transform_func=density_transform_pipeline,
        reload_density_maps=RELOAD_DENSITY_MAPS
    )
    data_module.prepare_data()
    data_module.setup('fit')
    for data in data_module.train_dataloader():
        print(data)
