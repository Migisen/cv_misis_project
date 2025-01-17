from pathlib import Path

from transformers import AutoImageProcessor

from cv_misis_project.utils.image_utils import load_image


def check_image_preprocessor(path_to_img: Path, default_model: str = 'microsoft/swinv2-tiny-patch4-window8-256'):
    image = load_image(path_to_img)
    image_processor = AutoImageProcessor.from_pretrained(default_model)
    image = image_processor(images=image, return_tensors="pt")
    print(image)


if __name__ == '__main__':
    test_path = (Path(__file__).parent.parent.parent / 'dataset' / 'ShanghaiTech_Crowd_Counting_Dataset' /
                 'part_B_final' / 'train_data' / 'images' / 'IMG_1.jpg')
    check_image_preprocessor(test_path)
