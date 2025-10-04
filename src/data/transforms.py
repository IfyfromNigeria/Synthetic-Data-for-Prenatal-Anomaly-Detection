import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transform_en():
    return A.Compose([
        A.RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0), ratio=(0.8, 1.2), p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=12,
                           border_mode=cv2.BORDER_CONSTANT, value=0, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transform_en():
    return A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

get_test_transform_en = get_val_transform_en
