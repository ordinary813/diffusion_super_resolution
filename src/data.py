import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# An object to handle generality of images (cropping, downsampling)
class SuperResDataset(Dataset):
    def __init__(self, data_path, hr_size=128, scale_factor=2):
        super().__init__()
        
        # Find all images
        png_files = glob.glob(os.path.join(data_path, '**', '*.png'), recursive=True)
        jpg_files = glob.glob(os.path.join(data_path, '**', '*.jpg'), recursive=True)
        self.image_files = png_files + jpg_files
        
        self.hr_size = hr_size
        self.lr_size = hr_size // scale_factor
        
        # Transformation for High-Resolution (HR) images
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(self.hr_size),
            transforms.ToTensor(), # [0, 255] -> [0.0, 1.0]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [0, 1] -> [-1, 1]
        ])
        
        # Transformation for Low-Resolution (LR) images
        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(), # Needs to be a PIL image to resize
            transforms.Resize(self.lr_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.Resize(self.hr_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_files)

    # Gets an image index and returns a (LR, HR) pair
    def __getitem__(self, idx: int):
        hr_image_pil = Image.open(self.image_files[idx]).convert('RGB')
        
        # Process HR with transfrom
        hr_tensor = self.hr_transform(hr_image_pil).clone()

        hr_tensor_unnormalized = (hr_tensor + 1) / 2
        
        # Process LR with transform
        lr_tensor = self.lr_transform(hr_tensor_unnormalized)
        
        return lr_tensor, hr_tensor