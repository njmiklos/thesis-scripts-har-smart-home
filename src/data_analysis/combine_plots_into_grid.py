"""
Combines resized images into a single grid image.
"""
from PIL import Image
from pathlib import Path
from typing import List
import math

from utils.get_env import get_input_path


def load_and_resize_images(image_paths: List[Path], scale: float = 0.25) -> List[Image.Image]:
    """
    Loads and resizes a list of image files.

    Args:
        image_paths (List[Path]): List of image file paths to load.
        scale (float): Factor by which to scale down the images.

    Returns:
        List[Image.Image]: List of resized image objects.
    """
    images = []
    for path in image_paths:
        img = Image.open(path)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
        images.append(img)
    return images

def create_image_grid(images: List[Image.Image], columns: int = 5) -> Image.Image:
    """
    Arranges images into a grid.

    Args:
        images (List[Image.Image]): List of images to arrange.
        columns (int): Number of columns in the grid.

    Returns:
        Image.Image: Combined grid image.
    """
    rows = math.ceil(len(images) / columns)
    width, height = images[0].size
    grid_img = Image.new('RGB', (columns * width, rows * height), color=(255, 255, 255))

    for idx, img in enumerate(images):
        x = (idx % columns) * width
        y = (idx // columns) * height
        grid_img.paste(img, (x, y))

    return grid_img

def process_images(input_dir: Path, output_dir: Path) -> None:
    """
    Processes all image files in the input directory by resizing them and
    combining them into a grid, saved in the output directory.

    Args:
        input_dir (Path): Directory containing input image files.
        output_dir (Path): Directory to write the output combined image.
    """
    supported_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = sorted([p for p in input_dir.glob('*') if p.suffix.lower() in supported_extensions])[:19]

    if not image_paths:
        raise FileNotFoundError('No image files found in input directory.')

    images = load_and_resize_images(image_paths)
    grid = create_image_grid(images)

    output_file = output_dir / 'combined_grid.jpg'
    grid.save(output_file)
    print(f'Saved combined image grid to: {output_file}')

def process_directories(input_dir: Path) -> None:
    """
    Processes all directories in the input directory.

    Args:
        input_dir (Path): Directory containing input directories.
    """
    directories = []
    for d in input_dir.iterdir():
        if d.is_dir():
            directories.append(d)

    for dir in directories:
        process_images(dir, dir)


if __name__ == '__main__':
    input_path = get_input_path()
    if not input_path.exists():
        raise FileNotFoundError(f'Input directory {input_path} does not exist.')

    process_directories(input_path)