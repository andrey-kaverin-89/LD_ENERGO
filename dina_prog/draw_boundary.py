import numpy as np
from PIL import Image
from tqdm import tqdm


def read_image(path):
    return np.array(Image.open(path))


def save_image(img, output_path):
    im = Image.fromarray(img)
    im.save(output_path)


def expand_the_border(img, expand_size=3):
    new_arr = np.copy(img)
    for x in range(expand_size, img.shape[0] - expand_size):
        for y in range(expand_size, img.shape[1] - expand_size):
            if img[x, y][0] == 255:
                mask = img[x - expand_size:x + expand_size, y - expand_size:y + expand_size] == 0
                new_arr[x - expand_size:x + expand_size, y - expand_size:y + expand_size][mask] = 128
    return new_arr


if __name__ == '__main__':
    folder_name = "dina_prog/"
    sample_name = "Sample.bmp"
    img = read_image(folder_name + sample_name)
    new_img = expand_the_border(img, 40)
    save_image(new_img, f"dina_prog/output/{sample_name}")

