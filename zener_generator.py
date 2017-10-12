from PIL import Image, ImageDraw, ImageFont
from random import randint
import sys


def draw_circle():
    image = Image.new('RGB', (25, 25), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((0, 0, 25, 24), fill=(255, 255, 255), outline=(0, 0, 0))
    return image


def draw_square():
    image = Image.new('RGB', (25, 25), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 22, 22), fill=(255, 255, 255), outline=(0, 0, 0))
    return image


def draw_plus():
    image = Image.new('RGB', (25, 25), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.line((0, 12.5, 25, 12.5), fill=(0, 0, 0))
    draw.line((12.5, 0, 12.5, 25), fill=(0, 0, 0))
    return image


def draw_star():
    return


def draw_waves():
    return


def apply_random_transformation(image):
    return image

'''python zener_generator.py folder_name num_examples '''
if __name__ == "__main__":
    folder_name = sys.argv[1]
    num_examples = sys.argv[2]

    options = {0: draw_circle, 1: draw_square, 2: draw_plus, 3: draw_star, 4: draw_waves}
    symbols_dict = {0: "O", 1: "Q", 2: "P", 3: "S", 4: "W"}

    for i in xrange(num_examples):
        symbol = randint(0, 2)
        image = options[symbol]()
        image = apply_random_transformation(image)
        image_name = str(i+1) + "_" + symbols_dict[symbol] + ".png"
        image.save(folder_name + "/" + image_name)