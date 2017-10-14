from PIL import Image, ImageDraw, ImageFont, ImageOps
from random import randint
import sys
import os
import math
import numpy
import random


# determines how much perspective distortion to use
# factor <= 1 - no distortion
# 1 >= factor <= 5 - slight distortion
# 5 >= factor <= 30 - large but usable distortion
# factor > 30 = very large distortion
image_perspective_factor = 60
image_size_factor = 100.0 * (1.0 / image_perspective_factor)
image_orientation_angle = 50
image_dimension = 25


def draw_circle():
    image = Image.open("baseimages/circle.png")
    return image


def draw_square():
    image = Image.open("baseimages/square.png")
    return image


def draw_plus():
    image = Image.open("baseimages/plus.png")
    return image


def draw_star():
    image = Image.open("baseimages/star.png")
    return image


def draw_waves():
    image = Image.open("baseimages/waves.png")
    return image


# Took this idea from : https://github.com/nathancahill/snippets/blob/master/image_perspective.py
def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(pb).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)


def generate_random_shifts(img_size, factor):
    w = img_size[0] / factor
    h = img_size[1] / factor
    shifts = []
    for s in range(0, 4):
        w_shift = (random.random() - 0.5) * w
        h_shift = (random.random() - 0.5) * h
        shifts.append((w_shift, h_shift))
    return shifts


def create_perspective(image):
    img_size = image.size
    w = img_size[0]
    h = img_size[1]
    shifts = generate_random_shifts(img_size, image_size_factor)
    coeffs = find_coeffs(
        [(shifts[0][0], shifts[0][1]),
            (w + shifts[1][0], shifts[1][1]),
            (w + shifts[2][0], h + shifts[2][1]),
            (shifts[3][0], h + shifts[3][1])], [(0, 0), (w, 0), (w, h), (0, h)])
    return image.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


# due to rotation and/or perspective we will need to fill in the background
def mask_image(img):
    mask = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.composite(img, mask, img)


def rotate_image(image):
    rotation_factor = math.pow(random.uniform(0.0, 1.0), 4)
    rotation_direction = (1, -1)[random.random() > 0.5]
    random_rotation_angle = int(math.floor(image_orientation_angle * rotation_factor * rotation_direction))
    return image.rotate(random_rotation_angle)


def resize(image):
    inv_img = ImageOps.invert(image.convert("RGB"))

    left, upper, right, lower = inv_img.getbbox()
    width = right - left
    height = lower - upper
    if width > height:
        # we want to add half the difference between width and height
        # to the upper and lower dimension
        padding = int(math.floor((width - height) / 2))
        upper -= padding
        lower += padding
    else:
        padding = int(math.floor((height - width) / 2))
        left -= padding
        right += padding

    image = image.crop((left, upper, right, lower))
    return image.resize((image_dimension, image_dimension), Image.LANCZOS)


def apply_random_transformation(image):
    image = create_perspective(image)
    image = rotate_image(image)
    image = mask_image(image)
    image = resize(image)
    return image

'''python zener_generator.py folder_name num_examples '''
if __name__ == "__main__":
    folder_name = sys.argv[1]
    num_examples = int(sys.argv[2])

    options = {0: draw_circle, 1: draw_square, 2: draw_plus, 3: draw_star, 4: draw_waves}
    symbols_dict = {0: "O", 1: "Q", 2: "P", 3: "S", 4: "W"}

    for i in xrange(num_examples):
        symbol = randint(0, 4)
        image = options[symbol]()
        image = apply_random_transformation(image)
        image_name = str(i+1) + "_" + symbols_dict[symbol] + ".png"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        image.save(folder_name + "/" + image_name)