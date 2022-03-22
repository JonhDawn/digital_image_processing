from __future__ import annotations

from typing import NamedTuple

import PIL.Image
import PIL.ImageTk
import numpy as np

from region import Region


class Color(NamedTuple):
    r: int
    g: int
    b: int
    a: int = 255

    @property
    def rgb(self) -> tuple[int, int, int]:
        return self.r, self.g, self.b

    def __add__(self, other):
        if isinstance(other, int) and 0 <= other <= 255:
            other = Color(other, other, other)
        if isinstance(other, Color):
            return Color(min(self.r + other.r, 255),
                         min(self.g + other.g, 255),
                         min(self.b + other.b, 255),
                         min(self.a + other.a, 255))
        raise TypeError(f"unsupported operand type(s) for +: 'Color' and '{type(other)}'")

    def __sub__(self, other):
        if isinstance(other, int) and 0 <= other <= 255:
            other = Color(other, other, other)
        if isinstance(other, Color):
            return Color(max(self.r - other.r, 0),
                         max(self.g - other.g, 0),
                         max(self.b - other.b, 0),
                         max(self.a - other.a, 0))
        raise TypeError(f"unsupported operand type(s) for +: 'Color' and '{type(other)}'")

    def closeness(self, other: Color):
        """ More the closeness is low, more the 2 colors seem similar. """
        return abs(self.r - other.r) + abs(self.g - other.g) \
               + abs(self.b - other.b) + abs(self.a - other.a)


class Image:
    """ An image is represented with a MxNx4 array of bytes (integer between 0 and 255). """

    def __init__(self, array: np.ndarray = None):
        if array is None:
            self.array = np.array([[[0, 0, 0, 0]]], dtype="uint8")
        else:
            assert len(array.shape) == 3
            if array.shape[-1] == 4:
                self.array = np.asarray(array, dtype="uint8")
            elif array.shape[-1] == 3:
                self.array = np.full(array.shape[:2] + (4,), 255, dtype="uint8")
                self.array[:, :, :3] = array
            else:
                raise ValueError("The shape of the given array has to be of the form ")

    @property
    def image(self) -> PIL.Image.Image:
        return PIL.Image.fromarray(self.array)

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def __getitem__(self, tup) -> Image:
        return Image(self.array[tup])

    def __setitem__(self, tup, value) -> None:
        if isinstance(value, Image):
            value = value.array
        self.array[tup] = value

    def superimpose(self, background: Image) -> None:
        """ Superimpose the image on a background. """
        assert self.shape == background.shape, f"{self.shape} != {background.shape}"
        foreground_color = self.array[:, :, :3] * np.asarray(self.array[:, :, 3:], dtype="uint16")
        background_color = background.array[:, :, :3] * np.asarray(255 - self.array[:, :, 3:], dtype="uint16")
        foreground_transparency = self.array[:, :, 3:] / 255
        background_transparency = background.array[:, :, 3:] / 255
        self.array[:, :, :3] = (foreground_color + background_color) // 255
        self.array[:, :, 3:] = 255 * (foreground_transparency + (1 - foreground_transparency) * background_transparency)

    def resize(self, height=None, width=None) -> None:
        """ Modify the height and the width of the image and keep the ratio height-width if it is possible. """
        if height is None:
            if width is None:
                return
            else:
                height = self.height * width // self.width
        else:
            if width is None:
                width = self.width * height // self.height
        self.array = np.array(self.image.resize((width, height)))

    def rotate(self, angle: float) -> None:
        """ Rotate the image by a given angle (keeping its dimensions). """
        image = self.image.rotate(angle, expand=True)
        self.array = np.array(image)

    def extend(self, left: int = 0, up: int = 0, right: int = 0, down: int = 0,
               background: Color = (0, 0, 0, 0)) -> None:
        """ Extend the image with the given background. """
        left_shape = (self.shape[0], left, 4)
        left_array = np.zeros(left_shape) + background
        right_shape = (self.shape[0], right, 4)
        right_array = np.zeros(right_shape) + background
        up_shape = (up, self.shape[1] + left + right, 4)
        up_array = np.zeros(up_shape) + background
        down_shape = (down, self.shape[1] + left + right, 4)
        down_array = np.zeros(down_shape) + background
        self.array = np.hstack((left_array, self.array, right_array))
        self.array = np.vstack((up_array, self.array, down_array))
        self.array = np.asarray(self.array, dtype="uint8")

    def stack(self, image: Image, vertically: bool = True) -> None:
        """ Add another image below the image if vertically is True and to the right otherwise. """
        if vertically:
            self.array = np.vstack((self.array, image.array))
        else:
            self.array = np.hstack((self.array, image.array))

    def get_photo(self) -> PIL.ImageTk.PhotoImage:
        """ Return the ImageTk.PhotoImage object from the image. """
        return PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.array))

    def copy(self) -> Image:
        """ Return a copy of the image object. """
        return Image(self.array.copy())

    def replace_color(self, old: Color, new: Color) -> None:
        """ Change the color of the pixels of a certain color. """
        mask = (self.array[:, :, :3] == old[:3]).all(axis=2)
        mask = mask.reshape(mask.shape + (1,))
        rgba = np.where(mask, new, self.array)
        self.array = np.asarray(rgba, dtype="uint8")

    def increase_brightness(self, increasing: int) -> None:
        """ Increase the brightness of each rgb value if increasing is positive and decrease them otherwise. """
        if increasing > 0:
            mask = self.array[:, :, :3] < (255 - increasing)
            self.array[:, :, :3] = np.where(mask, self.array[:, :, :3] + increasing, 255)
        elif increasing < 0:
            mask = self.array[:, :, :3] > increasing
            self.array[:, :, :3] = np.where(mask, self.array[:, :, :3] + increasing, 0)

    def show(self) -> None:
        """ Open a new window with the image. """
        self.image.show()

    def save(self, path: str) -> None:
        """ Save the image under the given filename. """
        self.image.save(path)

    @staticmethod
    def open(filename) -> Image:
        """ Create a new Image from the path of a png file. """
        pil_image = PIL.Image.open(filename).convert("RGBA")
        return Image(np.array(pil_image))

    @staticmethod
    def create(height: int = 1, width: int = 1, rgba: Color = (0, 0, 0, 0)) -> Image:
        """ Create a new Image of a fixed height and width with a single given color. """
        return Image(np.zeros((height, width, 4), dtype="uint8") + np.array(rgba, dtype="uint8"))

    @staticmethod
    def from_region(region: Region, inner_color: Color, background_color: Color) -> Image:
        """ Create a new Image that represents the given region. """
        left_bound, upper_bound, right_bound, lower_bound = map(int, region.bounds)
        height = lower_bound - upper_bound
        width = right_bound - left_bound
        image = Image.create(height=height, width=width)
        mask = np.zeros((height, width), dtype=bool)
        y, x = np.mgrid[upper_bound:lower_bound, left_bound:right_bound]
        for border in region.borders:
            vertical_mask = np.logical_and(border.upper(x) <= y, y <= border.lower(x))
            horizontal_mask = np.logical_and(border.interval.minimum <= x, x <= border.interval.maximum)
            border_mask = np.logical_and(vertical_mask, horizontal_mask)
            mask = np.logical_or(mask, border_mask)
        mask = mask.reshape(mask.shape + (1,))
        image.array = np.where(mask, inner_color, background_color)
        image.array = np.asarray(image.array, dtype="uint8")
        return image
