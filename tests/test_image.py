from unittest import TestCase

import PIL.Image
import numpy as np

from image import Image, Color


class TestImage(TestCase):
    def setUp(self) -> None:
        self.image = Image(np.array([[[256, 257, 258, 259], [260, 261, 262, 263]]], dtype="uint8"))
        self.image2 = Image()
        self.image3 = Image(np.array([[[1, 1, 1]]], dtype="uint8"))

    def test_init(self):
        for image in (self.image, self.image2):
            self.assertIsInstance(image, Image)
            self.assertEqual(image.array.dtype, np.uint8)
        self.assertTrue((self.image.array == np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]], dtype="uint8")).all())
        self.assertTrue((self.image2.array == np.array([[[0, 0, 0, 0]]], dtype="uint8")).all())
        self.assertTrue((self.image3.array == np.array([[[1, 1, 1, 255]]], dtype="uint8")).all())
        image = Image()
        self.assertRaises(ValueError, Image.__init__, image, np.array([[[0, 0, 0, "a"]]]))
        # Array elements have to be integers
        self.assertRaises(AssertionError, Image.__init__, image, np.array([[0, 0, 0]]))
        # Array's shape has to be (m, n, 3) or (m, n, 4)
        self.assertRaises(ValueError, Image.__init__, image, np.array([[[0, 0]]]))
        # Array's shape has to be (m, n, 3) or (m, n, 4)

    def test_image(self):
        self.assertIsInstance(self.image.image, PIL.Image.Image)

    def test_size(self):
        self.assertEqual(self.image.size, (2, 1))

    def test_height(self):
        self.assertEqual(self.image.height, 1)

    def test_width(self):
        self.assertEqual(self.image.width, 2)

    def test_shape(self):
        self.assertEqual(self.image.shape, (1, 2, 4))

    def test_get_item(self):
        self.assertTrue((self.image.array[:, :, -1] == np.array([[3, 7]])).all())

    def test_superimpose(self):
        image = Image(np.array([[[10, 11, 12, 13], [14, 15, 16, 17]]], dtype="uint8"))
        image.superimpose(self.image)
        self.assertTrue((image.array == np.array([[[0, 1, 2, 15], [4, 5, 6, 23]]], dtype="uint8")).all())
        self.assertRaises(AssertionError, image.superimpose, self.image2)
        # The superimposed images must have the same shape

    def test_resize(self):
        image = self.image.copy()
        image.resize()
        self.assertTrue((image.array == self.image.array).all())
        image.resize(height=2, width=4)
        self.assertTrue(image.size == (4, 2))
        image.resize(height=4)
        self.assertTrue(image.size == (8, 4))

    def test_rotate(self):
        image = self.image.copy()
        image.rotate(90)
        self.assertTrue((image.array == np.array([[[4, 5, 6, 7]], [[0, 1, 2, 3]]], dtype="uint8")).all())

    def test_extend(self):
        image = self.image.copy()
        image.extend()
        self.assertTrue((image.array == self.image.array).all())
        image.extend(left=1, down=2)
        self.assertTrue((image.array == np.array([[[0, 0, 0, 0], [0, 1, 2, 3], [4, 5, 6, 7]],
                                                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype="uint8")).all())

    def test_stack(self):
        image = self.image.copy()
        image.stack(image, vertically=True)  # True is the default value
        self.assertTrue((image.array == np.array([[[0, 1, 2, 3], [4, 5, 6, 7]],
                                                  [[0, 1, 2, 3], [4, 5, 6, 7]]], dtype="uint8")).all())
        image = self.image.copy()
        image.stack(image, vertically=False)
        self.assertTrue((image.array == np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]]],
                                                 dtype="uint8")).all())

    def test_copy(self):
        image = self.image.copy()
        self.assertTrue((image.array == self.image.array).all())

    def test_replace_color(self):
        image = self.image.copy()
        image.replace_color(Color(0, 1, 2, 3), Color(4, 5, 6, 7))
        self.assertTrue((image.array == np.array([[[4, 5, 6, 7], [4, 5, 6, 7]]], dtype="uint8")).all())
        image = self.image.copy()
        image.replace_color(Color(0, 1, 2, 3), (4, 5, 6))
        self.assertTrue((image.array == np.array([[[4, 5, 6, 3], [4, 5, 6, 7]]], dtype="uint8")).all())
        image = self.image.copy()
        image.replace_color(Color(0, 1, 2, 4), (4, 5, 6))
        self.assertTrue((image.array == np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]], dtype="uint8")).all())
        image = self.image.copy()
        image.replace_color((0, 1, 2), Color(4, 5, 6, 7))
        self.assertTrue((image.array == np.array([[[4, 5, 6, 7], [4, 5, 6, 7]]], dtype="uint8")).all())
        image = self.image.copy()
        image.replace_color((0, 1, 2), (4, 5, 6))
        self.assertTrue((image.array == np.array([[[4, 5, 6, 3], [4, 5, 6, 7]]], dtype="uint8")).all())

    def test_increase_brightness(self):
        image = self.image.copy()
        image.increase_brightness(3)
        self.assertTrue((image.array == np.array([[[3, 4, 5, 3], [7, 8, 9, 7]]], dtype="uint8")).all())
        image.increase_brightness(-5)
        self.assertTrue((image.array == np.array([[[0, 0, 0, 3], [2, 3, 4, 7]]], dtype="uint8")).all())
        image.increase_brightness(253)
        self.assertTrue((image.array == np.array([[[253, 253, 253, 3], [255, 255, 255, 7]]], dtype="uint8")).all())

    def test_open(self):
        image = Image.open("test_image.png")
        self.assertTrue((image.array == np.array([[[0, 1, 2, 255], [4, 5, 6, 255]]], dtype="uint8")).all())

    def test_create(self):
        image = Image.create()
        self.assertTrue((image.array == np.array([[[0, 0, 0, 0]]], dtype="uint8")).all())
        image = Image.create(height=2, rgba=Color(1, 2, 3, 4))
        self.assertTrue((image.array == np.array([[[1, 2, 3, 4]], [[1, 2, 3, 4]]], dtype="uint8")).all())
