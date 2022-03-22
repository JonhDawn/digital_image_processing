from unittest import TestCase

import PIL.Image
import numpy as np

from image import Image


class TestImage(TestCase):
    def setUp(self) -> None:
        self.image = Image(np.array([[[256, 257, 258, 259], [260, 261, 262, 263]]], dtype="uint8"))
        self.image2 = Image()
        self.image3 = Image(np.array([[[1, 1, 1]]], dtype="uint8"))

    def test_init(self):
        for image in (self.image, self.image2):
            self.assertIsInstance(image, Image)
            self.assertEqual(image.array.dtype, np.uint8)
        self.assertEqual(self.image.array, np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]], dtype="uint8"))
        self.assertEqual(self.image2.array, np.array([[[0, 0, 0, 0]]], dtype="uint8"))
        self.assertEqual(self.image3.array, np.array([[[1, 1, 1, 255]]], dtype="uint8"))
        self.failUnlessRaises(ValueError, Image.__init__, np.array([[[0, 0, 0, "a"]]]),
                              msg="Array elements have to be integers")
        self.failUnlessRaises(AssertionError, Image.__init__, np.array([[0, 0, 0]]),
                              msg="Array's shape has to be (m, n, 3) or (m, n, 4)")
        self.failUnlessRaises(ValueError, Image.__init__, np.array([[[0, 0]]]),
                              msg="Array's shape has to be (m, n, 3) or (m, n, 4)")

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
        self.assertEqual(self.image[:, :, -1].array, np.array([[255, 255]]))

    def test_superimpose(self):
        image = Image(np.array([[[10, 11, 12, 13], [14, 15, 16, 17]]], dtype="uint8"))
        image.superimpose(self.image)
        self.assertEqual(image, np.array([[[0,  1,  2, 15], [4,  5,  6, 23]]], dtype="uint8"))
        self.assertRaises(AssertionError, image.superimpose, self.image2,
                          msg="The superimposed images must have the same shape")

    def test_resize(self):
        image = self.image.copy()
        image.resize()
        self.assertEqual(image.array, self.image.array)
        image.resize(height=2, width=4)
        self.assertEqual(image.size, (4, 2))
        image.resize(height=4)
        self.assertEqual(image.size, (8, 4))

    def test_rotate(self):
        pass

    def test_extend(self):
        pass

    def test_stack(self):
        pass

    def test_copy(self):
        pass
