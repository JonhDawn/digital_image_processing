from region import Circle, Line, Point, Rectangle, RoundedRectangle, Triangle
from image import Image, Color

BLACK = Color(0, 0, 0)
TRANSPARENT = Color(0, 0, 0, 0)
WHITE = Color(255, 255, 255)


def donut(radius: int, increasing: int) -> Image:
    circle = Circle(radius)
    circle_image = Image.from_region(circle, inner_color=BLACK, background_color=TRANSPARENT)
    circle_image.extend(left=increasing, up=increasing, right=increasing, down=increasing)
    ext_circle = Circle(radius + increasing)
    ext_circle_image = Image.from_region(ext_circle, inner_color=WHITE, background_color=TRANSPARENT)
    circle_image.superimpose(ext_circle_image)
    circle_image.replace_color(BLACK, TRANSPARENT)  # Replace the black pixels by transparent ones
    return circle_image


if __name__ == '__main__':

    # Navig button
    rectangle = Rectangle(Point(22, 2))
    rectangle_image = Image.from_region(rectangle, inner_color=WHITE, background_color=TRANSPARENT)
    rectangle_image.extend(left=12, up=2, right=12, down=2)
    navig_image = rectangle_image.copy()
    for _ in range(2):
        navig_image.stack(rectangle_image)
    navig_image.extend(up=15, down=16)

    # Search button
    circle_image = donut(radius=10, increasing=3)
    rectangle = Rectangle(Point(3, 20))
    search_image = Image.from_region(rectangle, inner_color=WHITE, background_color=TRANSPARENT)
    search_image.extend(left=11, right=12)
    search_image.stack(circle_image)
    search_image.rotate(135)

    # Group button
    note_image = donut(radius=4, increasing=2)
    note_image.extend(up=13)
    bar = Rectangle(Point(6, 23), Point(4, 0))
    bar_image = Image.from_region(bar, inner_color=WHITE, background_color=TRANSPARENT)
    bar_image.extend(left=10, down=2)
    note_image.superimpose(bar_image)
    note_image_2 = note_image.copy()
    note_width = note_image.width
    note_image.extend(right=note_width + 4, up=2)
    note_image_2.extend(left=note_width + 4, down=2)
    note_image.superimpose(note_image_2)
    upper_bar = Rectangle(Point(15, 2))
    group_image = Image.from_region(upper_bar, inner_color=WHITE, background_color=TRANSPARENT)
    group_image.rotate(5)
    group_image.extend(left=note_image.width-group_image.width, down=note_image.height-group_image.height)
    group_image.superimpose(note_image)
    group_image.image.show()

    # Tests
    images = [navig_image,
              search_image,
              group_image,
              ]
    for img in images:
        img.image.show()




