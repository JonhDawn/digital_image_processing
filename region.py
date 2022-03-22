from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, NamedTuple, Any


class Interval(NamedTuple):
    minimum: float
    maximum: float
    bounds_included = (True, True)

    def __contains__(self, x) -> bool:
        if isinstance(x, float):
            return self.contain_element(x)
        elif isinstance(x, Interval):
            return self.contain_interval(x)

    def contain_element(self, x: float) -> bool:
        """ Return True if the interval contains x and False otherwise. """
        res = self.minimum <= x <= self.maximum
        if not self.bounds_included[0]:
            res &= self.minimum != x
        if not self.bounds_included[1]:
            res &= self.maximum != x
        return res

    def contain_interval(self, x: Interval) -> bool:
        """ Return True if x is a subinterval and False otherwise. """
        res = (self.minimum <= x.minimum) and (self.maximum >= x.maximum)
        if not self.bounds_included[1] and x.bounds_included[1]:
            res &= self.maximum != x.maximum
        if not self.bounds_included[0] and x.bounds_included[0]:
            res &= self.minimum != x.minimum
        return res


class Point(NamedTuple):
    x: float
    y: float


class Border(NamedTuple):
    interval: Interval
    upper: Callable[[Any], Any]
    lower: Callable[[Any], Any]


@dataclass
class Region:
    borders: list[Border]
    left_bound: float = None
    right_bound: float = None
    upper_bound: float = None
    lower_bound: float = None

    @property
    def center(self) -> Point:
        return Point((self.left_bound + self.right_bound) / 2,
                     (self.upper_bound + self.lower_bound) / 2)

    @property
    def height(self) -> float:
        return self.lower_bound - self.upper_bound

    @property
    def width(self) -> float:
        return self.right_bound - self.left_bound

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.left_bound, self.upper_bound, self.right_bound, self.lower_bound

    def __contains__(self, p: Point):
        x, y = p
        for border in self.borders:
            if x in border.interval:
                if border.lower <= y <= border.upper(x):
                    return True
        return False

    def __add__(self, other: Region):
        return Region(self.borders + other.borders,
                      min(self.left_bound, other.left_bound),
                      min(self.upper_bound, other.upper_bound),
                      max(self.right_bound, other.right_bound),
                      max(self.lower_bound, other.lower_bound))

    def restrict(self, interval: Interval):
        """ Restrict the region to the given subinterval. """
        borders = []
        for border in self.borders:
            borders.append(Border(interval, border.upper, border.lower))
        self.borders = borders


class Rectangle(Region):
    def __init__(self, down_right: Point, up_left: Point = (0, 0)):
        self.left_bound, self.upper_bound = up_left
        self.right_bound, self.lower_bound = down_right
        interval = Interval(self.left_bound, self.right_bound)

        def upper_border(x):
            """ Equation of the function that define the upper border. """
            return x*0 + self.upper_bound

        def lower_border(x):
            """ Equation of the function that define the lower border. """
            return x*0 + self.lower_bound

        self.borders = [Border(interval, upper_border, lower_border)]


class Circle(Region):
    def __init__(self, radius: float, center: Point = None):
        if center is None:
            center = Point(radius, radius)
        self.left_bound, self.right_bound = center.x - radius, center.x + radius
        self.upper_bound, self.lower_bound = center.y - radius, center.y + radius
        self.radius = radius
        interval = Interval(self.center.x - radius, self.center.x + radius)

        def upper_border(x):
            return (radius ** 2 - (x - self.center.x) ** 2) ** 0.5 + self.center.y

        def lower_border(x):
            return -(radius ** 2 - (x - self.center.x) ** 2) ** 0.5 + self.center.y

        self.borders = [Border(interval, lower_border, upper_border)]


class RoundedRectangle(Region):
    def __init__(self, radius: float, down_right: Point, up_left: Point = (0, 0)):
        self.left_bound, self.upper_bound = up_left
        self.right_bound, self.lower_bound = down_right
        self.radius = radius
        assert radius < (self.right_bound - self.left_bound) / 2 and radius < (self.lower_bound - self.upper_bound) / 2
        center = Rectangle(Point(self.right_bound - self.radius, self.lower_bound),
                           Point(self.left_bound + self.radius, self.upper_bound))
        border_1 = self.rounded_border(x_corner=self.left_bound, x_center=self.left_bound+self.radius)
        border_2 = center.borders[0]
        border_3 = self.rounded_border(x_corner=self.right_bound, x_center=self.right_bound-self.radius)
        self.borders = [border_1, border_2, border_3]

    def rounded_border(self, x_corner: float, x_center: float) -> Border:
        interval = Interval(min([x_corner, x_center]), max([x_corner, x_center]))
        upper_circle = Circle(self.radius, Point(x_center, self.upper_bound + self.radius))
        lower_circle = Circle(self.radius, Point(x_center, self.lower_bound - self.radius))
        lower_circle.restrict(interval)
        return Border(interval=interval,
                      upper=upper_circle.borders[0].lower,
                      lower=lower_circle.borders[0].upper)


@dataclass
class Polygon(Region):
    def __init__(self, vertices: list[Point]):
        self.vertices = vertices
        self.upper_bound, self.lower_bound, self.left_bound, self.right_bound = self.get_bounds()
        self.sort_vertices()
        triangles = self.decomposition()
        self.borders = []
        for border in [convex.borders for convex in triangles]:
            self.borders.extend(border)

    def sort_vertices(self) -> None:
        """ The first vertex is the leftmost one. """
        for i, vertex in enumerate(self.vertices):
            if vertex.x == self.left_bound:
                self.vertices = self.vertices[i:] + self.vertices[:i]
                return

    def decomposition(self) -> list[Triangle]:
        # TODO: https://fr.wikipedia.org/wiki/Triangulation_d%27un_polygone#:~:text=En%20g%C3%A9om%C3%A9trie
        #  %20algorithmique%2C%20la%20triangulation,dont%20l'union%20est%20P.
        """ Decompose the polygon into triangles. """
        pass

    def get_bounds(self) -> tuple[float, float, float, float]:
        """ Return the bounds of the polygon. """
        upper_bound = lower_bound = self.vertices[0].x
        left_bound = right_bound = self.vertices[0].y
        for vertex in self.vertices[1:]:
            if vertex.x < left_bound:
                left_bound = vertex.x
            elif vertex.x > right_bound:
                right_bound = vertex.x
            if vertex.y < upper_bound:
                upper_bound = vertex.y
            elif vertex.y > lower_bound:
                lower_bound = vertex.y
        return upper_bound, lower_bound, left_bound, right_bound


def line(p: Point, q: Point) -> Callable:
    """ Return the equation of the non-vertical line passing through p and q. """
    def equation(x):
        return (p.y-q.y)/(p.x-q.x) * x + p.y - (p.y-q.y)/(p.x-q.x) * p.x
    return equation


class Triangle(Region):
    def __init__(self, vertices: list[Point, Point, Point]):
        self.vertices = sorted(vertices, key=lambda vertex: vertex.x)
        self.upper_bound = min(map(lambda vertex: vertex.y, vertices))
        self.lower_bound = max(map(lambda vertex: vertex.y, vertices))
        self.left_bound, self.right_bound = self.vertices[0].x, self.vertices[-1].x
        if len(set(map(lambda vertex: vertex.x, self.vertices))) == 2:  # 2 vertices have the same abscissa
            self.borders = self.get_borders_with_vertical_edge()
        else:
            self.borders = self.get_borders_without_vertical_edge()

    def get_borders_with_vertical_edge(self) -> list[Border]:
        """ Return the borders when there is a vertical edge. """
        a, b, c = self.vertices  # goal:  b.x == c.x  and  b.y > c.y
        if a.x == b.x:
            a, c = c, a
        elif a.x == c.x:
            a, b = b, a
        if b.y < c.y:
            b, c = c, b
        return [Border(Interval(a.x, b.x), upper=line(a, c), lower=line(a, b))]

    def get_borders_without_vertical_edge(self) -> list[Border]:
        """ Return the borders when there is no vertical edge. """
        a, b, c = self.vertices
        interval_1 = Interval(a.x, b.x)
        interval_2 = Interval(b.x, c.x)
        if b.y > line(a, c)(b.x):
            return [Border(interval_1, upper=line(a, c), lower=line(a, b)),
                    Border(interval_2, upper=line(a, c), lower=line(b, c))]
        else:
            return [Border(interval_1, upper=line(a, b), lower=line(a, c)),
                    Border(interval_2, upper=line(b, c), lower=line(a, c))]


class Line(Polygon):
    def __init__(self, p: Point, q: Point, thickness: float):
        t = thickness/2
        if p.x == q.x:  # vertical line
            a, b, c, d = Point(p.x - t, p.y), Point(p.x + t, p.y), Point(q.x + t, q.y), Point(q.x - t, q.y)
        elif p.y == q.y:  # horizontal line
            a, b, c, d = Point(p.x, p.y - t), Point(p.x, p.y + t), Point(q.x, q.y + t), Point(q.x, q.y - t)
        else:
            pq = ((p.x-q.x)**2 + (p.y-q.y)**2) ** 0.5  # distance from p to q
            sign = (-1) ** int((p.x < q.x) ^ (p.y > q.y))  # -1 if the form of the line pq is / and 1 if it is \
            a = Point(p.x + sign * thickness / 2 * abs(p.x - q.x) / pq,
                      p.y - sign * thickness / 2 * abs(p.y - q.y) / pq)
            b = Point(q.x + sign * thickness / 2 * abs(p.x - q.x) / pq,
                      q.y - sign * thickness / 2 * abs(p.y - q.y) / pq)
            c = Point(q.x - sign * thickness / 2 * abs(p.x - q.x) / pq,
                      q.y + sign * thickness / 2 * abs(p.y - q.y) / pq)
            d = Point(p.x - sign * thickness / 2 * abs(p.x - q.x) / pq,
                      p.y + sign * thickness / 2 * abs(p.y - q.y) / pq)
        super().__init__([a, b, c, d])  # The points a, b, c and d form a rectangle which is the expected line

    def decomposition(self) -> list[Triangle]:
        """ Decompose the polygon into triangles. """
        return [Triangle(self.vertices[0:3]), Triangle(self.vertices[1:4])]
