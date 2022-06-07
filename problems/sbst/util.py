#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math

import numpy as np
from shapely.geometry import LineString, Polygon

from code_pipeline.tests_generation import RoadTestFactory
from code_pipeline.validation import TestValidator

def sbst_test_to_image(test, map_size):
    """
    Visualizes the road described as points in the plane in the map of specified
    size.
    """

    little_triangle = Polygon([(10, 0), (0, -5), (0, 5), (10, 0)])
    square = Polygon([(5, 5), (5, -5), (-5, -5), (-5, 5), (5, 5)])

    V = TestValidator(map_size=map_size)
    try:
        the_test = RoadTestFactory.create_road_test(test)
        valid, msg = V.validate_test(the_test)
    except:
        return

    # This code is mainly from https://github.com/se2p/tool-competition-av/code_pipeline/visualization.py
    plt.figure()

    # plt.gcf().set_title("Last Generated Test")
    plt.gca().set_aspect("equal", "box")
    plt.gca().set(xlim=(-30, map_size + 30), ylim=(-30, map_size + 30))

    # Add information about the test validity
    title_string = "Test is " + ("valid" if valid else "invalid")
    if not valid:
        title_string = title_string + ":" + msg

    plt.suptitle(title_string, fontsize=14)
    plt.draw()

    # Plot the map. Trying to re-use an artist in more than one Axes which is supported
    map_patch = patches.Rectangle(
        (0, 0),
        map_size,
        map_size,
        linewidth=1,
        edgecolor="black",
        facecolor="none",
    )
    plt.gca().add_patch(map_patch)

    # Road Geometry.
    road_poly = LineString(
        [(t[0], t[1]) for t in the_test.interpolated_points]
    ).buffer(8.0, cap_style=2, join_style=2)
    road_patch = PolygonPatch(
        road_poly, fc="gray", ec="dimgray"
    )  # ec='#555555', alpha=0.5, zorder=4)
    plt.gca().add_patch(road_patch)

    # Interpolated Points
    sx = [t[0] for t in the_test.interpolated_points]
    sy = [t[1] for t in the_test.interpolated_points]
    plt.plot(sx, sy, "yellow")

    # Road Points
    x = [t[0] for t in the_test.road_points]
    y = [t[1] for t in the_test.road_points]
    plt.plot(x, y, "wo")

    # Plot the little triangle indicating the starting position of the ego-vehicle
    delta_x = sx[1] - sx[0]
    delta_y = sy[1] - sy[0]

    current_angle = atan2(delta_y, delta_x)

    rotation_angle = degrees(current_angle)
    transformed_fov = rotate(
        little_triangle, origin=(0, 0), angle=rotation_angle
    )
    transformed_fov = translate(transformed_fov, xoff=sx[0], yoff=sy[0])
    plt.plot(*transformed_fov.exterior.xy, color="black")

    # Plot the little square indicating the ending position of the ego-vehicle
    delta_x = sx[-1] - sx[-2]
    delta_y = sy[-1] - sy[-2]

    current_angle = atan2(delta_y, delta_x)

    rotation_angle = degrees(current_angle)
    transformed_fov = rotate(square, origin=(0, 0), angle=rotation_angle)
    transformed_fov = translate(transformed_fov, xoff=sx[-1], yoff=sy[-1])
    plt.plot(*transformed_fov.exterior.xy, color="black")

    plt.suptitle(title_string, fontsize=14)
    plt.draw()

    return plt.gcf()

def sbst_validate_test(test, map_size):
    """
    Tests if the road described as points in the plane in the map of specified
    size is valid.
    """

    V = TestValidator(map_size=map_size)
    # Sometimes strange errors occur, and we work around them by declaring the
    # test as invalid.
    try:
        the_test = RoadTestFactory.create_road_test(test)
        valid, msg = V.validate_test(the_test)
    except ValueError as e:
        if e.args[0] == "GEOSGeom_createLinearRing_r returned a NULL pointer":
            return 0
        raise

    # print(msg)
    return 1 if valid else 0

def frechet_distance(P, Q):
    """
    Computes the discrete Fréchet distance between the polygonal curves defined
    by the point sequences P and Q.
    """

    # The implementation is based on
    # T. Eiter, H. Mannila. Computing discrete Fréchet distance.
    # Technical report CD-TR 94/64. Technical University of Vienna (1994).
    # http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf

    def C(ca, i, j, P, Q):
        # We use the Euclidean distance.
        if ca[i, j] > -1:
            return ca[i, j]

        if i == 0 and j == 0:
            m = 0
        elif i > 0 and j == 0:
            m = C(ca, i - 1, 0, P, Q)
        elif i == 0 and j > 0:
            m = C(ca, 0, j - 1, P, Q)
        else:
            m = min(
                C(ca, i - 1, j, P, Q),
                C(ca, i - 1, j - 1, P, Q),
                C(ca, i, j - 1, P, Q),
            )

        ca[i, j] = max(np.linalg.norm(P[i] - Q[j]), m)

        return ca[i, j]

    if len(P) == 0 or len(Q) == 0:
        raise ValueError("The input sequences must be nonempty.")

    ca = -1 * np.ones(shape=(len(P), len(Q)))
    return C(ca, len(P) - 1, len(Q) - 1, np.array(P), np.array(Q))

def move_road(P, x0, y0):
    """
    Moves the sequence of points P in such a way that the initial point is
    at (x0, y0) and the initial direction is up.
    """

    if len(P) == 1:
        return [(x0, y0)]

    # Find the translation angle.
    angle = math.pi / 2 - math.atan2(P[1][1] - P[0][1], P[1][0] - P[0][0])
    # Translation vector to origin.
    o_x = -P[0][0]
    o_y = -P[0][1]

    Q = []
    for n in range(len(P)):
        # Map to origin for rotation.
        x = P[n][0] + o_x
        y = P[n][1] + o_y
        # Rotate the point and translate the resulting point back.
        Q.append([0, 0])
        Q[n][0] = math.cos(angle) * x - math.sin(angle) * y + x0
        Q[n][1] = math.sin(angle) * x + math.cos(angle) * y + y0

    return Q

