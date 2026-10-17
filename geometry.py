"""Clip GLB world-space triangles to a spatial cell without shifting them."""

import numpy as np


def clip_triangle(vertices, lower, upper):
    polygon = list(vertices)
    for axis in range(3):
        for limit, keep_above in ((lower[axis], True), (upper[axis], False)):
            if not polygon:
                return []
            output = []
            previous = polygon[-1]
            previous_inside = previous[axis] >= limit if keep_above else previous[axis] <= limit
            for current in polygon:
                current_inside = current[axis] >= limit if keep_above else current[axis] <= limit
                if current_inside != previous_inside:
                    fraction = (limit-previous[axis])/(current[axis]-previous[axis])
                    output.append(previous + fraction*(current-previous))
                if current_inside:
                    output.append(current)
                previous, previous_inside = current, current_inside
            polygon = output
    return [np.asarray([polygon[0], polygon[i], polygon[i+1]]) for i in range(1, len(polygon)-1)]
