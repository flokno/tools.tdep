import numpy as np


def get_orthonormal_directions(direction: np.ndarray) -> list:
    """Take a vector an span the orthogonal space

    Args:
        direction: The reference direction, e.g., [1, 0, 0]

    Returns:
        (direction0, direction1, direction2): Orthonormal directions w.r.t. direction0

    Example:
        direction0 = [1, 0, 0]
        -> ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    """
    direction0 = np.asarray(direction).astype(float)

    direction0 /= np.linalg.norm(direction0)

    # span the orthogonal space
    ## go through x, y, z to find a new direction

    for xx in range(3):
        tmp = np.zeros(3)
        tmp[xx] += 1.0
        direction1 = tmp
        direction1 /= np.linalg.norm(direction1)
        direction2 = np.cross(direction0, direction1)

        if np.linalg.norm(direction2) > 1e-5:

            direction1 = direction1 - direction1 @ direction0 * direction0
            direction1 /= np.linalg.norm(direction1)
            direction2 = np.cross(direction0, direction1)
            direction2 /= np.linalg.norm(direction2)
            break

    return (direction0, direction1, direction2)


def get_rotation_matrix(
    phi: float, axis: np.ndarray = None, radians: bool = False, decimals: int = 12
):
    """Get the rotation matrix for a given rotation

    Args:
      phi: The rotation angle.
      axis: The rotation axis (3-tuple). Defaults to x-axis [1, 0, 0]
      radians: If True phi is in radians, otherwise in degree.
      decimals: Round the final matrix to this many decimal digits.

    Returns:
        np.ndarray: The rotation matrix.

    """
    if not radians:
        phi = phi / 180 * np.pi

    if axis is None:
        axis = [1, 0, 0]

    # Make sure axis is ndarray and norm the rotation axis:
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)

    cp = np.cos(phi)
    sp = np.sin(phi)
    r1, r2, r3 = axis
    Rm = np.array(
        [
            [
                r1**2 * (1 - cp) + cp,
                r1 * r2 * (1 - cp) - r3 * sp,
                r1 * r3 * (1 - cp) + r2 * sp,
            ],
            [
                r1 * r2 * (1 - cp) + r3 * sp,
                r2**2 * (1 - cp) + cp,
                r2 * r3 * (1 - cp) - r1 * sp,
            ],
            [
                r3 * r1 * (1 - cp) - r2 * sp,
                r2 * r3 * (1 - cp) + r1 * sp,
                r3**2 * (1 - cp) + cp,
            ],
        ]
    )

    # clean small values
    return Rm.round(decimals=decimals)


def inscribed_sphere_in_box(cell: np.ndarray) -> float:
    """Find the radius of an inscribed sphere in a unit cell"""

    # the normals of the faces of the box
    na = np.cross(cell[1, :], cell[2, :])
    nb = np.cross(cell[2, :], cell[0, :])
    nc = np.cross(cell[0, :], cell[1, :])
    na /= np.linalg.norm(na)
    nb /= np.linalg.norm(nb)
    nc /= np.linalg.norm(nc)
    # distances between opposing planes
    rr = 1.0e10
    rr = min(rr, abs(na @ cell[0, :]))
    rr = min(rr, abs(nb @ cell[1, :]))
    rr = min(rr, abs(nc @ cell[2, :]))
    rr *= 0.5
    return rr


def bounding_sphere_of_box(cell: np.ndarray) -> float:
    """Find the radius of the sphere bounding a box"""
    a, b, c = np.asarray(cell)

    d1 = a + b + c
    d2 = a + b - c
    d3 = a - b + c
    d4 = a - b - b

    return max(np.linalg.norm([d1, d2, d3, d4], axis=0)) / 2


def get_cubicness(cell: np.ndarray) -> float:
    """Quantify the cubicness of a cell"""
    # perfect radius: 1/2 * width of the cube
    radius_perfect = np.linalg.det(cell) ** (1 / 3) * 0.5
    radius_actual = inscribed_sphere_in_box(cell)

    return radius_actual / radius_perfect


def compute_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute the area between three points in 3D space.

    Args:
        p1: The coordinates of the first point.
        p2: The coordinates of the second point.
        p3: The coordinates of the third point.

    Returns:
        float: The area between the three points.

    """
    # Compute the vectors between the points
    v1 = p2 - p1
    v2 = p3 - p1

    # Compute the cross product of the vectors
    cross_product = np.cross(v1, v2)

    # Compute the magnitude of the cross product
    area = np.linalg.norm(cross_product) / 2

    return area


def compute_volume(points: list) -> float:
    """Compute the volume of a tetrahedron defined by four points in 3D space.

    Args:
        points: A list of four points in 3D space.

    Returns:
        float: The volume of the tetrahedron.

    """
    # Compute the vectors between the points
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    v3 = points[3] - points[0]

    # Compute the volume of the tetrahedron
    volume = np.abs(np.dot(v1, np.cross(v2, v3))) / 6

    return volume


def compute_mean_distance_to_centroid(points: list) -> float:
    """Compute the mean distance to the centroid for a list of points.

    Args:
        points: A list of points in 3D space.

    Returns:
        float: The mean distance to the centroid.

    """
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)
    # Compute the distances from each point to the centroid
    distances = [np.linalg.norm(point - centroid) for point in points]
    # Compute the mean distance
    mean_distance = np.mean(distances)
    return mean_distance


def compute_min_max_distance(points: list) -> (float, float):
    """Compute the minimum and maximum distance between a list of points.

    Args:
        points: A list of points in 3D space.

    Returns:
        (float, float): The minimum and maximum distance between the points.

    """
    # Compute the pairwise distances between the points
    distances = []
    for i, point1 in enumerate(points):
        for j, point2 in enumerate(points):
            if i != j:
                distance = np.linalg.norm(point1 - point2)
                distances.append(distance)
    # Compute the minimum and maximum distances
    min_distance = min(distances)
    max_distance = max(distances)
    return min_distance, max_distance
