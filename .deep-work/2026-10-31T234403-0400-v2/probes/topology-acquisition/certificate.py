"""Exact-rational sufficient certificate for one acquired-corridor learning step.

Fixed family Q=(z-3)(z**3-u)-epsilon, epsilon in [.0995,.1005].
No floating-point inequality is used in the proof obligations. Numerical root
centers/path vertices are interpreted as their exact binary rational values.
This certifies the mathematical family and fixed control polylines, not error
bounds for the floating-point path tracker or a noisy path tube.
"""
from fractions import Fraction as F
from math import comb


ZERO = (F(0), F(0))
RHO = F(1, 1000)
DELTA = F(1, 2000)
H = (F(1, 10), F(0), F(27), F(-18), F(3))
U = (F(0), F(0), F(-9), F(4))


def pair(z):
    return F(float(z.real)), F(float(z.imag))


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def mul(a, b):
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def scale(a, b):
    return a[0]*b, a[1]*b


def power(a, k):
    out = (F(1), F(0))
    for _ in range(k):
        out = mul(out, a)
    return out


def norm1(a):
    return abs(a[0]) + abs(a[1])


def norminf(a):
    return max(abs(a[0]), abs(a[1]))


def norm2sq(a):
    return a[0]*a[0] + a[1]*a[1]


def taylor(coeff, center):
    ans = []
    for k in range(len(coeff)):
        value = ZERO
        for j in range(k, len(coeff)):
            value = add(value, scale(power(center, j-k), coeff[j]*comb(j, k)))
        ans.append(value)
    return ans


def segment_distance_squared(point, a, b):
    direction = sub(b, a)
    length2 = norm2sq(direction)
    if not length2:
        return norm2sq(sub(point, a))
    rel = sub(point, a)
    projection = (rel[0]*direction[0]+rel[1]*direction[1])/length2
    t = max(F(0), min(F(1), projection))
    closest = add(a, scale(direction, t))
    return norm2sq(sub(point, closest))


def packed(a):
    return [str(a[0]), str(a[1])]


def certify(critical_centers, protected_paths):
    """Prove no finite critical value crosses complete fixed A/C polylines.

    Four disjoint disks each contain exactly one root of the quartic H_epsilon
    for every permitted epsilon. Taylor image bounds for u=4z**3-9z**2 then
    enclose every finite critical value. Exact distance comparisons cover every
    segment, including handles and basepoint. Passing is a sufficient condition.
    """
    centers = [pair(z) for z in critical_centers]
    result = dict(passed=False, exact_rational=True,
                  family="Q=(z-3)(z^3-u)-epsilon",
                  epsilon_center="1/10", epsilon_radius=str(DELTA),
                  root_disk_radius=str(RHO), disks=[], path_clearances=[],
                  scope="fixed exact polylines and mathematical parameter homotopy; not numeric-tracker error or noisy tubes")
    if len(centers) != 4 or set(protected_paths) != {"A", "C"}:
        result["failure"] = "need all four critical centers and both complete protected paths"
        return result
    if any(norminf(sub(a, b)) <= 2*RHO
           for i, a in enumerate(centers) for b in centers[:i]):
        result["failure"] = "critical-root disks not certified disjoint"
        return result
    if any(norminf(sub(a, (F(3), F(0)))) <= RHO for a in centers):
        result["failure"] = "a critical-root disk may contain the pole z=3"
        return result
    images = []
    for i, center in enumerate(centers):
        h = taylor(H, center)
        left = norminf(h[1])*RHO
        right = norm1(h[0])+DELTA+sum(norm1(h[k])*RHO**k for k in range(2, 5))
        u = taylor(U, center)
        image_radius = sum(norm1(u[k])*RHO**k for k in range(1, 4))
        result["disks"].append(dict(index=i, exact_center=packed(center),
            rouche_linear_lower=str(left), rouche_remainder_upper=str(right),
            positive_slack=str(left-right), critical_image_center=packed(u[0]),
            critical_image_radius_upper=str(image_radius)))
        if not left > right:
            result["failure"] = "uniform critical-root Rouche inequality failed"
            return result
        images.append((u[0], image_radius))
    for name, vertices in protected_paths.items():
        points = [pair(z) for z in vertices]
        if len(points) < 2 or points[0] != (F(1), F(0)) or points[-1] != points[0]:
            result["failure"] = "protected paths must be complete closed paths at basepoint1"
            return result
        for i, (center, radius) in enumerate(images):
            minimum = min(segment_distance_squared(center, a, b)
                          for a, b in zip(points[:-1], points[1:]))
            slack = minimum-radius*radius
            result["path_clearances"].append(dict(path=name, critical_index=i,
                segments=len(points)-1, squared_distance_lower=str(minimum),
                image_radius_squared_upper=str(radius*radius),
                positive_squared_slack=str(slack)))
            if not slack > 0:
                result["failure"] = "critical-image disk intersects a protected path"
                return result
    result["passed"] = True
    result["conclusion"] = ("all finite critical values stay off A,C and basepoint for entire coefficient homotopy; "
                            "carried branch identification preserves their permutations and all words")
    return result
