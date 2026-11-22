#!/usr/bin/env python3
"""Magnetic-tile starship blockout.

Rebuilds the photographed tile model as a list of flat tiles placed in 3D,
then exports it as GLB + OBJ (one named object per tile) in two variants:

  frames  - open tile frames, like the physical toy
  panels  - solid thin plates, a closed-ish hull for kitbashing / displacement

Units: 1.0 = one tile edge.  Build space is Z-up (X = nose -> tail, +Y = the
trapezoid wing side).  OBJ keeps Z-up; GLB is converted to glTF's Y-up.

Run:  python3 build_tiles.py
"""
import json
import math
import os
import struct

import numpy as np

H3 = math.sqrt(3) / 2            # height of the equilateral tile
TALL = math.sqrt(2.0**2 - 0.25)  # slant height of the tall isosceles tile (base 1, sides 2)
S = 0.5 + H3                     # y of the outrigger prism faces
BAR = 0.13                       # frame bar width
THICK = 0.05                     # tile thickness
CHAMFER = 0.05                   # rim bevel: each tile tapers to its edge so neighbours can hinge there
CLEAR = 0.006                    # hairline clearance pulled in from the ideal edge line

COLORS = {
    "red": "#e0262b", "orange": "#f07a1a", "yellow": "#f2c81e", "green": "#4fb53a",
    "blue": "#2456d6", "purple": "#8a3fc0", "pink": "#ec5fa8", "magenta": "#c2289a",
    "cyan": "#2aa7de",
}

Z = np.array([0.0, 0.0, 1.0])
tiles = []


def P(x, y, z=0.0):
    return np.array([x, y, z], float)


def add(name, color, pts, group, kind="regular"):
    tiles.append({"name": name, "color": color, "group": group, "kind": kind,
                  "pts": [np.asarray(p, float) for p in pts]})


def hinge(name, color, group, a, b, kind="sq", inward=(0, 0, 0), lean=0.0, down=False):
    """Tile attached along edge a-b, rising (or hanging) and leaning `lean` rad toward `inward`."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    inward = np.asarray(inward, float)
    u = math.cos(lean) * (-Z if down else Z) + math.sin(lean) * inward
    mid = (a + b) / 2
    if kind == "sq":
        pts = [a, b, b + u, a + u]
    elif kind == "tri":
        pts = [a, b, mid + H3 * u]
    elif kind == "tall":
        pts = [a, b, mid + TALL * u]
    add(name, color, pts, group, "tall" if kind == "tall" else "regular")


def wall(name, color, group, p, q, z0=0.0):
    hinge(name, color, group, P(p[0], p[1], z0), P(q[0], q[1], z0))


def flat(name, color, group, xy, z):
    add(name, color, [P(x, y, z) for x, y in xy], group)


# ---------------------------------------------------------------- lower deck
# Spine: three cells in a row. Nose cube, centre cell (under the spire), tail cube.
g = "spine"
wall("nose_cube_front", "purple", g, (-1.5, -0.5), (-1.5, 0.5))
wall("nose_cube_side_wing", "pink", g, (-1.5, 0.5), (-0.5, 0.5))
wall("nose_cube_side_dome", "blue", g, (-1.5, -0.5), (-0.5, -0.5))
wall("nose_cube_bulkhead", "orange", g, (-0.5, -0.5), (-0.5, 0.5))
flat("nose_cube_roof", "yellow", g, [(-1.5, -0.5), (-0.5, -0.5), (-0.5, 0.5), (-1.5, 0.5)], 1)
wall("centre_side_wing", "yellow", g, (-0.5, 0.5), (0.5, 0.5))
wall("centre_side_dome", "orange", g, (-0.5, -0.5), (0.5, -0.5))
wall("centre_bulkhead", "red", g, (0.5, -0.5), (0.5, 0.5))
flat("centre_roof", "green", g, [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], 1)
wall("tail_cube_side_wing", "yellow", g, (0.5, 0.5), (1.5, 0.5))
wall("tail_cube_side_dome", "orange", g, (0.5, -0.5), (1.5, -0.5))
wall("tail_cube_back", "yellow", g, (1.5, -0.5), (1.5, 0.5))
flat("tail_cube_roof", "orange", g, [(0.5, -0.5), (1.5, -0.5), (1.5, 0.5), (0.5, 0.5)], 1)

# Wing-side outrigger: triangular prism, its angled infill, and the trapezoid lying on the floor.
g = "outrigger_wing"
wall("wing_prism_face", "red", g, (0, S), (1, S))
wall("wing_prism_wall_nose", "blue", g, (0, S), (0.5, 0.5))
wall("wing_prism_wall_tail", "yellow", g, (1, S), (0.5, 0.5))
wall("wing_infill_wall", "magenta", g, (0, S), (-0.5, 0.5))
flat("wing_prism_roof", "cyan", g, [(0, S), (1, S), (0.5, 0.5)], 1)
flat("wing_infill_roof", "pink", g, [(0, S), (-0.5, 0.5), (0.5, 0.5)], 1)
flat("wing_trapezoid", "pink", g, [(-0.5, 0.5), (-1.5, 0.5), (-2, S), (0, S)], 0)

# Dome-side outrigger: mirrored prism.
g = "outrigger_dome"
wall("dome_prism_face", "orange", g, (0, -S), (1, -S))
wall("dome_prism_wall_nose", "blue", g, (0, -S), (0.5, -0.5))
wall("dome_prism_wall_tail", "yellow", g, (1, -S), (0.5, -0.5))
flat("dome_prism_roof", "red", g, [(0, -S), (1, -S), (0.5, -0.5)], 1)
flat("dome_infill_roof", "purple", g, [(0, -S), (-0.5, -0.5), (0.5, -0.5)], 1)

# ---------------------------------------------------------------- upper deck
# Spire: two tall isosceles tiles leaning together over the centre cell.
g = "spire"
lean_spire = math.asin(0.5 / TALL)
hinge("spire_wing", "pink", g, P(-0.5, 0.5, 1), P(0.5, 0.5, 1), "tall", (0, -1, 0), lean_spire)
hinge("spire_dome", "green", g, P(-0.5, -0.5, 1), P(0.5, -0.5, 1), "tall", (0, 1, 0), lean_spire)


# ---------------------------------------------------------------- side dome
# The ball of triangles beside the nose cube: an icosahedron resting on one face,
# minus the floor face and the faces that would sit inside the hull.
def icosahedron():
    phi = (1 + math.sqrt(5)) / 2
    v = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            v += [(0, s1, s2 * phi), (s1, s2 * phi, 0), (s1 * phi, 0, s2)]
    v = np.array(v, float) / 2  # edge length 1
    faces = []
    n = len(v)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if all(abs(np.linalg.norm(v[a] - v[b]) - 1) < 1e-6 for a, b in ((i, j), (j, k), (i, k))):
                    f = [i, j, k]
                    if np.dot(np.cross(v[j] - v[i], v[k] - v[i]), v[i] + v[j] + v[k]) < 0:
                        f = [i, k, j]
                    faces.append(f)
    return v, faces


def rot_to(a, b):
    """Rotation matrix taking unit vector a to unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def rot_z(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


DOME_CENTRE = (-1.15, -1.28)
DOME_SPIN = math.radians(-8)
iv, ifaces = icosahedron()
n0 = np.cross(iv[ifaces[0][1]] - iv[ifaces[0][0]], iv[ifaces[0][2]] - iv[ifaces[0][0]])
iv = iv @ rot_to(n0, -Z).T @ rot_z(DOME_SPIN).T
iv = iv - np.array([0, 0, iv[:, 2].min()]) + np.array([DOME_CENTRE[0], DOME_CENTRE[1], 0])
dome_cols = ["yellow", "yellow", "purple", "yellow", "blue", "green", "yellow", "red",
             "pink", "yellow", "green", "purple", "yellow", "blue", "red", "yellow"]
k = 0
for f in ifaces:
    c = iv[f].mean(axis=0)
    if c[2] < 0.05:                       # the face it rests on
        continue
    if c[1] > -0.62 or c[0] > -0.42:      # buried in the hull
        continue
    add(f"dome_{k:02d}", dome_cols[k % len(dome_cols)], list(iv[f]), "dome")
    k += 1


# ---------------------------------------------------------------- magnets
# Every tile carries one magnet at the middle of each unit-length side. Longer sides
# carry theirs where a square laid along that side would put its own: the trapezoid's
# 2-long base has two (at 0.5 and 1.5), and the tall tile's 2-long sides have one each,
# half an edge up from the base corner.
MATED, LOOSE = 0.06, 0.32   # distance thresholds, in tile edges


def tile_magnets(t):
    pts = np.array(t["pts"])
    m = len(pts)
    n = np.zeros(3)
    for i in range(m):
        n += np.cross(pts[i], pts[(i + 1) % m])
    n /= np.linalg.norm(n)
    out = []
    for i in range(m):
        a, b = pts[i], pts[(i + 1) % m]
        length = np.linalg.norm(b - a)
        d = (b - a) / length
        if length < 1.5:
            offs = [length / 2]
        elif t["kind"] == "tall":
            offs = [0.5] if i == 1 else [length - 0.5]   # nearest the base corner
        else:
            offs = [0.5, length - 0.5]
        for k, o in enumerate(offs):
            out.append({"tile": t["name"], "group": t["group"], "edge": i, "slot": k,
                        "pos": a + d * o, "dir": d, "out": np.cross(d, n), "normal": n})
    return out


def find_magnets():
    mags = [mg for t in tiles for mg in tile_magnets(t)]
    P_ = np.array([mg["pos"] for mg in mags])
    D = np.linalg.norm(P_[:, None] - P_[None], axis=2)
    same = np.array([[a["tile"] == b["tile"] for b in mags] for a in mags])
    D[same] = np.inf
    parent = list(range(len(mags)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, mg in enumerate(mags):
        j = int(D[i].argmin())
        mg["gap"] = float(D[i, j])
        mg["status"] = "mated" if mg["gap"] <= MATED else "loose" if mg["gap"] <= LOOSE else "open"
        mg["mates"] = sorted({mags[k]["tile"] for k in np.nonzero(D[i] <= (MATED if mg["status"] == "mated" else LOOSE))[0]}) \
            if mg["status"] != "open" else []
        for k in np.nonzero(D[i] <= MATED)[0]:
            parent[root(i)] = root(int(k))
    joints = {}
    for i, mg in enumerate(mags):
        mg["joint"] = root(i)
        joints.setdefault(root(i), []).append(mg)
    return mags, list(joints.values())


# ---------------------------------------------------------------- meshing
def inset(pts, n, w):
    m = len(pts)
    out = []
    for i in range(m):
        e1 = pts[i] - pts[i - 1]
        e2 = pts[(i + 1) % m] - pts[i]
        n1 = np.cross(n, e1 / np.linalg.norm(e1))
        n2 = np.cross(n, e2 / np.linalg.norm(e2))
        out.append(pts[i] + w * (n1 + n2) / (1 + np.dot(n1, n2)))
    return np.array(out)


def mesh_tile(pts, mode):
    """Returns (vertices, polygon faces) for one tile; faces wind outward.

    The rim is bevelled to a ridge on the tile's mid-plane, so tiles that share an
    edge touch along a line (like the rounded rims of the toy) instead of overlapping.
    """
    pts = np.array(pts)
    m = len(pts)
    n = np.zeros(3)
    for i in range(m):
        n += np.cross(pts[i], pts[(i + 1) % m])
    n /= np.linalg.norm(n)
    h = n * THICK / 2
    mid = inset(pts, n, CLEAR)
    face = inset(pts, n, CLEAR + CHAMFER)
    if mode == "panels":
        v = np.vstack([face + h, mid, face - h])
        top, rim, bot = 0, m, 2 * m
        faces = [list(range(m)), list(range(3 * m - 1, 2 * m - 1, -1))]
    else:
        hole = inset(pts, n, BAR)
        v = np.vstack([face + h, mid, face - h, hole + h, hole - h])
        top, rim, bot, it, ib = 0, m, 2 * m, 3 * m, 4 * m
        faces = []
    for i in range(m):
        j = (i + 1) % m
        faces.append([rim + i, rim + j, top + j, top + i])
        faces.append([bot + i, bot + j, rim + j, rim + i])
        if mode != "panels":
            faces.append([top + i, top + j, it + j, it + i])
            faces.append([bot + j, bot + i, ib + i, ib + j])
            faces.append([ib + j, ib + i, it + i, it + j])
    return v, faces


# ---------------------------------------------------------------- collision check
# Every tile's solid (the panel form, which contains the frame form) is convex, so a
# separating-axis test is exact: two tiles are clear if some axis separates them.
def solid(t):
    v, faces = mesh_tile(t["pts"], "panels")
    normals, dirs = [], []
    for f in faces:
        nn = np.cross(v[f[1]] - v[f[0]], v[f[2]] - v[f[0]])
        normals.append(nn / np.linalg.norm(nn))
        for i in range(len(f)):
            d = v[f[(i + 1) % len(f)]] - v[f[i]]
            d /= np.linalg.norm(d)
            if not any(abs(abs(np.dot(d, e)) - 1) < 1e-6 for e in dirs):
                dirs.append(d)
    return v, np.array(normals), np.array(dirs)


def penetration(a, b):
    """Depth by which two convex solids overlap; <= 0 means they are clear."""
    va, na, da = a
    vb, nb, db = b
    if np.linalg.norm(va.mean(0) - vb.mean(0)) > 3.2:
        return -1.0
    ax = np.cross(da[:, None, :], db[None, :, :]).reshape(-1, 3)
    ln = np.linalg.norm(ax, axis=1)
    ax = np.vstack([na, nb, ax[ln > 1e-6] / ln[ln > 1e-6, None]])
    pa, pb = va @ ax.T, vb @ ax.T
    sep = np.maximum(pa.min(0) - pb.max(0), pb.min(0) - pa.max(0))
    return float(-sep.max())


def collisions(subset=None):
    sol = [solid(t) for t in tiles]
    hits = []
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if subset is not None and i not in subset and j not in subset:
                continue
            d = penetration(sol[i], sol[j])
            if d > 1e-4:
                hits.append((d, i, j))
    return hits


# ---------------------------------------------------------------- pods
# The structure either side of the spire. Each pod is a chain of hinges over an end cube:
#   end square - hinged on the cube's outer top edge, leaning in by `end`
#   roof       - hinged on the end square's top edge, pitched up toward the spire by `pitch`
#   up tris    - hinged on the cube's two flank edges, leaning in by `up` (negative = out)
#   down tris  - hinged on each up tri's spire-side edge, folded in by `fold`; the free
#                side is the one that reaches for the magnet on the spire's long side
# The angles are not chosen by hand. `solve_pods` treats every magnet as attracting the
# nearest magnet on another tile, gives the magnets on the spire's long sides extra pull
# (the magnetic bias), pulls each roof's inner edge onto the spire's sides so the roof
# comes to rest against it, forbids any overlap, and searches for the pose with the
# least total gap. Run `build_tiles.py --solve` to redo the search; paste the result here.
POD_KEYS = ("end", "pitch", "up", "fold")
POD_POSE = {'end': 0.328, 'pitch': 0.1763, 'up': -0.4503, 'fold': 0.8308}
POD_LO, POD_HI = np.radians([0, -20, -40, -90]), np.radians([45, 80, 40, 90])
MAGNET_RANGE = 0.12
SPIRE_BIAS = 4.0   # weight of the spire's side magnets on the down tris
FOLLOW_BIAS = 10.0 # weight of the roof matching the upward slope of its flank triangles
REST_BIAS = 6.0    # weight of the roof's inner edge resting on the spire's sides


def _well(gap):
    """Magnets are short-range: a joint only pays off once it is nearly closed (within
    about a tenth of an edge), with a weak long-range slope to guide the search."""
    return 1 - np.exp(-(gap / MAGNET_RANGE) ** 2) + 0.2 * gap ** 2


def _seg_gap(a0, a1, b0, b1, n=25):
    t = np.linspace(0, 1, n)[:, None]
    pa, pb = a0 + (a1 - a0) * t, b0 + (b1 - b0) * t
    return float(np.linalg.norm(pa[:, None] - pb[None], axis=2).min())
POD_COLOURS = {  # end, roof, wing-side up/down, dome-side up/down
    "tail_pod": ("purple", "red", "pink", "yellow", "green", "orange"),
    "nose_pod": ("red", "blue", "pink", "red", "green", "red")}


def _perp(v, e):
    v = np.asarray(v, float) - np.dot(v, e) * e
    return v / np.linalg.norm(v)


def pod_tiles(pod, pose):
    c_end, c_roof, c_up_w, c_dn_w, c_up_d, c_dn_d = POD_COLOURS[pod]
    X, Y = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    u_e = math.cos(pose["end"]) * Z - math.sin(pose["end"]) * X
    u_r = -math.cos(pose["pitch"]) * X + math.sin(pose["pitch"]) * Z
    ta, tb = P(1.5, -0.5, 1) + u_e, P(1.5, 0.5, 1) + u_e
    out = [("end", c_end, [P(1.5, -0.5, 1), P(1.5, 0.5, 1), tb, ta]),
           ("roof", c_roof, [ta, tb, tb + u_r, ta + u_r])]
    for side, sy, c_up, c_dn in (("wing", 1, c_up_w, c_dn_w), ("dome", -1, c_up_d, c_dn_d)):
        inner, outer = P(0.5, 0.5 * sy, 1), P(1.5, 0.5 * sy, 1)
        apex = (inner + outer) / 2 + H3 * (math.cos(pose["up"]) * Z - math.sin(pose["up"]) * sy * Y)
        e = (apex - inner) / np.linalg.norm(apex - inner)
        v1 = _perp(-X, e)                       # in the up tri's plane, toward the spire
        v2 = _perp(-sy * Y, e)                  # folding inward, toward the pod's centreline
        v2 = _perp(v2 - np.dot(v2, v1) * v1, e)
        reach = (inner + apex) / 2 + H3 * (math.cos(pose["fold"]) * v1 + math.sin(pose["fold"]) * v2)
        out.append((f"up_{side}", c_up, [outer, inner, apex]))
        out.append((f"down_{side}", c_dn, [inner, reach, apex]))
    turn = np.array([-1.0, 1.0, 1.0]) if pod == "nose_pod" else np.ones(3)
    return [{"name": f"{pod}_{n}", "color": c, "group": pod, "kind": "regular",
             "pts": [np.asarray(p, float) * turn for p in pts]} for n, c, pts in out]


def pod_cost(pose, base_solids, base_mags):
    """Total squared magnet gap for the tail pod (lower is better); inf if anything overlaps."""
    pod = pod_tiles("tail_pod", pose)
    sol = [solid(t) for t in pod]
    for i, a in enumerate(sol):
        for b in sol[i + 1:] + base_solids:
            if penetration(a, b) > 1e-4:
                return math.inf
    mags = [m for t in pod for m in tile_magnets(t)]
    pp = np.array([m["pos"] for m in mags])
    owner = np.array([m["tile"] for m in mags])
    bp = np.array([m["pos"] for m in base_mags])
    d_pod = np.linalg.norm(pp[:, None] - pp[None], axis=2)
    d_pod[owner[:, None] == owner[None]] = np.inf
    d_base = np.linalg.norm(pp[:, None] - bp[None], axis=2)
    gap = np.minimum(np.minimum(d_pod.min(1), d_base.min(1)), 0.6)
    spire = [i for i, m in enumerate(base_mags) if m["tile"].startswith("spire") and m["edge"] > 0 and m["pos"][0] > 0]
    pull = np.minimum(d_base[:, spire].min(0), 0.6)
    roof = pod[1]["pts"]
    rest = sum(_seg_gap(roof[2], roof[3], t["pts"][0 if t["pts"][0][0] > 0 else 1], t["pts"][2]) ** 2
               for t in tiles if t["group"] == "spire")
    # The roof follows the flanks: its pitch is drawn to the slope of the down tris' top
    # edges, which rise toward the spire once those tris have snapped onto it.
    top = pod[3]["pts"][1] - pod[3]["pts"][2]
    slope = math.asin(top[2] / np.linalg.norm(top))
    return float(_well(gap).sum() + SPIRE_BIAS * _well(pull).sum() + REST_BIAS * rest
                 + FOLLOW_BIAS * (pose["pitch"] - slope) ** 2)


def solve_pods(seed=7, samples=3000):
    base = [t for t in tiles if t["group"] in ("spine", "spire", "outrigger_wing", "outrigger_dome")]
    base_solids = [solid(t) for t in base]
    base_mags = [m for t in base for m in tile_magnets(t)]
    lo, hi, n = POD_LO, POD_HI, len(POD_KEYS)
    cost = lambda x: pod_cost(dict(zip(POD_KEYS, x)), base_solids, base_mags)
    rng = np.random.default_rng(seed)
    pool = sorted(((cost(x), tuple(x)) for x in rng.uniform(lo, hi, (samples, n))))[:8]
    best = (math.inf, None)
    for c, x in pool:
        x, step = np.array(x), math.radians(8)
        while step > math.radians(0.02):
            moved = False
            for i in range(n):
                for sgn in (1, -1):
                    y = x.copy()
                    y[i] = np.clip(y[i] + sgn * step, lo[i], hi[i])
                    cy = cost(y)
                    if cy < c - 1e-9:
                        x, c, moved = y, cy, True
            if not moved:
                step /= 2
        print(f"  candidate cost {c:.4f}  {np.degrees(x).round(1)}")
        if c < best[0]:
            best = (c, x)
    return dict(zip(POD_KEYS, (round(float(v), 4) for v in best[1]))), best[0]


import sys
if "--solve" in sys.argv:
    POD_POSE, _c = solve_pods()
    print(f"solved pod pose (cost {_c:.4f}): POD_POSE = {POD_POSE}")
    print("  degrees:", {k: round(math.degrees(v), 1) for k, v in POD_POSE.items()})
for _pod in POD_COLOURS:
    tiles.extend(pod_tiles(_pod, POD_POSE))


# The dome keeps only the faces that clear the hull.
while True:
    dome_idx = {i for i, t in enumerate(tiles) if t["group"] == "dome"}
    bad = {i if i in dome_idx else j for _, i, j in collisions(dome_idx)
           if (i in dome_idx) != (j in dome_idx)}
    if not bad:
        break
    tiles[:] = [t for i, t in enumerate(tiles) if i not in bad]


magnets, joints = find_magnets()


# ---------------------------------------------------------------- exporters
def write_obj(path, mode):
    mtl = os.path.splitext(os.path.basename(path))[0] + ".mtl"
    with open(os.path.join(os.path.dirname(path), mtl), "w") as f:
        for name, hx in COLORS.items():
            r, g_, b = (int(hx[i:i + 2], 16) / 255 for i in (1, 3, 5))
            f.write(f"newmtl {name}\nKd {r:.4f} {g_:.4f} {b:.4f}\nKs 0.3 0.3 0.3\nNs 60\n\n")
    with open(path, "w") as f:
        f.write(f"# Magnetic-tile starship blockout ({mode}). Z-up, 1 unit = 1 tile edge.\nmtllib {mtl}\n")
        base = 1
        for t in tiles:
            v, faces = mesh_tile(t["pts"], mode)
            f.write(f"o {t['group']}.{t['name']}\nusemtl {t['color']}\n")
            for p in v:
                f.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
            for face in faces:
                f.write("f " + " ".join(str(base + i) for i in face) + "\n")
            base += len(v)


def write_glb(path, mode):
    """Minimal GLB: one flat-shaded mesh node per tile, grouped under parent nodes, Y-up."""
    blob = bytearray()
    views, accessors, meshes, nodes = [], [], [], []
    mat_names = list(COLORS)
    materials = []
    for name in mat_names:
        hx = COLORS[name]
        rgb = [(int(hx[i:i + 2], 16) / 255) ** 2.2 for i in (1, 3, 5)]
        materials.append({"name": name, "doubleSided": True, "pbrMetallicRoughness": {
            "baseColorFactor": rgb + [1.0], "metallicFactor": 0.0, "roughnessFactor": 0.45}})
    groups = {}
    for t in tiles:
        v, faces = mesh_tile(t["pts"], mode)
        v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)  # Z-up -> Y-up
        pos, nor = [], []
        for face in faces:
            for i in range(1, len(face) - 1):
                tri = v[[face[0], face[i], face[i + 1]]]
                fn = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                fn /= np.linalg.norm(fn)
                pos.append(tri)
                nor.append(np.tile(fn, (3, 1)))
        pos = np.concatenate(pos).astype("<f4")
        nor = np.concatenate(nor).astype("<f4")
        acc = []
        for arr in (pos, nor):
            views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": arr.nbytes, "target": 34962})
            blob += arr.tobytes()
            a = {"bufferView": len(views) - 1, "componentType": 5126, "count": len(arr), "type": "VEC3"}
            if arr is pos:
                a["min"], a["max"] = pos.min(0).tolist(), pos.max(0).tolist()
            accessors.append(a)
            acc.append(len(accessors) - 1)
        meshes.append({"name": t["name"], "primitives": [{
            "attributes": {"POSITION": acc[0], "NORMAL": acc[1]},
            "material": mat_names.index(t["color"])}]})
        nodes.append({"name": t["name"], "mesh": len(meshes) - 1})
        groups.setdefault(t["group"], []).append(len(nodes) - 1)
    roots = []
    for gname, children in groups.items():
        nodes.append({"name": gname, "children": children})
        roots.append(len(nodes) - 1)
    conn = []
    for mg in magnets:
        # local axes: X along the tile side, Y pointing out of the tile, Z the tile normal
        R = np.stack([mg["dir"], mg["out"], mg["normal"]], axis=1)
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float) @ R
        w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        if w > 1e-6:
            q = [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w]
        else:
            i = int(np.argmax(np.diag(R)))
            j, k_ = (i + 1) % 3, (i + 2) % 3
            r = math.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k_, k_]))
            q = [0.0, 0.0, 0.0, 0.0]
            q[i] = r / 2
            q[j] = (R[j, i] + R[i, j]) / (2 * r)
            q[k_] = (R[k_, i] + R[i, k_]) / (2 * r)
        p = mg["pos"]
        nodes.append({"name": f"magnet.{mg['status']}.{mg['tile']}.{mg['edge']}.{mg['slot']}",
                      "translation": [float(p[0]), float(p[2]), float(-p[1])],
                      "rotation": [float(x) for x in q],
                      "extras": {"status": mg["status"], "tile": mg["tile"], "mates": mg["mates"],
                                 "gap": round(mg["gap"], 4) if math.isfinite(mg["gap"]) else None}})
        conn.append(len(nodes) - 1)
    nodes.append({"name": "magnets", "children": conn})
    roots.append(len(nodes) - 1)
    nodes.append({"name": "tile_starship", "children": roots})
    gltf = {"asset": {"version": "2.0", "generator": "build_tiles.py"}, "scene": 0,
            "scenes": [{"nodes": [len(nodes) - 1]}], "nodes": nodes, "meshes": meshes,
            "materials": materials, "accessors": accessors, "bufferViews": views,
            "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(blob), b"BIN\0") + bytes(blob))


def write_magnets(path):
    r4 = lambda a: [round(float(x), 4) for x in a]
    with open(path, "w") as f:
        json.dump({"units": "tile edges, Z-up", "thresholds": {"mated": MATED, "loose": LOOSE},
                   "magnets": [{"tile": m["tile"], "group": m["group"], "edge": m["edge"], "slot": m["slot"],
                                "status": m["status"], "mates": m["mates"],
                                "gap": round(m["gap"], 4) if math.isfinite(m["gap"]) else None,
                                "pos": r4(m["pos"]), "along": r4(m["dir"]), "outward": r4(m["out"]),
                                "normal": r4(m["normal"])} for m in magnets]}, f, indent=1)


def write_viewer(template, path):
    marks = []
    for js_ in joints:
        if js_[0]["status"] == "mated":
            marks.append((np.mean([m_["pos"] for m_ in js_], axis=0), js_[0]["dir"], "mated"))
        else:
            marks += [(m_["pos"], m_["dir"], m_["status"]) for m_ in js_]
    r4 = lambda a: [round(float(x), 4) for x in a]
    data = {"bar": BAR, "thick": THICK, "chamfer": CHAMFER, "clear": CLEAR, "colors": COLORS,
            "magnets": [{"p": r4(p), "d": r4(d), "s": st} for p, d, st in marks],
            "stats": {st: sum(1 for m_ in magnets if m_["status"] == st) for st in ("mated", "loose", "open")},
            "joints": sum(1 for j_ in joints if j_[0]["status"] == "mated"),
            "tiles": [
        {"n": t["name"], "g": t["group"], "c": t["color"],
         "p": [[round(float(x), 4) for x in p] for p in t["pts"]]} for t in tiles]}
    with open(template) as f:
        html = f.read()
    with open(path, "w") as f:
        f.write(html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))


if __name__ == "__main__":
    hits = collisions()
    for d, i, j in sorted(hits, reverse=True):
        print(f"  INTERSECT {tiles[i]['name']} x {tiles[j]['name']}  depth {d:.4f}")
    if hits:
        raise SystemExit(f"{len(hits)} intersecting tile pairs - not exporting")
    print("collision check: 0 intersecting pairs")
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "out")
    os.makedirs(out, exist_ok=True)
    for mode in ("frames", "panels"):
        write_obj(os.path.join(out, f"tile_starship_{mode}.obj"), mode)
        write_glb(os.path.join(out, f"tile_starship_{mode}.glb"), mode)
    write_magnets(os.path.join(out, "magnets.json"))
    write_viewer(os.path.join(here, "viewer.template.html"), os.path.join(out, "viewer.html"))
    counts = {}
    for t in tiles:
        counts[len(t["pts"])] = counts.get(len(t["pts"]), 0) + 1
    st = {k: sum(1 for m in magnets if m["status"] == k) for k in ("mated", "loose", "open")}
    print(f"{len(magnets)} magnets: {st}, {sum(1 for j in joints if j[0]['status'] == 'mated')} mated joints")
    for m in magnets:
        if m["status"] == "loose":
            print(f"  loose {m['tile']}[{m['edge']}] gap {m['gap']:.2f} -> {m['mates']}")
    print(f"{len(tiles)} tiles  (by vertex count: {counts})  ->  {out}")
