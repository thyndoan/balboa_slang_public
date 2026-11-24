# HW 4.7: a rope toy squiggling in front of a cat.
# uv run python main.py 4_6 scenes/hw4/hw4_7/cat_rope.py

import numpy as np
import trimesh
from pathlib import Path

def make_scene():
    # Load the cat mesh
    mesh = trimesh.load_mesh(Path(__file__).with_name('cat.obj'), process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32).copy()
    triangles = np.asarray(mesh.faces, dtype=np.uint32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32).copy()

    # Make the cat 1.2 tall, centered, and standing on the floor (y = 0)
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    vertices -= (bmin + bmax) / 2
    vertices *= 1.2 / (bmax[1] - bmin[1])
    vertices[:, 1] -= vertices[:, 1].min()

    # Rope: hangs straight down in front of the cat's face
    count = 40
    t = np.linspace(0, 1, count, dtype=np.float32)
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 1] = 1.7 - 1.1 * t  # from y = 1.7 down to y = 0.6
    positions[:, 2] = 0.95           # in front of the cat's face

    edges = np.stack((np.arange(count - 1),
                      np.arange(1, count)), axis=1).astype(np.uint32)
    rest_lengths = np.linalg.norm(
        np.diff(positions, axis=0), axis=1).astype(np.float32)
    inv_masses = np.ones(count, dtype=np.float32)
    inv_masses[0] = 0.0              # pinned point (like a hand holding it)

    # Wavy starting speed, so the rope squiggles side to side
    velocities = np.zeros_like(positions)
    velocities[:, 0] = 6.0 * t * np.sin(4 * np.pi * t)
    velocities[:, 2] = -2.5 * t      # swing a little toward the cat

    return {
        'positions': positions,
        'velocities': velocities,
        'inv_masses': inv_masses,
        'constraint_edges': edges,
        'rest_lengths': rest_lengths,
        'edges': edges,  # Visualization only
        'collision_mesh': {
            'vertices': np.ascontiguousarray(vertices),
            'triangles': np.ascontiguousarray(triangles),
            'vertex_normals': np.ascontiguousarray(normals),
        },
        'planes': np.array([[0, 1, 0, 0]], dtype=np.float32),  # floor
        'spheres': np.zeros((0, 4), dtype=np.float32),
        'collision_distance': 0.02,
        'contact_margin': 0.05,
        'static_friction': 0.45,
        'kinetic_friction': 0.30,
        'substeps': 8,
        'iterations': 20,
        'cam_pos': (2.4, 1.1, 2.2),
        'cam_target': (0.0, 0.6, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
