from scipy.spatial import Delaunay

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



puntos_de_control = [
    (0, 0, 0),
    (0.2, 0.3, 0),
    (0.4, 0.5, 0),
    (0.6, 0.65, 0),
    (0.75, 0.7, 0),
    (0.85, 0.8, 0),
    (0.95, 0.9, 0),
    (1, 1, 0),
    (0.9, 0.95, 0),
    (0.8, 0.85, 0),
    (0.7, 0.75, 0),
    (0.65, 0.7, 0),
    (0.5, 0.6, 0),
    (0.3, 0.4, 0),
    (0.15, 0.3, 0),
    (0, 0.2, 0),
    (0, 0, 0)
]


tri = Delaunay(puntos_de_control)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Dibuja los triángulos en 3D
tri_mesh = Poly3DCollection(puntos_de_control[tri.simplices], alpha=0.25)
ax.add_collection3d(tri_mesh)

# Establece los límites y etiquetas de los ejes
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)
ax.set_zlim(min_z, max_z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
