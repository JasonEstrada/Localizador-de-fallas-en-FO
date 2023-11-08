import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Definir los vértices del prisma triangular
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0.5, 0.866, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0.5, 0.866, 1]
])

# Función de interpolación tridimensional
def trilinear_interpolation(v1, v2, v3, v4, v5, v6, u, v, w):
    x = (1 - u) * (1 - v) * (1 - w) * v1[0] + u * (1 - v) * (1 - w) * v2[0] + (1 - u) * v * (1 - w) * v3[0] + u * v * (1 - w) * v4[0] + (1 - u) * (1 - v) * w * v5[0] + u * (1 - v) * w * v6[0] + (1 - u) * v * w * v1[0] + u * v * w * v2[0]
    y = (1 - u) * (1 - v) * (1 - w) * v1[1] + u * (1 - v) * (1 - w) * v2[1] + (1 - u) * v * (1 - w) * v3[1] + u * v * (1 - w) * v4[1] + (1 - u) * (1 - v) * w * v5[1] + u * (1 - v) * w * v6[1] + (1 - u) * v * w * v1[1] + u * v * w * v2[1]
    z = (1 - u) * (1 - v) * (1 - w) * v1[2] + u * (1 - v) * (1 - w) * v2[2] + (1 - u) * v * (1 - w) * v3[2] + u * v * (1 - w) * v4[2] + (1 - u) * (1 - v) * w * v5[2] + u * (1 - v) * w * v6[2] + (1 - u) * v * w * v1[2] + u * v * w * v2[2]
    return x, y, z

# Visualización del volumen del prisma
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Dibujar los vértices
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='r', marker='o')

# Generar puntos interpolados en todo el volumen del prisma
num_points = 20
for i in range(num_points):
    for j in range(num_points):
        for k in range(num_points):
            u = i / (num_points - 1)
            v = j / (num_points - 1)
            w = k / (num_points - 1)
            x, y, z = trilinear_interpolation(vertices[0], vertices[1], vertices[2], vertices[3], vertices[4], vertices[5], u, v, w)
            ax.scatter(x, y, z, c='b', marker='.')

# Ajustar los límites y etiquetas de los ejes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
