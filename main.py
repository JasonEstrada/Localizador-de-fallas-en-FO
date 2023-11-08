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

# Definir las caras del prisma
caras = [
    [vertices[0], vertices[1], vertices[2]],
    [vertices[3], vertices[4], vertices[5]]
]

# Visualización de los vértices y las caras
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Dibujar los vértices
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='r', marker='o')

# Dibujar las caras
ax.add_collection3d(Poly3DCollection(caras, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

# Función de interpolación de Lagrange para una cara del prisma 
def lagrange_interpolation(face_vertices, num_points=100):
    # Obtén los vértices de la cara
    v1, v2, v3 = face_vertices

    # Crea una lista de puntos interpolados
    interpolated_points = []

    # Genera puntos intermedios a lo largo de los bordes de la cara
    for i in range(num_points):
        for j in range(num_points):
            # Calcula las coordenadas barycentricas
            u = i / (num_points - 1)
            v = j / (num_points - 1)
            w = 1 - u - v

            # Calcula las coordenadas 3D interpoladas
            x = u * v1[0] + v * v2[0] + w * v3[0]
            y = u * v1[1] + v * v2[1] + w * v3[1]
            z = u * v1[2] + v * v2[2] + w * v3[2]

            interpolated_points.append([x, y, z])

    return np.array(interpolated_points)

# Realizar interpolación en cada cara del prisma
for i, cara in enumerate(caras):
    interpolados = lagrange_interpolation(cara)  # Llama a la función de interpolación
    ax.plot(interpolados[:, 0], interpolados[:, 1], interpolados[:, 2], label=f'Cara {i+1}', color='b')

# Ajustar los límites y etiquetas de los ejes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.legend()
plt.show()