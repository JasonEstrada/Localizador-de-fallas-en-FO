import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Puntos de datos (vertices)
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0.5, 0.866, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0.5, 0.866, 1]
])

# Definir las caras del prisma triangular
caras = [
    [vertices[0], vertices[1], vertices[2]],
    [vertices[3], vertices[4], vertices[5]],
    [vertices[0], vertices[1], vertices[4], vertices[3]],
    [vertices[1], vertices[2], vertices[5], vertices[4]],
    [vertices[2], vertices[0], vertices[3], vertices[5]]
]

# Crear una figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Dibujar las caras en el gráfico
ax.add_collection3d(Poly3DCollection(caras, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))


def lagrange_interpolation(face_vertices, num_points=100):
    # Obtén los vértices de la cara
    if len(face_vertices) == 3:
        v1, v2, v3 = face_vertices
    else:
        v1, v2, v3, v4 = face_vertices

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

# Configuración del gráfico
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.set_zlabel('Eje Z')
ax.set_title('Prisma Triangular con Caras')

# Mostrar el gráfico
plt.show()








