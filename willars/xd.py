from Bio import SeqIO
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Parámetros BFOA e IA
num_bacterias = 20
num_generaciones = 100
paso_tumbling = 0.05

# Matriz BLOSUM62 manual
matriz_blosum = {
    ('A', 'A'): 4, ('A', 'C'): 0, ('A', 'D'): -2, 
}

# Red Neuronal para predecir el fitness
class FitnessPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(FitnessPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Funciones auxiliares
def cargar_secuencias(archivo_fasta):
    secuencias = []
    for secuencia in SeqIO.parse(archivo_fasta, "fasta"):
        secuencias.append(str(secuencia.seq))
    return secuencias

def calcular_fitness(secuencia1, secuencia2):
    score = 0
    for a, b in zip(secuencia1, secuencia2):
        score += matriz_blosum.get((a, b), matriz_blosum.get((b, a), 0))
    return score

def iniciar_bacterias(secuencias):
    longitud = len(secuencias[0])
    return ["".join(np.random.choice(list("ARNDCQEGHILKMFPSTWYV"), longitud)) for _ in range(num_bacterias)]

def tumbling(bacteria):
    nueva_posicion = list(bacteria)
    idx = np.random.randint(0, len(bacteria))
    nueva_posicion[idx] = np.random.choice(list("ARNDCQEGHILKMFPSTWYV"))
    return "".join(nueva_posicion)

def secuencia_a_vector(secuencia):
    amino_acidos = "ARNDCQEGHILKMFPSTWYV"
    vector = []
    for aa in secuencia:
        vector.extend([1 if aa == x else 0 for x in amino_acidos])
    return vector

# Optimización con BFOA e IA
def optimizar_bfoa_ia(secuencias):
    bacterias = iniciar_bacterias(secuencias)
    mejor_bacteria = None
    mejor_fitness = -float('inf')
    num_evaluaciones = 0

    input_size = len(bacterias[0]) * 20
    predictor = FitnessPredictor(input_size)
    optimizer = optim.Adam(predictor.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for generacion in range(num_generaciones):
        fitness_values = []
        entrenamiento = []

        for i, bacteria in enumerate(bacterias):
            nueva_bacteria = tumbling(bacteria)
            fitness_real = calcular_fitness(bacteria, nueva_bacteria)
            num_evaluaciones += 1
            entrenamiento.append((secuencia_a_vector(bacteria), fitness_real))
            fitness_values.append(fitness_real)

            if fitness_real > mejor_fitness:
                mejor_bacteria = nueva_bacteria
                mejor_fitness = fitness_real

            bacterias[i] = nueva_bacteria

        if generacion > 1:
            datos = torch.tensor([x[0] for x in entrenamiento], dtype=torch.float32)
            labels = torch.tensor([x[1] for x in entrenamiento], dtype=torch.float32).view(-1, 1)
            optimizer.zero_grad()
            output = predictor(datos)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        print(f"Generación {generacion + 1}, Mejor Fitness: {mejor_fitness}, Evaluaciones: {num_evaluaciones}")

    return mejor_bacteria, mejor_fitness, num_evaluaciones

# Ejecución de 30 iteraciones y generación de imágenes
secuencias = cargar_secuencias(r"C:\Users\dilli\OneDrive - Universidad Autonoma de Coahuila\Escritorio\YO\ESCUELA\ModelosComp\willars\multifasta.fasta")
resultados = []

for i in range(30):
    print(f"\n--- Ejecución {i + 1} ---")
    mejor_bacteria, mejor_fitness, total_evaluaciones = optimizar_bfoa_ia(secuencias)
    resultados.append((i + 1, mejor_fitness, total_evaluaciones))

    # Crear la gráfica de resultados para la iteración actual
    fig, ax = plt.subplots()
    ax.bar(range(1, i + 2), [res[1] for res in resultados], color='skyblue')
    ax.set_xlabel('Iteración')
    ax.set_ylabel('Mejor Fitness')
    ax.set_title(f'Iteración {i + 1}: Mejor Fitness hasta ahora')
    plt.tight_layout()
    
    # Guardar la imagen
    plt.savefig(f"resultados_iteracion_{i + 1}.png")
    plt.close(fig)

# Mostrar un resumen general de las 30 ejecuciones
print("\n--- Resumen de 30 Ejecuciones ---")
for i, (num, fitness, evaluaciones) in enumerate(resultados):
    print(f"Ejecución {num}: Mejor fitness: {fitness}, Evaluaciones: {evaluaciones}")
