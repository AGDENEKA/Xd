from bacteria import bacteria
from chemiotaxis import chemiotaxis
import numpy as np
import matplotlib.pyplot as plt

# Parameters and initializations
poblacion = []
path = r"C:\Users\dilli\OneDrive - Universidad Autonoma de Coahuila\Escritorio\YO\ESCUELA\ModelosComp\willars\fastaReader.py"

numeroDeBacterias = 5
numRandomBacteria = 1
iteraciones = 30
tumbo = 1  # Number of gaps to insert
nado = 3
chemio = chemiotaxis()
veryBest = bacteria(path)  # Best bacteria
tempBacteria = bacteria(path)  # Temporary bacteria for validation
original = bacteria(path)  # Original bacteria without gaps
globalNFE = 0  # Number of function evaluations

# Attraction and repulsion parameters
dAttr = 0.1
wAttr = 0.2
hRep = dAttr
wRep = 10

# List to store best fitness values at each iteration
fitness_values = []

# Cloning the best bacteria
def clonaBest(veryBest, best):
    veryBest.matrix.seqs = np.array(best.matrix.seqs)
    veryBest.blosumScore = best.blosumScore
    veryBest.fitness = best.fitness
    veryBest.interaction = best.interaction

# Validate sequences
def validaSecuencias(path, veryBest):
    tempBacteria.matrix.seqs = np.array(veryBest.matrix.seqs)
    for i in range(len(tempBacteria.matrix.seqs)):
        tempBacteria.matrix.seqs[i] = tempBacteria.matrix.seqs[i].replace("-", "")
    for i in range(len(tempBacteria.matrix.seqs)):
        if tempBacteria.matrix.seqs[i] != original.matrix.seqs[i]:
            print("*****************Secuencias no coinciden********************")
            return

# Initial population
for i in range(numeroDeBacterias):
    poblacion.append(bacteria(path))

# Main loop with 30 iterations
for iteration in range(iteraciones):
    for bacteria in poblacion:
        bacteria.tumboNado(tumbo)
        bacteria.autoEvalua()
    chemio.doChemioTaxis(poblacion, dAttr, wAttr, hRep, wRep)
    globalNFE += chemio.parcialNFE
    best = max(poblacion, key=lambda x: x.fitness)
    if (veryBest is None) or (best.fitness > veryBest.fitness):
        clonaBest(veryBest, best)
    
    # Save the best fitness for this iteration
    fitness_values.append(veryBest.fitness)
    
    # Print iteration summary
    print(f"Iteración {iteration + 1}: Fitness máximo = {veryBest.fitness}, NFE = {globalNFE}")

    chemio.eliminarClonar(path, poblacion)
    chemio.insertRamdomBacterias(path, numRandomBacteria, poblacion)

# Plotting the fitness progression over all iterations
plt.figure(figsize=(10, 6))
plt.plot(range(1, iteraciones + 1), fitness_values, marker='o', color='b', linestyle='-')
plt.title("Evolución de la Aptitud Máxima en Cada Iteración")
plt.xlabel("Iteración")
plt.ylabel("Fitness Máximo")
plt.grid(True)
plt.savefig("fitness_progression.png")
plt.show()

# Display genome of the best bacteria and validate sequences
veryBest.showGenome()
validaSecuencias(path, veryBest)
