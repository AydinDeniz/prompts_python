
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Problem setup: Traveling Salesman Problem (TSP)
# Define the coordinates of cities
CITIES = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(20)]

# Distance calculation
def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

def total_distance(individual):
    return sum(distance(CITIES[individual[i]], CITIES[individual[i + 1]]) for i in range(len(individual) - 1)) + distance(CITIES[individual[-1]], CITIES[individual[0]])

# DEAP setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

toolbox.register("indices", random.sample, range(len(CITIES)), len(CITIES))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", total_distance)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Genetic Algorithm parameters
POP_SIZE = 100
N_GEN = 200
CX_PROB = 0.8
MUT_PROB = 0.2

def main():
    population = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # Run the Genetic Algorithm
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB, ngen=N_GEN,
                                               stats=stats, halloffame=hof, verbose=True)

    # Extract the best individual
    best_individual = hof[0]
    print("Best route:", best_individual)
    print("Shortest distance:", total_distance(best_individual))

    # Convergence plotting
    gen = logbook.select("gen")
    fit_min = logbook.select("min")
    fit_avg = logbook.select("avg")
    plt.plot(gen, fit_min, label="Minimum Fitness")
    plt.plot(gen, fit_avg, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.title("Convergence of Genetic Algorithm")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
