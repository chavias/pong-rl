import random
from pong import PongGame, PaddleGenome, HumanPlayer


# Define the genome
class PaddleGenome:
    def __init__(self):
        self.position = random.randint(0, SCREEN_HEIGHT - PADDLE_HEIGHT)
        self.velocity = random.randint(-PADDLE_SPEED, PADDLE_SPEED)
    
    def mutate(self):
        if random.random() < MUTATION_RATE:
            self.position = random.randint(0, SCREEN_HEIGHT - PADDLE_HEIGHT)
        if random.random() < MUTATION_RATE:
            self.velocity = random.randint(-PADDLE_SPEED, PADDLE_SPEED)

# Create the initial population
population = [PaddleGenome() for _ in range(POPULATION_SIZE)]

class PongTrainer:
    def __init__(self, population_size, num_generations, mutation_rate):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.population = []

    def train(self):
        # Initialize population
        for _ in range(self.population_size):
            genome = PaddleGenome()
            self.population.append(genome)

        # Evolution loop
        for generation in range(self.num_generations):
            # Test the agents
            for agent1 in self.population:
                for agent2 in self.population:
                    if agent1 != agent2:
                        # Create a new game and play it
                        game = PongGame(agent1, agent2)
                        game.play()

                        # Evaluate fitness
                        agent1_fitness = game.score[0]
                        agent2_fitness = game.score[1]

                        agent1.fitness += agent1_fitness
                        agent2.fitness += agent2_fitness

            # Select parents
            parents = []
            total_fitness = sum(agent.fitness for agent in self.population)
            for _ in range(self.population_size):
                parent = None
                fitness_threshold = random.uniform(0, total_fitness)
                fitness_sum = 0
                for agent in self.population:
                    fitness_sum += agent.fitness
                    if fitness_sum >= fitness_threshold:
                        parent = agent
                        break
                parents.append(parent)

            # Crossover
            offspring = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                child = PaddleGenome()
                for gene in child.genes:
                    if random.random() < 0.5:
                        gene.position = parent1.genes.position
                    else:
                        gene.position = parent2.genes.position
                    if random.random() < 0.5:
                        gene.velocity = parent1.genes.velocity
                    else:
                        gene.velocity = parent2.genes.velocity
                offspring.append(child)

            # Mutation
            for child in offspring:
                child.mutate(self.mutation_rate)

            # Create the next generation
            self.population = offspring

        # Test the final agent
        final_agent = max(self.population, key=lambda agent: agent.fitness)
        game = PongGame(final_agent, HumanPlayer())
        game.play()
