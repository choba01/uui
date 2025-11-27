import csv
import sys
import numpy as np


class Node:
    def __init__(self, input_size):
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=input_size)
        self.bias = np.random.normal(loc=0.5, scale=0.01)
    
    def compute(self, input_data):
        """Izračunava weighted sum za ovaj neuron."""
        return np.dot(self.weights, input_data) + self.bias


class Layer:
    def __init__(self, layer_size, input_size):
        self.nodes = [Node(input_size) for _ in range(layer_size)]
        self.size = layer_size
        #self.weights_extracted = [node.weights for node in self.nodes]
        #self.biases_extracted = [node.bias for node in self.nodes]
    
    def forward(self, input_data, activation_fn):
        """Prolazi input kroz sve neurone u sloju i primjenjuje aktivaciju."""
        outputs = np.array([node.compute(input_data) for node in self.nodes])
        return activation_fn(outputs)


class NeuralNet:
    
    def __init__(self, num_layers, layer_size, input_size, input_data, output_data):
        self.layers = []
        current_input_size = input_size
        
        # Hidden layers
        for _ in range(num_layers):
            self.layers.append(Layer(layer_size, current_input_size))
            current_input_size = layer_size
        
        # Output layer (1 neuron)
        self.output_layer = Layer(1, current_input_size)
        
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.input_size = input_size
        self.input_data = input_data
        self.output_data = output_data
        self.fitness = None
        
    def __repr__(self):
        return (f"Neural Network Architecture:\n"
                f"  Hidden layers: {self.num_layers} x {self.layer_size} nodes\n"
                f"  Input size: {self.input_size}\n"
                f"  Output size: 1\n"
                f"  Example weights (first node): {self.layers[0].nodes[0].weights}\n"
                f"  Example bias (first node): {self.layers[0].nodes[0].bias:.4f}\n"
                f"  Input data sample: {self.input_data[0]}\n"
                f"  Output data sample: {self.output_data[0]}")
     
    @staticmethod
    def sigmoid(x):
        """Sigmoidalna aktivacijska funkcija (numerički stabilna)."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
     
    def feed_forward(self, input_row):
        """
        Propagira input kroz mrežu.
        
        Args:
            input_row: NumPy array ulaznih vrijednosti
        
        Returns:
            Skalarna vrijednost output neurona
        """
        activation = np.array(input_row)
        
        # Propagacija kroz hidden layers
        for i, layer in enumerate(self.layers):            
            activation = layer.forward(activation, self.sigmoid)
        
        # Output layer (bez aktivacije ili s linearnom)
        output = self.output_layer.forward(activation, lambda x: x)  # Linearna aktivacija
        
        return float(output)
    
    def error(self, calculated_output, output_data):
        return np.square(np.subtract(output_data, calculated_output)).mean()

def read_csv_file(csv_file):
        dataset = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                dataset.append(row)
        return dataset


# Funkcija za roulette wheel selection
def roulette_wheel_selection(population, total_fitness):
    """
    Odabira jednu jedinku koristeći roulette wheel metodu.
    Veći fitness = veća šansa za odabir.
    """
    pick = np.random.uniform(0, total_fitness)
    current_sum = 0
    for individual in population:
        current_sum += individual.fitness
        if current_sum >= pick:
            return individual
    return population[-1]  # Fallback

def mutate(neural_net, mutation_probability, mutation_scale):
    # Hidden layers
    for layer in neural_net.layers:
        for node in layer.nodes:
            for i in range(len(node.weights)):  # ← Koristi indeks
                if np.random.random() < mutation_probability:
                    node.weights[i] += np.random.normal(0.0, mutation_scale)
            if np.random.random() < mutation_probability:
                node.bias += np.random.normal(0.0, mutation_scale)
    
    # Output layer
    for node in neural_net.output_layer.nodes:
        for i in range(len(node.weights)):
            if np.random.random() < mutation_probability:
                node.weights[i] += np.random.normal(0.0, mutation_scale)
        if np.random.random() < mutation_probability:
            node.bias += np.random.normal(0.0, mutation_scale)

          
def genetic_algorithm(num_layers, layer_size, input_size, input_data, output_data, population_size, elitism, mutation_probability, mutation_scale, num_iterations, test_input, test_output):
    # Početna populacija
    population = [NeuralNet(num_layers, layer_size, input_size, input_data, output_data) for _ in range(population_size)]
    
    # Evaluiraj početnu populaciju
    for neural_net in population:
        calculated_output = np.array([neural_net.feed_forward(row) for row in input_data])
        error = neural_net.error(calculated_output,output_data)
        neural_net.fitness = 1/error
    
    population.sort(key=lambda nn: nn.fitness, reverse=True)
    
    done_iterations = 0
    while done_iterations < num_iterations:
        new_population = []
        
        # Elitizam - kopiraj najbolje (već imaju fitness!)
        new_population.extend(population[:elitism])
        
        if (done_iterations + 1) % 2000 == 0:
            error = population[0].error(np.array([population[0].feed_forward(row) for row in input_data]), output_data)
            print(f"[Train error @{done_iterations+1}]: {error:.6f}")
        
        # Generiraj djecu
        total_fitness = sum(nn.fitness for nn in population)
        while len(new_population) < population_size:
            parent1 = roulette_wheel_selection(population, total_fitness)
            parent2 = roulette_wheel_selection(population, total_fitness)
            while parent2 is parent1:
                parent2 = roulette_wheel_selection(population, total_fitness)
            
            # Crossover
            child = NeuralNet(num_layers, layer_size, input_size, input_data, output_data)
            for i, child_layer in enumerate(child.layers):
                for j, child_node in enumerate(child_layer.nodes):
                    child_node.weights = (parent1.layers[i].nodes[j].weights + parent2.layers[i].nodes[j].weights) / 2
                    child_node.bias = (parent1.layers[i].nodes[j].bias + parent2.layers[i].nodes[j].bias) / 2
            
            for j, child_node in enumerate(child.output_layer.nodes):
                child_node.weights = (parent1.output_layer.nodes[j].weights + parent2.output_layer.nodes[j].weights) / 2
                child_node.bias = (parent1.output_layer.nodes[j].bias + parent2.output_layer.nodes[j].bias) / 2
            
            # Mutacija
            mutate(child, mutation_probability, mutation_scale)
            
            new_population.append(child)
        
        # P = P' (zamijeni populaciju)
        population = new_population
        
        # evaluiraj(P) - evaluiraj SAMO NOVU djecu (ne elite!)
        for i in range(elitism, len(population)):  # ← Preskoči elite!
            calculated_output = np.array([population[i].feed_forward(row) for row in input_data])
            error = population[i].error(calculated_output,output_data)
            population[i].fitness = 1/error
        
        # Sortiraj za sljedeću iteraciju
        population.sort(key=lambda nn: nn.fitness, reverse=True)
        
        done_iterations += 1
    error = population[0].error(np.array([population[0].feed_forward(row) for row in test_input]), test_output)
    print(f"[Test error]:: {error:.6f}")

if __name__ == "__main__":
    args = sys.argv[1:]
    train_data = ''
    test_data = ''
    architecture = ''
    popsize = ''
    elitism = ''
    mutation_prob = ''
    mutation_scale = ''
    num_of_iterations = ''
    for i in range(len(args)):
        if args[i] == '--train':
            train_data = read_csv_file(args[i+1])
        elif args[i] == '--test':
            test_data = read_csv_file(args[i+1])
        elif args[i] == '--nn':
            architecture = args[i+1]
        elif args[i] == '--popsize':
            popsize = args[i+1]
        elif args[i] == '--elitism':
            elitism = args[i+1]
        elif args[i] == '--p':
            mutation_prob = args[i+1]
        elif args[i] == '--K':
            mutation_scale = args[i+1]
        elif args[i] == '--iter':
            num_of_iterations = args[i+1]
    
    
    input_data=np.array([row[:-1] for row in train_data[1:]],dtype=float)
    output_data=np.array([row[-1] for row in train_data[1:]],dtype=float)
    test_input=np.array([row[:-1] for row in test_data[1:]],dtype=float)
    test_output=np.array([row[-1] for row in test_data[1:]],dtype=float)
    #nn = NeuralNet(num_layers=2, layer_size=5, input_size=len(input_data[0]), input_data=input_data, output_data=output_data)
    #calculated_output = np.array([nn.feed_forward(row) for row in input_data])
    
    #error = nn.error(calculated_output)
    #print(f"Mean Squared Error: {error}")
    parts = architecture.strip().split('s')
    layer_sizes = [int(part) for part in parts if part]
    num_layers = len(layer_sizes)
    layer_size = layer_sizes[0]
    
    genetic_algorithm(num_layers, layer_size, len(input_data[0]), input_data, output_data, int(popsize), int(elitism), float(mutation_prob), float(mutation_scale), int(num_of_iterations), test_input, test_output)
    