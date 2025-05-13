from dataclasses import dataclass
import numpy as np
import igraph as ig

from genome import Genome, dna_to_f32, f32_to_dna

HIDDEN_SIZE = 0 # set to 0 for no hidden layer
OUTPUT_SIZE = 4  # UP/DOWN/LEFT/RIGHT
INPUT_SIZE = 9   # 3x3 grid 

DEBUG = False

@dataclass
class BrainData:
    weights: list[np.ndarray]  # Corrected type hint for clarity
    biases: list[np.ndarray]


class AntBrain:
    DNA_PRECISION_PER_FLOAT = 16  # Number of DNA characters used to represent one float parameter

    def __init__(self, data: BrainData):
        self.data = data

    @staticmethod
    def random():
        if HIDDEN_SIZE > 0:
            weights = []
            biases = []
            weights.append(np.random.randn(INPUT_SIZE, HIDDEN_SIZE))
            weights.append(np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE))
            biases.append(np.random.randn(HIDDEN_SIZE))
            biases.append(np.random.randn(OUTPUT_SIZE))
        else:
            weights = [np.random.randn(INPUT_SIZE, OUTPUT_SIZE)]
            biases = [np.random.randn(OUTPUT_SIZE)]
        data = BrainData(weights, biases)
        return AntBrain(data)
    
    def mutate(self, distance, dispersion=0.5, high_mod=0, rm_char=False): 
        g = self.to_genome()
        g.mutate(distance=distance, dispersion=dispersion, high_mod=high_mod)
        self.data = AntBrain.from_genome(g).data

    # forward pass
    def __call__(self, inputs):

        import torch

        tensor_inputs = torch.tensor(inputs, dtype=torch.double)

        # application des poids et biais
        if HIDDEN_SIZE > 0:
            t = torch.matmul(tensor_inputs, torch.tensor(self.data.weights[0])) + torch.tensor(self.data.biases[0])
            if DEBUG:
                print(f"DEBUG: layer 1 pre-activation: {t}")
            t = torch.maximum(t, torch.tensor(0))
            if DEBUG:
                print(f"DEBUG: layer 1 post-ReLU: {t}")
            t = torch.matmul(t, torch.tensor(self.data.weights[1],dtype=torch.double)) + torch.tensor(self.data.biases[1])
            t = torch.maximum(t, torch.tensor(0.1))
            if DEBUG:
                print(f"DEBUG: layer 2 output: {t}")
                print(f"DEBUG: __call__ output shape: {t.shape}")
        else:
            t = torch.matmul(tensor_inputs, torch.tensor(self.data.weights[0],dtype=torch.double)) + torch.tensor(self.data.biases[0])
            t = torch.maximum(t, torch.tensor(0.1))
            if DEBUG:
                print(f"DEBUG: direct layer output: {t}")
        return t.numpy()

    # perception et mouvement
    def act(self, view):
        view = view.flatten()
        inputs = np.array(view)
        t = self(inputs)

        # Output layer is always OUTPUT_SIZE neurons:
        # 0: up, 1: right, 2: down, 3: left
        up = t[0]
        right = t[1]
        down = t[2]
        left = t[3]

        direction = np.argmax([up, right, down, left])
        return direction


    def visualize(self, filename="brain.png"):
        def get_edge_color(weight_val):
            # Handle NaN or non-numeric weights first
            if not isinstance(weight_val, (float, int, np.number)) or np.isnan(weight_val):
                return "#808080"  # Return a neutral gray color for NaN or invalid weights

            # Blue for positive, red for negative, gray for near zero
            if abs(weight_val) < 0.1:
                return "#cccccc"
            
            # Original color intensity logic
            component_val = int(200 - min(abs(weight_val) * 100.0, 150.0))

            if weight_val > 0:
                # Positive weights: blue (format #RRGGBB)
                return f"#{component_val:02x}{component_val:02x}ff"
            else:
                # Negative weights: red (format #RRGGBB)
                # Ensure the hex string starts with #
                return f"#ff{component_val:02x}{component_val:02x}"

        def get_edge_width(weight_val):
            # Handle NaN or non-numeric weights first
            if not isinstance(weight_val, (float, int, np.number)) or np.isnan(weight_val):
                return 1.0  # Default width for NaN or invalid weights
            # Thicker for stronger weights
            return 1 + abs(weight_val) * 3

        def get_edge_label(weight_val):
            # Show sign and value, smaller font for small weights
            if not isinstance(weight_val, (float, int, np.number)) or np.isnan(weight_val):
                return "NaN"
            return f"{weight_val:+.2f}"

        if HIDDEN_SIZE > 0:
            num_vertices = INPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE
            g = ig.Graph(directed=True)
            labels = [f"input {i}" for i in range(INPUT_SIZE)] + \
                        [f"hidden {i}" for i in range(HIDDEN_SIZE)] + \
                        [f"output {i}" for i in range(OUTPUT_SIZE)]
            g.add_vertices(num_vertices)
            g.vs["label"] = labels

            # Increase horizontal spacing between layers
            def column_layout(count, x_coord_base, y_spacing_factor=10.0):
                return [[x_coord_base, y * y_spacing_factor] for y in np.linspace((count - 1)/2.0, -(count - 1)/2.0, count)]

            layout = []
            layout.extend(column_layout(INPUT_SIZE, 0))
            layout.extend(column_layout(HIDDEN_SIZE, 2))  # Increased x-coordinate
            layout.extend(column_layout(OUTPUT_SIZE, 4))  # Increased x-coordinate

            threshold = 0.1  # threshold below which connection is omitted

            # Input -> Hidden connections
            for i in range(INPUT_SIZE):
                for j in range(HIDDEN_SIZE):
                    weight = self.data.weights[0][i][j]
                    weight_val = float(weight)
                    if abs(weight_val) >= threshold or np.isnan(weight_val):  # Also draw NaN edges to see them if any (colored gray by get_edge_color)
                        g.add_edge(i, INPUT_SIZE + j)
                        g.es[-1]["label"] = get_edge_label(weight_val)
                        g.es[-1]["color"] = get_edge_color(weight_val)
                        g.es[-1]["width"] = get_edge_width(weight_val)

            # Hidden -> Output connections
            for i in range(HIDDEN_SIZE):
                for j in range(OUTPUT_SIZE):
                    weight = self.data.weights[1][i][j]
                    weight_val = float(weight)
                    if abs(weight_val) >= threshold or np.isnan(weight_val):
                        g.add_edge(INPUT_SIZE + i, INPUT_SIZE + HIDDEN_SIZE + j)
                        g.es[-1]["label"] = get_edge_label(weight_val)
                        g.es[-1]["color"] = get_edge_color(weight_val)
                        g.es[-1]["width"] = get_edge_width(weight_val)
        else:
            num_vertices = INPUT_SIZE + OUTPUT_SIZE
            g = ig.Graph(directed=True)
            labels = [f"input {i}" for i in range(INPUT_SIZE)] + \
                        [f"output {i}" for i in range(OUTPUT_SIZE)]
            g.add_vertices(num_vertices)
            g.vs["label"] = labels

            # Increase horizontal spacing between layers
            def column_layout(count, x_coord_base, y_spacing_factor=1.0):
                return [[x_coord_base, y * y_spacing_factor] for y in np.linspace((count - 1)/2.0, -(count - 1)/2.0, count)]

            layout = []
            layout.extend(column_layout(INPUT_SIZE, 0))
            layout.extend(column_layout(OUTPUT_SIZE, 2))  # Increased x-coordinate

            threshold = 0.1
            for i in range(INPUT_SIZE):
                for j in range(OUTPUT_SIZE):
                    weight = self.data.weights[0][i][j]
                    weight_val = float(weight)
                    if abs(weight_val) >= threshold or np.isnan(weight_val):  # Also draw NaN edges
                        g.add_edge(i, INPUT_SIZE + j)
                        g.es[-1]["label"] = get_edge_label(weight_val)
                        g.es[-1]["color"] = get_edge_color(weight_val)
                        g.es[-1]["width"] = get_edge_width(weight_val)

        visual_style = {
            "vertex_size": 30,  # Slightly smaller nodes might help with dense layers
            "vertex_label_size": 16,  # Slightly smaller labels
            "edge_label_size": 10,  # Smaller edge labels for less clutter
            "layout": layout,
            "bbox": (900, 700),  # Increased bounding box for more space
            "margin": 100,  # Increased margin
            "edge_curved": 0.0,  # Straight edges
            "edge_arrow_size": 0.8,  # Adjust arrow size if needed
            "edge_arrow_width": 0.8,  # Adjust arrow width if needed
        }
        ig.plot(g, target=filename, **visual_style)


    @staticmethod
    def from_genome(genome: Genome):
        # Calculate the expected number of parameters for the brain configuration
        num_expected_params = 0
        if HIDDEN_SIZE > 0:
            num_expected_params = (INPUT_SIZE * HIDDEN_SIZE) + HIDDEN_SIZE + \
                                  (HIDDEN_SIZE * OUTPUT_SIZE) + OUTPUT_SIZE
        else:
            num_expected_params = (INPUT_SIZE * OUTPUT_SIZE) + OUTPUT_SIZE

        # Validate genome structure
        assert genome.n == num_expected_params, \
            f"Genome parameter count mismatch. Genome has {genome.n} parameters, brain expects {num_expected_params}."
        assert genome.m == 1, \
            f"Genome structure mismatch. Expected m=1 (single DNA string per parameter), got m={genome.m}."
        assert genome.edge_size == AntBrain.DNA_PRECISION_PER_FLOAT, \
            f"Genome DNA precision mismatch. Genome has {genome.edge_size}, brain expects {AntBrain.DNA_PRECISION_PER_FLOAT}."

        all_params_floats = []
        for i in range(genome.n):  # genome.n is the total number of parameters
            dna_edge_list = genome.get_edge(i)  # Returns list of chars for the i-th parameter
            dna_edge_str = "".join(dna_edge_list)
            float_val = dna_to_f32(dna_edge_str)
            
            # dna_to_f32 sums values from a table; 'X' or unknown chars are 0.
            # NaN is unlikely unless dna_edge_str is empty and dna_to_f32 handles it poorly,
            # or table values lead to NaN, but good practice to check.
            if np.isnan(float_val):
                float_val = 0.0  # Default to 0.0 if NaN occurs
            all_params_floats.append(float_val)

        ptr = 0  # Pointer for iterating through all_params_floats

        def extract_layer_params(num_inputs, num_outputs, params_list, current_ptr):
            # Extract weights
            weights_shape = (num_inputs, num_outputs)
            num_weight_elements = num_inputs * num_outputs
            weight_elements = params_list[current_ptr : current_ptr + num_weight_elements]
            weights_matrix = np.array(weight_elements).reshape(weights_shape)
            current_ptr += num_weight_elements

            # Extract biases
            num_bias_elements = num_outputs
            bias_elements = params_list[current_ptr : current_ptr + num_bias_elements]
            biases_vector = np.array(bias_elements)
            current_ptr += num_bias_elements
            
            return weights_matrix, biases_vector, current_ptr

        if HIDDEN_SIZE > 0:
            weights0, biases0, ptr = extract_layer_params(INPUT_SIZE, HIDDEN_SIZE, all_params_floats, ptr)
            weights1, biases1, ptr = extract_layer_params(HIDDEN_SIZE, OUTPUT_SIZE, all_params_floats, ptr)
            data = BrainData(weights=[weights0, weights1], biases=[biases0, biases1])
        else:
            weights0, biases0, ptr = extract_layer_params(INPUT_SIZE, OUTPUT_SIZE, all_params_floats, ptr)
            data = BrainData(weights=[weights0], biases=[biases0])
        
        assert ptr == num_expected_params, \
            f"Parameter consumption error. Consumed {ptr} parameters, but expected {num_expected_params}."

        return AntBrain(data)

    def to_genome(self) -> Genome:
        all_params_flat = []
        
        # Flatten weights
        for weight_matrix in self.data.weights:
            for weight_value in np.nditer(weight_matrix):
                all_params_flat.append(float(weight_value))
        
        # Flatten biases
        for bias_vector in self.data.biases:
            for bias_value in np.nditer(bias_vector):
                all_params_flat.append(float(bias_value))

        num_total_params = len(all_params_flat)

        g = Genome(n=num_total_params, m=1, precision=AntBrain.DNA_PRECISION_PER_FLOAT)

        for idx, param_val in enumerate(all_params_flat):
            dna_string = f32_to_dna(param_val, size=AntBrain.DNA_PRECISION_PER_FLOAT)
            g.set_edge(edge_k_idx=idx, edge_dna_str=dna_string)
            
        return g
    
    def copy(self):
        new_brain = AntBrain(self.data)
        return new_brain


if __name__ == "__main__":
    if HIDDEN_SIZE > 0:
        weights = []
        biases = []
        weights.append(np.random.randn(INPUT_SIZE, HIDDEN_SIZE))
        weights.append(np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE))
        biases.append(np.random.randn(HIDDEN_SIZE))
        biases.append(np.random.randn(OUTPUT_SIZE))
    else:
        weights = [np.random.randn(INPUT_SIZE, OUTPUT_SIZE)]
        biases = [np.random.randn(OUTPUT_SIZE)]
    
    data = BrainData(weights, biases)
    brain = AntBrain(data)

    brain.visualize()

    grid = [[0, 0, 0], [0, 1, 0], [1, 0, 1]]
    grid = np.array(grid)
    grid = grid.flatten()
    direction = brain.act(grid)

    directions = ["up", "right", "down", "left"]
    print(f"Direction: {directions[direction]}")

    genome = brain.to_genome()
    print(genome)

    new_brain = AntBrain.from_genome(genome)
    print(new_brain.to_genome())

    new_brain.visualize("new_brain.png")