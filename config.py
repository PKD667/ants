import time
DEFAULT_CONFIG = {
    "log_enabled": True,
    "log_location": "logs",
    "width": 500,
    "height": 500,
    "num_ants": 100,
    "steps_per_epoch": 600,
    "epochs": 10,
    "position": "random",
    "food_distribution": "spoty",
    "food_density": 0.01,
    "mutation_distance": 0.02,  # Percentage of total DNA characters to mutate (e.g., 0.02 for 2%)
    "mutation_percent": 0.2, 
    "mutation_dispersion": 0.2, 
    "seed": int(time.time()),
    "render_frequency": 10,
    "zoom": 1,
    "survive": 1/4,
    "insert_genomes": None
}