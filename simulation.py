from config import DEFAULT_CONFIG
from world import World, Entity, Ant # Entity n'est pas directement utilisé ici, mais peut l'être par World ou Ant
from render import render
from utils import position_distribution
import food

import numpy as np
import random
import time
import pygame
import itertools

from graph import simDataGraph

from concurrent.futures import ThreadPoolExecutor
# Initialise un pool de threads, qui sera utilisé dans les fonctions ci-dessous
# Le nombre de workers peut être ajusté, par exemple os.cpu_count()
executor = ThreadPoolExecutor(max_workers=8)

# Graine pour la génération de nombres aléatoires, basée sur l'heure actuelle pour des exécutions variées par défaut
SEED = int(time.time())
random.seed(SEED)
np.random.seed(SEED)

def init(config):
    """
    Initialise Pygame pour le rendu graphique si la fréquence de rendu est spécifiée.

    Args:
        config (dict): Dictionnaire de configuration contenant les paramètres
                       de la simulation, notamment "width", "height", "zoom",
                       et "render_frequency".

    Returns:
        tuple: Un tuple contenant (screen, font) de Pygame si initialisé,
               sinon (None, None).
    """
    if config["render_frequency"] is not None:
        pygame.init()
        screen = pygame.display.set_mode((config["width"] * config["zoom"], config["height"] * config["zoom"]))
        pygame.display.set_caption('Simulation de Fourmis')
        font = pygame.font.SysFont(None, 24)
        return screen, font
    return None, None

def run(world, ants, n, config, show_freq=10, zoom=1, screen=None, font=None):
    """
    Exécute la simulation pour n étapes.

    Args:
        world (World): L'objet monde contenant la grille et les entités.
        ants (list): Liste des objets Ant à simuler.
        n (int): Nombre d'étapes de simulation à exécuter.
        config (dict): Dictionnaire de configuration.
        show_freq (int, optional): Fréquence d'affichage du rendu. Par défaut 10.
                                   Si None, aucun rendu n'est effectué.
        zoom (int, optional): Facteur de zoom pour le rendu. Par défaut 1.
        screen (pygame.Surface, optional): Surface Pygame pour le rendu. Par défaut None.
        font (pygame.font.Font, optional): Police Pygame pour le texte. Par défaut None.
    """
    # Ajoute de la nourriture dans le monde
    food.distribute(world, config["food_distribution"], config["food_density"])

    # Exécute la simulation
    for i in range(n):
        # Parallel execution of ant steps
        # WARNING: This assumes ant.step(world) is thread-safe or that modifications to 'world'
        # are handled in a thread-safe manner (e.g., internal locking in World/Ant methods,
        # or ant.step returns actions to be applied sequentially).
        # If not, this can lead to race conditions.
        if ants:
            def ant_step_task(ant_obj):
                ant_obj.step(world)

            # The list() call ensures all tasks complete before proceeding to the next simulation step.
            list(executor.map(ant_step_task, ants))

        # Gère le rendu et l'affichage des statistiques si nécessaire
        if show_freq is not None and screen and font and (i % show_freq == 0):
            print("Étape", i)
            render(screen, world, zoom)
            # Calcule les statistiques de nourriture en temps réel
            max_food = 0
            if ants: # S'assure que la liste des fourmis n'est pas vide
                max_food = max(ant.food for ant in ants)
                mean_food = np.mean([ant.food for ant in ants])
            else:
                mean_food = 0.0
            # Affiche le texte en superposition
            stat_text = "Max: {}  Moyenne: {:.2f}".format(max_food, mean_food)
            text_surface = font.render(stat_text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10))
            pygame.display.flip()
            
            # Gère l'événement de fermeture de la fenêtre Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit() # Quitte Pygame proprement
                    # Idéalement, il faudrait un moyen de signaler à la boucle principale de s'arrêter
                    print("Fermeture de Pygame demandée.")
                    return # Sort de la fonction run

def survive(world, ants, config=DEFAULT_CONFIG):
    """
    Fonction de sélection naturelle.

    Conserve un pourcentage des meilleures fourmis (celles avec le plus de nourriture)
    et les retourne. Calcule également un score de diversité génétique et identifie
    la fourmi la plus centrale génétiquement pour la visualisation de son cerveau.

    Args:
        world (World): L'objet monde (actuellement non utilisé dans cette fonction).
        ants (list): Liste des objets Ant actuels.
        config (dict): Dictionnaire de configuration, notamment "survive".

    Returns:
        list: Liste des fourmis survivantes.
    """
    # Les fourmis avec le plus de nourriture survivent ; les autres meurent
    ants.sort(key=lambda x: x.food, reverse=True)
    if ants:
        print("Meilleure fourmi a", ants[0].food, "de nourriture")
        print("Nourriture moyenne:", np.mean([ant.food for ant in ants]))
    else:
        print("Aucune fourmi à évaluer pour la survie.")
        return []
    if config["survive"] < 1:
        num_survivors = int(len(ants) * config["survive"])
        survivors = ants[:num_survivors]
    else :
        # select andts with more than survive food
        survivors = [ant for ant in ants if ant.food > config["survive"]]
    print("Survivants (nourriture):", [ant.food for ant in survivors])

    # Calcule les distances entre les génomes de tous les survivants et un score de diversité
    # et trouve la fourmi la plus proche du centre génétique
    all_pairwise_distances = []
    center_ant_to_visualize = None

    if survivors:
        if len(survivors) == 1:
            center_ant_to_visualize = survivors[0]
            print("Un seul survivant, visualisation de son cerveau.")
        elif len(survivors) > 1:
            survivor_total_distances = {ant: 0.0 for ant in survivors}
            
            combinations_list = list(itertools.combinations(survivors, 2))

            # Helper function for parallel distance calculation
            def calculate_distance_task(pair):
                s1, s2 = pair
                return s1.brain.to_genome().distance(s2.brain.to_genome())

            if combinations_list:
                # Parallel computation of distances
                distances = list(executor.map(calculate_distance_task, combinations_list))
                
                all_pairwise_distances.extend(distances)

                # Update survivor_total_distances sequentially using the computed distances
                for idx, (s1, s2) in enumerate(combinations_list):
                    dist = distances[idx]
                    survivor_total_distances[s1] += dist
                    survivor_total_distances[s2] += dist
            
            # Trouve la fourmi avec la distance totale minimale par rapport aux autres
            if survivor_total_distances: # Ensure dict is not empty
                center_ant_to_visualize = min(survivor_total_distances, key=survivor_total_distances.get)
                print(f"Visualisation du cerveau de la fourmi génétiquement la plus proche du centre (distance totale: {survivor_total_distances[center_ant_to_visualize]:.2f}).")
            else: # Should not happen if len(survivors) > 1 and combinations_list was processed
                center_ant_to_visualize = survivors[0] # Fallback, though logic implies this path isn't taken
                print("Avertissement: survivor_total_distances est vide, sélection du premier survivant.")

        if center_ant_to_visualize and hasattr(center_ant_to_visualize.brain, 'visualize'):
            center_ant_to_visualize.brain.visualize()
            # save genome
            with open("good.genome", "w") as f:
                f.write(center_ant_to_visualize.brain.to_genome().to_string())
    
    # Vérifie si la liste all_pairwise_distances n'est pas vide avant de calculer la moyenne
    if all_pairwise_distances: 
        print("Score de diversité:", np.mean(all_pairwise_distances))
    else:
        if len(survivors) <= 1:
            print("Score de diversité: N/A (moins de 2 survivants)")
        else: # Ne devrait pas arriver si len(survivors) > 1, mais comme solution de repli
            print("Score de diversité: N/A (aucune distance calculée)")

    return survivors

def populate(survivors, n, config=DEFAULT_CONFIG):
    """
    Fonction de reproduction des fourmis.

    Génère n nouvelles fourmis. Si des survivants existent, les nouvelles fourmis
    sont des copies (potentiellement mutées) des survivants. Sinon, de nouvelles
    fourmis aléatoires sont créées.

    Args:
        survivors (list): Liste des fourmis survivantes de l'époque précédente.
        n (int): Nombre total de fourmis à générer pour la nouvelle population.
        config (dict): Dictionnaire de configuration, contenant les paramètres
                       de mutation ("mutation_percent", "mutation_distance") et
                       de positionnement ("position", "width", "height").

    Returns:
        list: Liste des n nouvelles fourmis.
    """
    # Vérification plus idiomatique pour une liste vide
    if not survivors: 
        # Utilise _ pour la variable de boucle non utilisée
        return [Ant(*position_distribution(config["position"], config["width"], config["height"])) for _ in range(n)] 
    
    new_ants = []
    if n == 0:
        return []

    # Première fourmi : une copie du meilleur survivant, placée à une nouvelle position aléatoire.
    # Son cerveau est une copie du cerveau du meilleur survivant, non muté pour cette première copie.
    best_survivor_copy = survivors[0].copy() 
    best_survivor_copy.x, best_survivor_copy.y = position_distribution(config["position"], config["width"], config["height"])
    new_ants.append(best_survivor_copy)

    # Remplit les n-1 places restantes
    if n > 1:
        # Parcourt cycliquement tous les survivants pour sélectionner les parents des nouvelles fourmis
        parent_cycler = itertools.cycle(survivors)
        
        # Prepare parents for the n-1 new ants
        parents_for_generation = [next(parent_cycler) for _ in range(n - 1)]

        def create_and_mutate_ant_task(parent_ant):
            new_ant = parent_ant.copy()
            # Copie explicite du cerveau avant une mutation potentielle, 
            # pour s'assurer que les mutations n'affectent pas le cerveau du parent ou d'autres copies.
            new_ant.brain = parent_ant.brain.copy() 
            if random.random() < config["mutation_percent"]: # Probabilité de mutation pour cette fourmi
                # Calcule la mutation basée sur mutation_distance (pourcentage global de caractères ADN à inverser)
                # et mutation_percent (utilisé ici comme dispersion des paramètres à muter).
                genome = new_ant.brain.to_genome() # Génome du cerveau : n=num_params, m=1, edge_size=DNA_PRECISION_PER_FLOAT
                num_params = genome.n
                dna_precision_per_param = genome.edge_size

                if num_params > 0 and dna_precision_per_param > 0:
                    total_dna_chars = num_params * dna_precision_per_param
                    
                    target_total_char_flips = config["mutation_distance"] * total_dna_chars
                    edge_dispersion_for_brain = config["mutation_dispersion"] 
                    num_params_selected_for_mutation = int(round(num_params * edge_dispersion_for_brain))

                    if edge_dispersion_for_brain > 0 and target_total_char_flips > 0 and num_params_selected_for_mutation == 0 and num_params > 0:
                        num_params_selected_for_mutation = 1
                    
                    distance_val_for_brain_mutate = 0
                    if num_params_selected_for_mutation > 0:
                        distance_val_for_brain_mutate = int(round(target_total_char_flips / num_params_selected_for_mutation))
                    
                    distance_val_for_brain_mutate = max(0, distance_val_for_brain_mutate)

                    new_ant.brain.mutate(
                        distance=distance_val_for_brain_mutate,
                        dispersion=edge_dispersion_for_brain,
                        high_mod=1
                    )
            
            pos = position_distribution(config["position"], config["width"], config["height"])
            new_ant.x, new_ant.y = pos[0], pos[1]
            return new_ant

        # Use executor to create and mutate ants in parallel
        # The list() call ensures all tasks complete.
        additional_ants = list(executor.map(create_and_mutate_ant_task, parents_for_generation))
        new_ants.extend(additional_ants)
            
    return new_ants

def simulate(config=DEFAULT_CONFIG):
    """
    Exécute le cycle complet de la simulation sur plusieurs époques.

    Initialise le monde, les fourmis, et Pygame (si rendu activé).
    Pour chaque époque :
    1. Exécute les étapes de simulation.
    2. Enregistre les statistiques.
    3. Applique la sélection naturelle.
    4. Réinitialise le monde.
    5. Repeuple le monde avec de nouvelles fourmis.
    Finalement, sauvegarde le graphique des statistiques et nettoie Pygame.

    Args:
        config (dict, optional): Dictionnaire de configuration. Utilise DEFAULT_CONFIG
                                 si non fourni.
    """
    from brain import AntBrain 

    ants = []
    world = World(config["width"], config["height"])
    offset = 0
    if config["insert_genomes"] is not None:
        assert type(config["insert_genomes"]) == list, "Les génomes à insérer doivent être fournis sous forme de liste."

        for genome in config["insert_genomes"]:
            ant = Ant(*position_distribution(config["position"], config["width"], config["height"]))
            ant.brain = AntBrain.from_genome(genome)
            ants.append(ant)
            world.add_entity(ant)
            offset += 1

    ants += [Ant(*position_distribution(config["position"], config["width"], config["height"])) for _ in range(config["num_ants"]-offset)]
    for ant in ants:
        world.add_entity(ant)
    

        
    # Initialise Pygame une fois pour toute la simulation si le rendu est activé
    screen, font = init(config)

    graph = simDataGraph(config)
    
    start_time = time.time()

    for epoch in range(config["epochs"]):
        print(f"Exécution de l'époque {epoch}/{config['epochs']}")
        run(world, ants, config["steps_per_epoch"], config, show_freq=config["render_frequency"],
            zoom=config["zoom"], screen=screen, font=font)
        
        # Enregistre les statistiques de nourriture à la fin de l'époque
        current_epoch_foods = [ant.food for ant in ants]
        current_max = max(current_epoch_foods) if current_epoch_foods else 0
        current_mean = np.mean(current_epoch_foods) if current_epoch_foods else 0
        
        graph.update(current_max, current_mean, epoch)

        survivors = survive(world, ants, config)
        world.reset() # Réinitialise le monde (par exemple, enlève la nourriture restante, les anciennes fourmis)
        ants = populate(survivors, config["num_ants"], config)
        for ant in ants:
            world.add_entity(ant) # Ajoute les nouvelles fourmis au monde
    
    elapsed_time = time.time() - start_time
    print(f"Temps écoulé: {elapsed_time:.2f} secondes")
    if config["epochs"] > 0:
        print(f"Temps par époque: {elapsed_time / config['epochs']:.2f} secondes")

    graph.save("graph.png")
    graph.close()

    if screen: # Si Pygame a été initialisé
        pygame.quit()
    
    # Shutdown the executor when simulation is complete
    executor.shutdown(wait=True)

if __name__ == "__main__":
    from brain import AntBrain, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

    config = DEFAULT_CONFIG.copy() # Utilise une copie pour éviter de modifier l'original
    config["height"] = 100
    config["width"] = 100
    config["zoom"] = 20
    config["num_ants"] = 20
    config["render_frequency"] = 10 # Mettre une valeur (ex: 10) pour activer le rendu
    config["food_distribution"] = "spoty"
    config["food_density"] = 0.1
    config["steps_per_epoch"] = 100
    config["epochs"] = 10_000
    # mutation_distance: par exemple, 0.1 signifie que 10% du total des caractères ADN dans le génome sont ciblés pour mutation
    config["mutation_distance"] = 0.2
    config["mutation_dispersion"] = 0.3
    # mutation_percent: probabilité qu'une fourmi mute 
    config["mutation_percent"] = 0.2
    config["survive"] = 0.4

    from genome import Genome
    # Correctly calculate n_params for the Genome object
    if HIDDEN_SIZE > 0:
        n_params = (INPUT_SIZE * HIDDEN_SIZE) + HIDDEN_SIZE + \
                   (HIDDEN_SIZE * OUTPUT_SIZE) + OUTPUT_SIZE
    else:
        n_params = (INPUT_SIZE * OUTPUT_SIZE) + OUTPUT_SIZE
    
    genome_precision = AntBrain.DNA_PRECISION_PER_FLOAT

    try:
        with open("good.genome", "r") as f:
            genome_data_str = f.read().strip()
        
        inserted_genome = Genome(
            n=n_params, 
            m=1,  
            genome_data=genome_data_str,
            precision=genome_precision
        )
        config["insert_genomes"] = [inserted_genome]
    except FileNotFoundError:
        print("Warning: 'good.genome' not found. Proceeding without inserted genomes.")
        config["insert_genomes"] = None
    except ValueError as e:
        print(f"Error initializing genome from 'good.genome': {e}")
        print("Proceeding without inserted genomes.")
        config["insert_genomes"] = None
    
    print("Configuration de la simulation:", config)
    print(f"Graine aléatoire (SEED): {SEED}")
    simulate(config=config)