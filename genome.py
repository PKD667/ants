import random
import re  # Pour la validation des entrées de from_binary
import struct  # Pour empaqueter/dépaqueter les flottants
import itertools
from typing import List, Dict, Union, Tuple, Optional, TypeVar, Type

# Type générique pour la classe Genome elle-même, utilisé pour les annotations de type.
GenomeType = TypeVar('GenomeType', bound='Genome')

# Table de correspondance entre les caractères ADN et leurs valeurs numériques.
# Utilisée pour convertir les séquences d'ADN en valeurs flottantes.
# Le but, c'est d'avoir un encodage qui soit a peu près linéaire.
# On veut pouvoir faire des mutations de manière a ce que les valeurs flottantes
# soient proches de la valeur flottante d'origine.
table: Dict[str, float] = {
    "A":  0.5,
    "T": 0.05,
    "C": -0.2,
    "G": -0.01,
}

def dna_to_f32(dna: str) -> float:
    """
    Convertit une chaîne d'ADN en une valeur float32.
    Chaque caractère est mappé à un flottant en utilisant la `table`.
    La somme des valeurs des caractères constitue le flottant résultant.
    """
    # Somme des valeurs de chaque caractère ADN, ou 0.0 si le caractère n'est pas dans la table.
    return sum(table.get(c, 0.0) for c in dna)


def f32_to_dna(f32_val: float, size: int = 16) -> str:
    """
    Conversion "gourmande" d'une valeur float32 en une chaîne d'ADN de longueur `size`.
    À chaque position dans la chaîne d'ADN résultante, choisit la base (A, T, C, G)
    dont la valeur est la plus proche de la moyenne nécessaire pour les positions restantes
    afin d'atteindre `f32_val`.
    """
    if size == 0:
        return ""  # Retourne une chaîne vide si la taille demandée est nulle.
        
    dna_bases: List[str] = list(table.keys())  # Liste des bases ADN possibles (A, T, C, G)
    base_values: List[float] = [table[c] for c in dna_bases] # Leurs valeurs numériques correspondantes

    dna_chars: List[str] = []  # Liste pour construire la chaîne d'ADN résultante
    current_sum: float = 0.0  # Somme actuelle des valeurs des bases choisies

    for i in range(size):
        rem_slots: int = size - i  # Nombre de positions restantes à remplir
        # Moyenne nécessaire pour chaque position restante pour atteindre la f32_val cible
        avg_needed: float = (f32_val - current_sum) / rem_slots

        # Trouve l'index de la base dont la valeur est la plus proche de la moyenne nécessaire
        best_idx: int = min(
            range(len(dna_bases)),
            key=lambda idx: abs(base_values[idx] - avg_needed)
        )
        dna_chars.append(dna_bases[best_idx]) # Ajoute la meilleure base à la séquence
        current_sum += base_values[best_idx] # Met à jour la somme actuelle

    return "".join(dna_chars) # Retourne la chaîne d'ADN construite


class Genome():
    """
    Représente le génome d'une fourmi, utilisé pour encoder son "cerveau".

    Le génome est organisé symboliquement comme suit :
    Noeud 1:
        - Arête 1 (F32 = 4 octets = 16 caractères, si precision=16)
        ...
        - Arête m (F32 = 4 octets = 16 caractères, si precision=16)
    ...
    Noeud n:
        - Arête 1 (F32 = 4 octets = 16 caractères, si precision=16)
        ...
        - Arête m (F32 = 4 octets = 16 caractères, si precision=16)
    
    La taille totale est n * m * 4 octets = n * m * precision caractères.
    Les arêtes manquantes peuvent être représentées par un caractère spécial (par ex. "X").
    """
    char_set: str = "ATCG" # Ensemble des caractères valides pour l'ADN (hors 'X')

    @classmethod
    def rand(cls: Type[GenomeType], n: int, m: int, precision: int = 16) -> GenomeType:
        """
        Crée une instance de Genome avec une séquence d'ADN aléatoire.

        Args:
            n: Nombre de nœuds.
            m: Nombre d'arêtes par nœud.
            precision: Nombre de caractères ADN par arête (taille de l'arête).

        Returns:
            Une nouvelle instance de Genome.
        """
        # Génère une chaîne aléatoire de la longueur requise en utilisant les caractères de char_set.
        random_genome_str = "".join(random.choices(cls.char_set, k=n * m * precision))
        return cls(n, m, random_genome_str, precision=precision)
    
    def __init__(self, n: int, m: int, genome_data: Optional[Union[str, List[str]]] = None, precision: int = 16):
        """
        Initialise un génome.

        Args:
            n: Nombre de nœuds.
            m: Nombre d'arêtes par nœud.
            genome_data: La séquence du génome sous forme de chaîne, de liste de caractères, ou None.
                         Si None, le génome est initialisé avec des 'X' (arêtes manquantes).
            precision: Nombre de caractères ADN par arête.
        """
        self.n: int = n  # Nombre de nœuds
        self.m: int = m  # Nombre d'arêtes par nœud
        self.edge_size: int = precision  # Nombre de caractères par arête
        total_length: int = n * m * self.edge_size # Longueur totale attendue de la chaîne du génome

        if genome_data is None:
            # Si aucune donnée de génome n'est fournie, initialise avec des 'X' (indiquant des arêtes manquantes).
            self.genome: List[str] = list("X" * total_length)
        else:
            if isinstance(genome_data, str):
                genome_list = list(genome_data)
            elif isinstance(genome_data, list):
                genome_list = genome_data
            else:
                raise TypeError("Le génome doit être une chaîne de caractères, une liste de caractères, ou None.")

            if len(genome_list) != total_length:
                raise ValueError(
                    f"La longueur du génome ({len(genome_list)}) est incorrecte. "
                    f"Attendu : {total_length} (n={n}, m={m}, precision={precision})."
                )
            
            # Vérifie que tous les caractères sont valides (ATCG ou X).
            valid_chars = self.char_set + "X"
            if not all(c in valid_chars for c in genome_list):
                invalid_chars_found = set(c for c in genome_list if c not in valid_chars)
                raise ValueError(
                    f"Le génome contient des caractères invalides : {invalid_chars_found}. "
                    f"Caractères autorisés : {valid_chars}"
                )
            self.genome: List[str] = genome_list
    
    def __str__(self) -> str:
        """Retourne la représentation en chaîne de caractères du génome."""
        return "".join(self.genome)
    
    def high_view(self) -> str:
        """
        Fournit une vue d'ensemble de l'état de chaque arête du génome.
        'O': Arête complète (uniquement ATCG).
        'X': Arête manquante (uniquement X).
        'M': Arête mixte (contient des ATCG et des X).
        'E': Arête vide ou état inattendu (ne devrait pas se produire avec une logique correcte).
        """
        view_chars: List[str] = []
        num_total_edges = self.n * self.m
        for edge_k_idx in range(num_total_edges):
            start_idx = edge_k_idx * self.edge_size
            end_idx = start_idx + self.edge_size
            edge_chars = self.genome[start_idx:end_idx]
            
            # Vérifie la présence de caractères ADN (ATCG) et du caractère 'X'.
            has_atcg = any(c in self.char_set for c in edge_chars)
            has_x = any(c == "X" for c in edge_chars)

            if has_atcg and not has_x:
                view_chars.append("O") # Arête complète (Only ATCG)
            elif not has_atcg and has_x:
                view_chars.append("X") # Arête manquante (Only X)
            elif has_atcg and has_x:
                view_chars.append("M") # Arête mixte (Mixed ATCG and X)
            else: # Ni ATCG ni X, ou une arête vide si edge_size est 0
                if not edge_chars: # Cas d'une arête de taille 0
                    view_chars.append("E") # Empty/Error
                elif all(c == "X" for c in edge_chars): # Devrait être couvert par le deuxième elif
                    view_chars.append("X")
                else: # État inattendu, par exemple si char_set est vide et edge_chars non vide
                    view_chars.append("E") # Error/Unexpected
        return "".join(view_chars)

    def get_node(self, node_idx: int) -> List[str]:
        """
        Récupère toutes les arêtes (sous forme de chaînes ADN) d'un nœud spécifique.

        Args:
            node_idx: Index du nœud (0 à n-1).

        Returns:
            Une liste de chaînes, où chaque chaîne représente une arête du nœud.
        """
        if not (0 <= node_idx < self.n):
            raise IndexError("Index de nœud hors limites.")
        
        node_edges: List[str] = []
        # Calcule l'index de début des caractères pour ce nœud dans le génome global.
        node_start_char_idx = node_idx * self.m * self.edge_size
        for edge_in_node_idx in range(self.m):
            # Calcule les indices de début et de fin pour chaque arête du nœud.
            edge_start_char_idx = node_start_char_idx + (edge_in_node_idx * self.edge_size)
            edge_end_char_idx = edge_start_char_idx + self.edge_size
            edge_dna_list = self.genome[edge_start_char_idx:edge_end_char_idx]
            node_edges.append("".join(edge_dna_list))
        return node_edges

    def set_node(self, node_idx: int, node_edges_dna: List[str]) -> None:
        """
        Définit toutes les arêtes d'un nœud spécifique.

        Args:
            node_idx: Index du nœud (0 à n-1).
            node_edges_dna: Une liste de chaînes ADN, une pour chaque arête du nœud.
        """
        if not (0 <= node_idx < self.n):
            raise IndexError("Index de nœud hors limites.")
        if len(node_edges_dna) != self.m:
            raise ValueError(
                f"Le nombre d'arêtes fournies ({len(node_edges_dna)}) est incorrect. "
                f"Attendu : {self.m} pour le nœud {node_idx}."
            )
        
        node_start_char_idx = node_idx * self.m * self.edge_size
        for edge_in_node_idx in range(self.m):
            edge_dna_str = node_edges_dna[edge_in_node_idx]
            if len(edge_dna_str) != self.edge_size:
                raise ValueError(
                    f"La longueur de l'arête ({len(edge_dna_str)}) pour l'arête {edge_in_node_idx} "
                    f"du nœud {node_idx} est incorrecte. Attendu : {self.edge_size}."
                )
            
            valid_chars = self.char_set + "X"
            if not all(c in valid_chars for c in edge_dna_str):
                raise ValueError(
                    f"La chaîne d'arête '{edge_dna_str}' contient des caractères invalides. "
                    f"Autorisés : {valid_chars}"
                )

            edge_start_char_idx = node_start_char_idx + (edge_in_node_idx * self.edge_size)
            # Met à jour la portion correspondante du génome.
            self.genome[edge_start_char_idx : edge_start_char_idx + self.edge_size] = list(edge_dna_str)

    def get_edge(self, edge_k_idx: int) -> List[str]:
        """
        Récupère une arête spécifique par son index global.

        Args:
            edge_k_idx: Index global de l'arête (0 à n*m-1).

        Returns:
            Une liste de caractères représentant l'ADN de l'arête.
        """
        if not (0 <= edge_k_idx < self.n * self.m):
            raise IndexError("Index d'arête hors limites.")
        start_idx = edge_k_idx * self.edge_size
        end_idx = start_idx + self.edge_size
        return self.genome[start_idx:end_idx]

    def set_edge(self, edge_k_idx: int, edge_dna_str: str) -> None:
        """
        Définit une arête spécifique par son index global.

        Args:
            edge_k_idx: Index global de l'arête.
            edge_dna_str: La chaîne ADN pour cette arête.
        """
        if not (0 <= edge_k_idx < self.n * self.m):
            raise IndexError("Index d'arête hors limites.")
        if len(edge_dna_str) != self.edge_size:
            raise ValueError(
                f"La longueur de la chaîne d'arête ({len(edge_dna_str)}) est incorrecte. "
                f"Attendu : {self.edge_size}."
            )
        
        valid_chars = self.char_set + "X"
        if not all(c in valid_chars for c in edge_dna_str):
            raise ValueError(
                f"La chaîne d'arête '{edge_dna_str}' contient des caractères invalides. "
                f"Autorisés : {valid_chars}"
            )

        start_idx = edge_k_idx * self.edge_size
        self.genome[start_idx : start_idx + self.edge_size] = list(edge_dna_str)

    def rm_edge(self, edge_k_idx: int) -> List[str]:
        """
        Supprime une arête (la remplace par des 'X').

        Args:
            edge_k_idx: Index global de l'arête à supprimer.

        Returns:
            L'ADN de l'arête originale avant sa suppression.
        """
        if not (0 <= edge_k_idx < self.n * self.m):
            raise IndexError("Index d'arête hors limites.")
        
        original_edge_dna_list = self.get_edge(edge_k_idx)
        # Remplace l'arête par une séquence de 'X' de la bonne longueur.
        self.genome[edge_k_idx * self.edge_size : (edge_k_idx + 1) * self.edge_size] = list("X" * self.edge_size)
        return original_edge_dna_list

    def rand_edge(self, edge_k_idx: int) -> List[str]:
        """
        Randomise une arête (la remplace par une séquence ADN aléatoire).

        Args:
            edge_k_idx: Index global de l'arête à randomiser.

        Returns:
            La nouvelle séquence ADN aléatoire de l'arête.
        """
        if not (0 <= edge_k_idx < self.n * self.m):
            raise IndexError("Index d'arête hors limites.")
        
        # Génère une nouvelle séquence ADN aléatoire pour l'arête.
        new_edge_chars = random.choices(self.char_set, k=self.edge_size)
        self.set_edge(edge_k_idx, "".join(new_edge_chars))
        return new_edge_chars

    def mutate(self, distance: int = 10, dispersion: float = 0.2, high_mod: int = 0, prob_rm_char: Optional[float] = None) -> None:
        """
        Applique des mutations au génome.

        Args:
            distance: Nombre de mutations de caractères à appliquer par arête sélectionnée pour la mutation.
            dispersion: Fraction du nombre total d'arêtes qui subiront des mutations de caractères.
            high_mod: Nombre d'arêtes à "basculer" (activer une arête 'X' ou désactiver une arête existante).
            prob_rm_char: Probabilité qu'une mutation de caractère remplace le caractère par 'X'.
                           Si None, les mutations remplacent toujours par un caractère de `char_set`.
        """
        num_total_edges = self.n * self.m
        if num_total_edges == 0:
            return # Aucune mutation possible si pas d'arêtes

        # --- Mutation de caractères au sein des arêtes ---
        if distance > 0 and dispersion > 0:
            # Nombre d'arêtes à affecter par la mutation de caractères.
            num_edges_for_char_mutation = int(num_total_edges * dispersion)
            # Assure qu'au moins une arête est mutée si dispersion > 0 et qu'il y a des arêtes.
            if num_edges_for_char_mutation == 0 and dispersion > 0 and num_total_edges > 0:
                num_edges_for_char_mutation = 1
            
            num_edges_for_char_mutation = min(num_edges_for_char_mutation, num_total_edges)

            if num_edges_for_char_mutation > 0:
                # Sélectionne aléatoirement les arêtes à muter.
                edge_indices_to_mutate = random.sample(range(num_total_edges), k=num_edges_for_char_mutation)
                
                for edge_idx in edge_indices_to_mutate:
                    current_edge_chars = list(self.get_edge(edge_idx)) # Copie modifiable
                    for _ in range(distance): # Applique `distance` mutations à cette arête
                        if self.edge_size == 0: continue # Pas de mutation si l'arête est vide
                        pos_to_mutate = random.randint(0, self.edge_size - 1)
                        
                        new_char: str
                        if prob_rm_char is not None and random.random() < prob_rm_char:
                            new_char = "X" # Chance de remplacer par 'X'
                        else:
                            new_char = random.choice(self.char_set) # Mutation standard
                        current_edge_chars[pos_to_mutate] = new_char
                    self.set_edge(edge_idx, "".join(current_edge_chars))

        # --- Modification de haut niveau : activation/désactivation d'arêtes ---
        if high_mod > 0:
            # Nombre d'arêtes à basculer (activer ou désactiver).
            num_edges_to_toggle = min(high_mod, num_total_edges)
            if num_edges_to_toggle > 0:
                # Sélectionne aléatoirement les arêtes à basculer.
                edge_indices_to_toggle = random.sample(range(num_total_edges), k=num_edges_to_toggle)
                
                for edge_idx in edge_indices_to_toggle:
                    edge_content_str = "".join(self.get_edge(edge_idx))
                    is_missing = all(c == 'X' for c in edge_content_str)
                    if is_missing:
                        self.rand_edge(edge_idx) # Active une arête manquante en la randomisant
                    else:
                        self.rm_edge(edge_idx) # Désactive une arête existante en la remplaçant par 'X'


    def crossover(self, other: GenomeType) -> GenomeType:
        """
        Effectue un croisement (crossover) entre ce génome et un autre.
        Si les caractères à une position diffèrent entre les deux parents,
        le caractère résultant dans le génome enfant est 'X'. Sinon, il est identique.

        Args:
            other: L'autre génome parent.

        Returns:
            Un nouveau génome résultant du croisement.
        """
        if not (self.n == other.n and self.m == other.m and self.edge_size == other.edge_size):
             raise ValueError(
                 "Les génomes doivent avoir les mêmes dimensions (n, m, precision) pour le croisement."
                )
        
        # Crée la liste de caractères du nouveau génome.
        # Si les caractères à la même position sont identiques, on le garde.
        # S'ils diffèrent, on met 'X'.
        new_genome_chars = [
            s_char if s_char == o_char else 'X'
            for s_char, o_char in zip(self.genome, other.genome)
        ]
        return self.__class__(self.n, self.m, "".join(new_genome_chars), precision=self.edge_size)
    
    def distance(self, other: GenomeType) -> float:
        """
        Calcule la distance (euclidienne) entre ce génome et un autre.
        La distance est basée sur les valeurs numériques des caractères ADN (via `table`).

        Args:
            other: L'autre génome.

        Returns:
            La distance calculée.
        """
        if not (self.n == other.n and self.m == other.m and self.edge_size == other.edge_size):
             raise ValueError(
                 "Les génomes doivent avoir les mêmes dimensions (n, m, precision) pour le calcul de distance."
                )

        # Calcule la somme des carrés des différences entre les valeurs des caractères correspondants.
        # Utilise 0.0 pour les caractères non trouvés dans `table` (comme 'X').
        dist_sq = sum(
            (table.get(s_char, 0.0) - table.get(o_char, 0.0))**2
            for s_char, o_char in zip(self.genome, other.genome)
        )
        return dist_sq**0.5 # Racine carrée de la somme des carrés

    def merge(self, other: GenomeType) -> GenomeType:
        """
        Fusionne ce génome avec un autre.
        1. Effectue d'abord un `crossover`.
        2. Pour les arêtes entièrement 'X' dans le résultat du crossover, choisit aléatoirement
           l'arête correspondante de l'un des deux parents.
        3. Pour les arêtes partiellement 'X' (mixtes), remplit les 'X' en choisissant aléatoirement
           le caractère du parent correspondant à cette position.

        Args:
            other: L'autre génome parent.

        Returns:
            Un nouveau génome résultant de la fusion.
        """
        if not (self.n == other.n and self.m == other.m and self.edge_size == other.edge_size):
             raise ValueError("Les génomes doivent avoir les mêmes dimensions (n, m, precision) pour la fusion.")

        # Étape 1: Croisement initial
        crossed_genome = self.crossover(other)
        
        num_total_edges = self.n * self.m
        for edge_k_idx in range(num_total_edges):
            crossed_edge_chars_list = crossed_genome.get_edge(edge_k_idx) # Liste de caractères
            
            # Étape 2: Gérer les arêtes entièrement 'X'
            if all(c == 'X' for c in crossed_edge_chars_list):
                # Choisit aléatoirement l'arête de l'un des parents.
                if random.random() < 0.5:
                    parent_edge_str = "".join(self.get_edge(edge_k_idx))
                else:
                    parent_edge_str = "".join(other.get_edge(edge_k_idx))
                crossed_genome.set_edge(edge_k_idx, parent_edge_str)
            else:
                # Étape 3: Gérer les arêtes partiellement 'X' (mixtes)
                final_edge_chars_list = list(crossed_edge_chars_list) # Copie modifiable
                modified_in_step3 = False
                for char_idx in range(self.edge_size):
                    if final_edge_chars_list[char_idx] == 'X':
                        # Remplit le 'X' avec le caractère d'un des parents, choisi aléatoirement.
                        # Accès direct à self.genome et other.genome pour le caractère spécifique.
                        genome_char_idx = edge_k_idx * self.edge_size + char_idx
                        if random.random() < 0.5:
                            final_edge_chars_list[char_idx] = self.genome[genome_char_idx]
                        else:
                            final_edge_chars_list[char_idx] = other.genome[genome_char_idx]
                        modified_in_step3 = True
                
                if modified_in_step3:
                    crossed_genome.set_edge(edge_k_idx, "".join(final_edge_chars_list))
        return crossed_genome


    def to_binary(self) -> str:
        """
        Convertit le génome en une chaîne binaire.
        Chaque arête est d'abord convertie en float32 (via `dna_to_f32`),
        puis ce float32 est empaqueté en une séquence binaire de 32 bits.
        Toutes les séquences binaires des arêtes sont concaténées.

        Returns:
            Une chaîne de '0' et '1' représentant le génome.
        """
        binary_chunks: List[str] = []
        # Itère sur chaque arête (identifiée par l'index du nœud i et l'index de l'arête j dans ce nœud)
        for i, j in itertools.product(range(self.n), range(self.m)):
            edge_k_idx = i * self.m + j # Index global de l'arête
            edge_dna_list = self.get_edge(edge_k_idx)
            edge_dna_str = "".join(edge_dna_list)
            
            # Convertit la séquence ADN de l'arête en une valeur flottante.
            f32_val = dna_to_f32(edge_dna_str)
            
            # Empaquète le flottant en une séquence de 4 octets (big-endian).
            packed_float_bytes: bytes = struct.pack('>f', f32_val) # '>f' pour float32 big-endian
            
            # Convertit chaque octet en sa représentation binaire de 8 bits.
            edge_binary_str: str = "".join(format(byte_val, '08b') for byte_val in packed_float_bytes)
            binary_chunks.append(edge_binary_str)
                
        return "".join(binary_chunks) # Concatène toutes les chaînes binaires des arêtes.
    
    def to_string(self) -> str:
        """
        Convertit le génome en une chaîne de caractères.
        Chaque arête est convertie en une chaîne ADN, puis toutes les chaînes sont concaténées.

        Returns:
            La chaîne ADN complète du génome.
        """
        return "".join(self.genome)

    @staticmethod
    def from_binary(binary_str: str, n: int, m: int, precision: int = 16) -> GenomeType:
        """
        Crée un génome à partir d'une chaîne binaire.
        La chaîne binaire est découpée en segments de 32 bits, chacun représentant un float32.
        Chaque float32 est dépaqueté, puis converti en une séquence ADN (via `f32_to_dna`).
        Ces séquences ADN sont concaténées pour former le génome.

        Args:
            binary_str: La chaîne binaire ('0' et '1').
            n: Nombre de nœuds pour le nouveau génome.
            m: Nombre d'arêtes par nœud.
            precision: Nombre de caractères ADN par arête (taille de l'arête).

        Returns:
            Une nouvelle instance de Genome.
        """
        BITS_PER_FLOAT_ENCODING = 32 # Chaque arête est encodée comme un float32 (32 bits).
        num_edges = n * m
        expected_binary_len = num_edges * BITS_PER_FLOAT_ENCODING

        if len(binary_str) != expected_binary_len:
            raise ValueError(
                f"La longueur de la chaîne binaire ({len(binary_str)}) ne correspond pas à la longueur attendue "
                f"({expected_binary_len}) pour {n} nœuds, {m} arêtes/nœud, et "
                f"{BITS_PER_FLOAT_ENCODING} bits/arête (f32)."
            )
        
        # Valide que la chaîne binaire ne contient que des '0' et des '1'.
        if not re.fullmatch(r"[01]+", binary_str):
            raise ValueError("La chaîne binaire ne peut contenir que '0' et '1'.")

        if precision < 0: # La précision (taille de l'arête ADN) ne peut être négative.
            raise ValueError("La précision (edge_size) ne peut pas être négative.")

        all_dna_chars_list: List[str] = [] # Pour stocker tous les caractères ADN du génome reconstruit.

        for i in range(num_edges): # Pour chaque arête encodée dans la chaîne binaire
            # Extrait le segment binaire de 32 bits pour l'arête actuelle.
            start_idx = i * BITS_PER_FLOAT_ENCODING
            end_idx = start_idx + BITS_PER_FLOAT_ENCODING
            bit_chunk_str = binary_str[start_idx:end_idx]
            
            # Convertit le segment binaire en une séquence d'octets.
            byte_sequence = bytearray()
            for k in range(0, BITS_PER_FLOAT_ENCODING, 8): # Par blocs de 8 bits (1 octet)
                byte_str = bit_chunk_str[k:k+8]
                byte_sequence.append(int(byte_str, 2)) # Convertit la chaîne binaire de l'octet en entier
            
            # Dépaquète la séquence d'octets en un float32 (big-endian).
            f32_val: float = struct.unpack('>f', byte_sequence)[0]

            # Convertit la valeur flottante en une séquence ADN de la `precision` spécifiée.
            dna_edge_segment_str: str = f32_to_dna(f32_val, size=precision)
            
            all_dna_chars_list.extend(list(dna_edge_segment_str)) # Ajoute les caractères ADN de l'arête
            
        final_dna_string = "".join(all_dna_chars_list)
        # Crée et retourne une nouvelle instance de Genome avec la chaîne ADN reconstruite.
        # Utilise `Genome` directement au lieu de `cls` car c'est une staticmethod.
        return Genome(n, m, final_dna_string, precision=precision)

if __name__ == "__main__":
    # Importation nécessaire uniquement pour le bloc de test.
    # from brain import AntBrain # Supposons que cela n'est pas crucial pour le test de Genome seul.
    # import numpy as np # Supposons que cela n'est pas crucial pour le test de Genome seul.

    precision_test = 4 # Précision (taille d'arête) pour les tests
    genome_test = Genome.rand(2, 2, precision=precision_test) 
    print(f"Génome (n={genome_test.n}, m={genome_test.m}, precision={genome_test.edge_size}):")
    print("Chaîne du génome initial:", str(genome_test))
    
    aretes_initiales_str = [ "".join(genome_test.get_edge(i)) for i in range(genome_test.n * genome_test.m)]
    print("Arêtes initiales (str):", aretes_initiales_str)
    flottants_initiaux = [dna_to_f32("".join(genome_test.get_edge(i))) for i in range(genome_test.n * genome_test.m)]
    print("Flottants f32 initiaux:", flottants_initiaux)

    print("\nTest de to_binary et from_binary:")
    representation_binaire = genome_test.to_binary()
    print(f"Représentation binaire (longueur {len(representation_binaire)}): {representation_binaire[:64]}...")
    
    genome_reconstruit = Genome.from_binary(representation_binaire, genome_test.n, genome_test.m, precision=genome_test.edge_size)
    print("Chaîne du génome reconstruit:", str(genome_reconstruit))
    aretes_reconstruites_str = ["".join(genome_reconstruit.get_edge(i)) for i in range(genome_reconstruit.n * genome_reconstruit.m)]
    print("Arêtes reconstruites (str):", aretes_reconstruites_str)
    flottants_reconstruits = [dna_to_f32("".join(genome_reconstruit.get_edge(i))) for i in range(genome_reconstruit.n * genome_reconstruit.m)]
    print("Flottants f32 reconstruits:", flottants_reconstruits)

    # La conversion f32 <-> ADN n'est pas toujours parfaitement réversible en raison de la nature de f32_to_dna.
    # Donc, la chaîne ADN peut différer, mais les valeurs flottantes devraient être proches si f32_to_dna est stable.
    if str(genome_test) == str(genome_reconstruit):
        print("SUCCÈS : La chaîne du génome est parfaitement reconstruite.")
    else:
        print("NOTE : La chaîne du génome diffère après la conversion binaire (ce qui peut être attendu).")
        print("Original :     ", str(genome_test))
        print("Reconstruit :", str(genome_reconstruit))

    # Comparaison des flottants est plus pertinente ici.
    # En raison des imprécisions de la conversion float -> DNA -> float, une égalité exacte n'est pas garantie.
    # Une comparaison avec une tolérance serait plus robuste, mais pour ce test, on vérifie l'égalité.
    if flottants_initiaux == flottants_reconstruits:
        print("SUCCÈS : Les valeurs f32 sont parfaitement reconstruites.")
    else:
        print("NOTE : Les valeurs f32 diffèrent après la conversion binaire (ce qui peut être attendu).")
        print("Flottants initiaux:    ", flottants_initiaux)
        print("Flottants reconstruits:", flottants_reconstruits)


    print("\nTest de la mutation:")
    genome_mute = Genome.rand(2, 2, precision=precision_test)
    print("Génome avant mutation:", str(genome_mute))
    genome_mute.mutate(distance=2, dispersion=0.5, high_mod=1, prob_rm_char=0.1)
    print("Chaîne du génome muté:", str(genome_mute))
    flottants_mutes = [dna_to_f32("".join(genome_mute.get_edge(i))) for i in range(genome_mute.n * genome_mute.m)]
    print("Flottants f32 mutés:", flottants_mutes)

    print("\nTest du croisement et de la fusion:")
    genome1 = Genome.rand(2,2, precision=precision_test)
    genome2 = Genome.rand(2,2, precision=precision_test)
    print("Génome 1:", str(genome1))
    print("Génome 2:", str(genome2))

    genome_croise = genome1.crossover(genome2)
    print("G1 croisé avec G2:", str(genome_croise))
    
    genome_fusionne = genome1.merge(genome2)
    print("G1 fusionné avec G2:", str(genome_fusionne))

    print("\nDistance G1 à G2:", genome1.distance(genome2))
    print("Distance G1 à Fusionné:", genome1.distance(genome_fusionne))
    print("Distance G2 à Fusionné:", genome2.distance(genome_fusionne))