import numpy as np  
import random
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from itertools import permutations, combinations
import time
from torch.nn.utils.rnn import pad_sequence
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'



MODE = "both"  

# Define the mapping from shot to index
shot_type_dict = {'ELS':0,'LS':1,'MS':2,'CU':3,'ECU':4}
# shot_type_dict = {'Extreme Long Shot':0,'Long Shot':1,'Medium shot':2,'Close-up':3,'Extreme close-up':4}
inverse_shot_type_dict = {v:k for k,v in shot_type_dict.items()}

shot_type_to_index_gan = {
    "远景": 0,
    "全景": 1,
    "中景": 2,
    "近景": 3,
    "特写": 4
}

# Map the lens type to the number
movement_type_to_index_gan = {
    "stable": "STATIC",
    "tilt_up":'TILT-U',
    "tilt_down":'TILT-D',
    "PAN-L":"PAN-L",
    "PAN-R":"PAN-R",
    "zoom_in":"ZOOM-I",
    "zoom_out":"ZOOM-O"
}


padding_idx_shot = len(shot_type_dict)
num_shot_types = len(shot_type_dict)  

def shot_type_to_index(shot_type):
    return shot_type_to_index_gan.get(shot_type, padding_idx_shot)

def index_to_shot_type(index):
    return inverse_shot_type_dict.get(index, 'UNKNOWN')


# Define the mapping of lens attributes to indexes
camera_movement_dict = {
    'STATIC': 0,
    'TILT-U': 1,
    'TILT-D': 2,
    'PAN-L': 3,
    'PAN-R': 4,
    'ZOOM-I': 5,
    'ZOOM-O': 6
}
inverse_camera_movement_dict = {v:k for k,v in camera_movement_dict.items()}
padding_idx_movement = len(camera_movement_dict)
num_camera_movements = len(camera_movement_dict)

def camera_movement_to_index(movement_type):
    return camera_movement_dict.get(movement_type, padding_idx_movement)

def index_to_camera_movement(index):
    return inverse_camera_movement_dict.get(index, 'UNKNOWN')



# Define the shot grammar score matrix
shot_type_grammar_score_matrix = np.array([
    [0,   0.5, 1,   0,   0],
    [0.5, 0.6, 1,   1,   0],
    [1,   1,   1,   1,   1],
    [0,   0.6, 0.8, 0.6, 1],
    [0,   0,   0.3, 1,   0],
])


# Define the syntax score matrix for the lens
camera_movement_grammar_score_matrix = np.array([
    [1,   1, 1,   1,   1, 1, 1],
    [1,   1, 0,   0,   0, 1, 1],
    [1,   0, 1,   0,   0, 1, 1],
    [1,   0, 0,   1,   0, 1, 1],
    [1,   0, 0,   0,   1, 1, 1],
    [1,   0, 0,   0,   0, 1, 0],
    [1,   0, 0,   0,   0, 0, 1],
])


# Define the shot grammar score calculation function
def compute_shot_type_grammar_score(sequence_shots):
    padding_idx = padding_idx_shot
    score = 0.0
    for i in range(len(sequence_shots) - 1):
        current_shot = sequence_shots[i]
        next_shot = sequence_shots[i + 1]
        idx_current = shot_type_to_index(current_shot)
        idx_next = shot_type_to_index(next_shot)
        if idx_current == padding_idx or idx_next == padding_idx:
            raise ValueError(f"Unknown movement type: {current_shot} or {next_shot}")
        score += shot_type_grammar_score_matrix[idx_current, idx_next]
    return score  # Return the cumulative total score


# Define the syntax score calculation function for the camera
def compute_camera_movement_grammar_score(sequence_movements):
    padding_idx = padding_idx_movement
    score = 0.0
    for i in range(len(sequence_movements) - 1):
        current_movement = sequence_movements[i]
        next_movement = sequence_movements[i + 1]
        idx_current = camera_movement_to_index(current_movement)
        idx_next = camera_movement_to_index(next_movement)
        if idx_current == padding_idx or idx_next == padding_idx:
            raise ValueError(f"Unknown movement type: {current_movement} or {next_movement}")
        score += camera_movement_grammar_score_matrix[idx_current, idx_next]
    return score  # Return the cumulative total score



# Define the dataset class
class SequenceDatasetShot(Dataset):
    def __init__(self, sequences_shot):
        self.sequences_shot = sequences_shot
    
    def __len__(self):
        return len(self.sequences_shot)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences_shot[idx], dtype=torch.long)

class SequenceDatasetMovement(Dataset):
    def __init__(self, sequences_movement):
        self.sequences_movement = sequences_movement
    
    def __len__(self):
        return len(self.sequences_movement)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences_movement[idx], dtype=torch.long)


# Define the collate_fn for padding and masking
def collate_fn_shot(batch):
    """
    Fill the shot sequences in the batch to the same length and generate a mask.
    """
    batch = [seq.clone().detach().long() if isinstance(seq, torch.Tensor) else torch.tensor(seq, dtype=torch.long) for seq in batch]
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=padding_idx_shot)  # padding 
    masks = (padded_batch != padding_idx_shot)  # The non-filled portion is True
    return padded_batch, masks

def collate_fn_movement(batch):
    """
    Fill the shot sequences in the batch to the same length and generate a mask.
    """
    batch = [seq.clone().detach().long() if isinstance(seq, torch.Tensor) else torch.tensor(seq, dtype=torch.long) for seq in batch]
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=padding_idx_movement)  # padding 
    masks = (padded_batch != padding_idx_movement)  # The non-filled portion is True
    return padded_batch, masks


# Define the traditional TransitionEnergyModel
class TransitionEnergyModel(nn.Module):
    def __init__(self, num_types):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(num_types, num_types))
        nn.init.xavier_uniform_(self.W)


    def forward(self, sequence, padding_idx):
        """
        Calculate the energy value of the sequence.
        :param sequence: 1D tensor, Index by type (can be shot size or camera movement).
        :param padding_idx: index of padding.
        :return: The scalar tensor, which represents the energy of the sequence.
        """
        device = sequence.device

        if len(sequence) < 2:
            return torch.tensor(0.0, device=device)

        idx_current = sequence[:-1]
        idx_next = sequence[1:]
        
        mask = (idx_current != padding_idx) & (idx_next != padding_idx)
        idx_current = idx_current[mask]
        idx_next = idx_next[mask]
        if len(idx_current) == 0:
            return torch.tensor(0.0, device=device)

        score = torch.sum(self.W[idx_current, idx_next])

        energy = -score

        return energy

# ================================
# The TransitionOnlyModel is used to train the TransitionEnergyModel separately.
# ================================
class TransitionOnlyModel(nn.Module):
    def __init__(self, num_shot_types, num_camera_movements):
        super().__init__()
        self.transition_model_shot = TransitionEnergyModel(num_shot_types)
        self.transition_model_movement = TransitionEnergyModel(num_camera_movements)
    
    def forward_shot(self, sequence_shot, mask_batch):
        padding_idx_temp = padding_idx_shot
        transition_energy = []
        for i in range(sequence_shot.size(0)):
            seq = sequence_shot[i]
            mask = mask_batch[i]
            valid_seq = seq[mask]
            if valid_seq.numel() < 2:
                energy = torch.tensor(0.0, device=seq.device)
            else:
                energy = self.transition_model_shot(valid_seq, padding_idx_temp)
            transition_energy.append(energy)
        transition_energy = torch.stack(transition_energy)
        return transition_energy
    
    def forward_movement(self, sequence_movement, mask_batch):
        padding_idx_temp = padding_idx_movement
        transition_energy = []
        for i in range(sequence_movement.size(0)):
            seq = sequence_movement[i]
            mask = mask_batch[i]
            valid_seq = seq[mask]
            if valid_seq.numel() < 2:
                energy = torch.tensor(0.0, device=seq.device)
            else:
                energy = self.transition_model_movement(valid_seq, padding_idx_temp)
            transition_energy.append(energy)
        transition_energy = torch.stack(transition_energy)
        return transition_energy



# ================================
# Definition of the optimization algorithm: Genetic Algorithm + Langevin Optimization
# ================================
def langevin_with_genetic_optimization(video_clips, num_select,
                                       shot_type_model, movement_model,
                                       device,
                                       population_size=100, num_iterations=100,
                                       epsilon=0.5, temperature=1.0,
                                       crossover_rate=0.9, mutation_rate=0.1,
                                       target_best_sequences_count=10,
                                       include_shot_type_nn=True,
                                       include_camera_movement_nn=True,
                                       include_shot_type_grammar=False,
                                       include_camera_movement_grammar=False,
                                       weight_shot_type_nn=1.0,
                                       weight_camera_movement_nn=1.0,
                                       weight_shot_type_grammar=1.0,
                                       weight_camera_movement_grammar=1.0,
                                       max_score=1.0,
                                       text_list=None):
    num_clips = len(video_clips)
    clip_indices = list(range(num_clips))

    population = [random.sample(clip_indices, num_select) for _ in range(population_size)]

    best_sequences = []
    best_sequences_set = set()
    best_energy = float('inf')
    max_energy_score = float('-inf')

    for iteration in tqdm(range(num_iterations), desc="Langevin Optimization with Genetic Elements"):
        new_population = []
        
        for individual in population:
            current_energy = energy_function(device=device, sequence_indices=individual, video_clips=video_clips,
                                            include_shot_type_nn=include_shot_type_nn,
                                            include_camera_movement_nn=include_camera_movement_nn,
                                            include_shot_type_grammar=include_shot_type_grammar,
                                            include_camera_movement_grammar=include_camera_movement_grammar,
                                            shot_type_model=shot_type_model,
                                            camera_movement_model=movement_model,
                                            weight_shot_type_nn=weight_shot_type_nn,
                                            weight_camera_movement_nn=weight_camera_movement_nn,
                                            weight_shot_type_grammar=weight_shot_type_grammar,
                                            weight_camera_movement_grammar=weight_camera_movement_grammar,
                                            max_score=max_score,
                                            text_list=text_list
                                            )
            
            neighbor_sequences = []
            for _ in range(5):
                new_seq = individual.copy()
                operation = random.choice(['swap', 'replace'])
                if operation == 'swap' and len(new_seq) >= 2:
                    idx1, idx2 = random.sample(range(num_select), 2)
                    new_seq[idx1], new_seq[idx2] = new_seq[idx2], new_seq[idx1]
                elif operation == 'replace':
                    idx_to_replace = random.randint(0, num_select - 1)
                    available = list(set(clip_indices) - set(new_seq))
                    if available:
                        new_seq[idx_to_replace] = random.choice(available)
                neighbor_sequences.append(new_seq)
            
            neighbor_energies = [energy_function(device=device, sequence_indices=seq, video_clips=video_clips,
                                               include_shot_type_nn=include_shot_type_nn,
                                               include_camera_movement_nn=include_camera_movement_nn,
                                               include_shot_type_grammar=include_shot_type_grammar,
                                               include_camera_movement_grammar=include_camera_movement_grammar,
                                               shot_type_model=shot_type_model,
                                               camera_movement_model=movement_model,
                                               weight_shot_type_nn=weight_shot_type_nn,
                                               weight_camera_movement_nn=weight_camera_movement_nn,
                                               weight_shot_type_grammar=weight_shot_type_grammar,
                                               weight_camera_movement_grammar=weight_camera_movement_grammar,
                                               max_score=max_score,
                                               text_list=text_list
                                               ) for seq in neighbor_sequences]
            
            min_energy = min(neighbor_energies)
            min_idx = neighbor_energies.index(min_energy)
            best_neighbor = neighbor_sequences[min_idx]
            
            delta_energy = min_energy - current_energy
            
            acceptance_prob = min(1, math.exp(-delta_energy / (epsilon * temperature)))
            if delta_energy < 0 or random.random() < acceptance_prob:
                updated_sequence = best_neighbor
                updated_energy = min_energy
            else:
                updated_sequence = individual
                updated_energy = current_energy
            new_population.append(updated_sequence)
            
            if updated_energy < best_energy:
                best_energy = updated_energy
                max_energy_score = -best_energy
                best_sequences = [(updated_sequence.copy(), max_energy_score)]
                best_sequences_set = {tuple(updated_sequence)}
            elif updated_energy == best_energy:
                seq_tuple = tuple(updated_sequence)
                if seq_tuple not in best_sequences_set:
                    best_sequences.append((updated_sequence.copy(), -updated_energy))
                    best_sequences_set.add(seq_tuple)

        fitness_values = []
        for ind in new_population:
            fitness = -energy_function(device=device, sequence_indices=ind, video_clips=video_clips,
                                      include_shot_type_nn=include_shot_type_nn,
                                      include_camera_movement_nn=include_camera_movement_nn,
                                      include_shot_type_grammar=include_shot_type_grammar,
                                      include_camera_movement_grammar=include_camera_movement_grammar,
                                      shot_type_model=shot_type_model,
                                      camera_movement_model=movement_model,
                                      weight_shot_type_nn=weight_shot_type_nn,
                                      weight_camera_movement_nn=weight_camera_movement_nn,
                                      weight_shot_type_grammar=weight_shot_type_grammar,
                                      weight_camera_movement_grammar=weight_camera_movement_grammar,
                                      max_score=max_score,
                                      text_list=text_list
                                      )
            fitness_values.append(fitness)

        fitness_array = np.array(fitness_values)
        
        exp_fitness = np.exp(fitness_array - np.max(fitness_array))
        selection_prob = exp_fitness / np.sum(exp_fitness)

        selected_indices = np.random.choice(len(new_population), size=population_size, p=selection_prob)
        selected_population = [new_population[idx] for idx in selected_indices]

        population = []
        while len(population) < population_size:
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            
            if random.random() < crossover_rate:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            if random.random() < mutation_rate:
                mutate(child1, clip_indices)
            if random.random() < mutation_rate:
                mutate(child2, clip_indices)
            
            population.append(child1)
            if len(population) < population_size:
                population.append(child2)
        
        if len(best_sequences) >= target_best_sequences_count:
            print("\nA sufficient number of top-scoring sequences has been found; exiting early.")
            break
    total_iterations = iteration + 1
    return best_sequences, max_energy_score, total_iterations

# ================================
# Define crossover and mutation functions.
# ================================
def order_crossover(parent1, parent2):
    """Order Crossover (OX) """
    size = len(parent1)
    child1 = [None] * size
    child2 = [None] * size

    idx1, idx2 = sorted(random.sample(range(size), 2))
    
    child1[idx1:idx2] = parent1[idx1:idx2]
    child2[idx1:idx2] = parent2[idx1:idx2]

    fill_genes(child1, parent2, idx2, size)
    fill_genes(child2, parent1, idx2, size)

    return child1, child2

def fill_genes(child, parent, start, size):
    """Helper function for filling the remaining genes of the offspring"""
    idx = start
    parent_idx = start
    while None in child:
        gene = parent[parent_idx % size]
        if gene not in child:
            child[idx % size] = gene
            idx += 1
        parent_idx += 1

def mutate(individual, clip_indices):
    """Mutation operation: randomly swap two genes"""
    if len(individual) < 2:
        return
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

# ================================
# Calculate the similarity between video and text
# ================================
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from align import AltCLIPSimilarityCalculator
optimized_model = AltCLIPSimilarityCalculator()

# Text list (script)
# TEXT = [
#
#     "Here is a glimpse into my day.",
#     "It started with a quiet morning at home.",
#     "I made myself breakfast, ",
#     "and good orange juice.",
#     "I decided to go for a drive, ",
#     "and the scenery was just beautiful.",
#     "I even had a little treat, showing off my new nails.",
#     "It was one of those days that just feels really good."
#
# ]

# ================================
# Define the energy function
# ================================
global ans
ans = []
def energy_function(device, sequence_indices, video_clips,
                    include_shot_type_nn,
                    include_camera_movement_nn,
                    include_shot_type_grammar,
                    include_camera_movement_grammar,
                    shot_type_model,
                    camera_movement_model,
                    weight_shot_type_nn,
                    weight_camera_movement_nn,
                    weight_shot_type_grammar,
                    weight_camera_movement_grammar,
                    max_score,
                    weight_semantic_similarity=50.0,
                    text_list=None):  # Semantic similarity weight parameter


    sequence_shots = [video_clips[i]['type_shot'] for i in sequence_indices]
    sequence_movements = [video_clips[i]['type_movement'] for i in sequence_indices]
    sequence_name = [video_clips[i]['name'] for i in sequence_indices]

    video_paths = []
    base_video_path = "app/lang/dataset/candidate_video/"
    for name in sequence_name:
        video_paths.append(os.path.join(base_video_path, f"{name}.mp4"))

    if video_paths and text_list:
        similarity_score = optimized_model.calculate_similarity_optimized(video_paths, text_list) # Calculate semantic matching
    else:
        print("No video clips or text list to concatenate.")
        similarity_score = 0.0

    total_energy = []

    # Semantic component
    total_energy.append(similarity_score * weight_semantic_similarity * (-50))

    # Scenographic section
    if include_shot_type_nn and shot_type_model is not None:
        
        sequence_indices_mapped_shot = [shot_type_to_index(shot) for shot in sequence_shots]
        input_seq_shot = torch.tensor(sequence_indices_mapped_shot, dtype=torch.long).unsqueeze(0).to(device)  
        shot_mask = torch.ones_like(input_seq_shot, dtype=torch.bool).to(device)  
        
        energy_shot = shot_type_model.forward_shot(input_seq_shot, shot_mask)
        if energy_shot.numel() == 1:
            shot_type_score_nn = -energy_shot.squeeze().item()  
        else:
            shot_type_score_nn = -energy_shot.mean().item()  
        shot_type_score_nn *= max_score
        total_energy.append(weight_shot_type_nn * shot_type_score_nn * (-1))

    # Mirror section
    if include_camera_movement_nn and camera_movement_model is not None:
        
        sequence_indices_mapped_movement = [camera_movement_to_index(mov) for mov in sequence_movements]
        input_seq_movement = torch.tensor(sequence_indices_mapped_movement, dtype=torch.long).unsqueeze(0).to(device) 
        movement_mask = torch.ones_like(input_seq_movement, dtype=torch.bool).to(device) 
        
        energy_movement = camera_movement_model.forward_movement(input_seq_movement, movement_mask)
        if energy_movement.numel() == 1:
            camera_movement_score_nn = -energy_movement.squeeze().item()  
        else:
            camera_movement_score_nn = -energy_movement.mean().item()
        camera_movement_score_nn *= max_score
        total_energy.append(weight_camera_movement_nn * camera_movement_score_nn * (-1))

    total_energy_array = np.array(total_energy)
    ans.append(total_energy_array)

    return np.mean(total_energy_array)
    


def plot_transition_heatmap(shot_type_grammar_score_matrix, learned_W_shot, 
                           camera_movement_grammar_score_matrix, learned_W_movement,
                           title1="Original Grammar Matrix (Shot Types)", 
                           title2="Learned Transition Matrix W (Shot Types)",
                           title3="Original Grammar Matrix (Camera Movements)",
                           title4="Learned Transition Matrix W (Camera Movements)",
                           filename_prefix='transition_matrices_comparison'):
    W_shot_normalized = learned_W_shot
    grammar_shot_normalized = shot_type_grammar_score_matrix
    W_movement_normalized = learned_W_movement
    grammar_movement_normalized = camera_movement_grammar_score_matrix
    
    fig, axes = plt.subplots(2, 2, figsize=(24, 12))
    
    im1 = axes[0, 0].imshow(grammar_shot_normalized, cmap='viridis', aspect='auto')
    axes[0, 0].set_title(title1)
    axes[0, 0].set_xticks(range(num_shot_types))
    axes[0, 0].set_yticks(range(num_shot_types))
    axes[0, 0].set_xticklabels(list(shot_type_dict.keys()))
    axes[0, 0].set_yticklabels(list(shot_type_dict.keys()))
    plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(im1, ax=axes[0, 0])

    
    im2 = axes[0, 1].imshow(W_shot_normalized, cmap='viridis', aspect='auto')
    axes[0, 1].set_title(title2)
    axes[0, 1].set_xticks(range(num_shot_types))
    axes[0, 1].set_yticks(range(num_shot_types))
    axes[0, 1].set_xticklabels(list(shot_type_dict.keys()))
    axes[0, 1].set_yticklabels(list(shot_type_dict.keys()))
    plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(im2, ax=axes[0, 1])

    
    im3 = axes[1, 0].imshow(grammar_movement_normalized, cmap='viridis', aspect='auto')
    axes[1, 0].set_title(title3)
    axes[1, 0].set_xticks(range(num_camera_movements))
    axes[1, 0].set_yticks(range(num_camera_movements))
    axes[1, 0].set_xticklabels(list(camera_movement_dict.keys()))
    axes[1, 0].set_yticklabels(list(camera_movement_dict.keys()))
    plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(im3, ax=axes[1, 0])

    
    im4 = axes[1, 1].imshow(W_movement_normalized, cmap='viridis', aspect='auto')
    axes[1, 1].set_title(title4)
    axes[1, 1].set_xticks(range(num_camera_movements))
    axes[1, 1].set_yticks(range(num_camera_movements))
    axes[1, 1].set_xticklabels(list(camera_movement_dict.keys()))
    axes[1, 1].set_yticklabels(list(camera_movement_dict.keys()))
    plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(f'{filename_prefix}.png', dpi=300)
    plt.show()

# ================================
# Define the main function
# ================================
def langevin_(num_select=8, num_candidate_video=34, text_list=None):
    if text_list is None:
        text_list = []
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    angle_type,method_type,view_type,mv_type = read_json_files()


    # ================================
    # Statistical transfer matrix (no training, just statistics)
    # ================================
    def compute_transition_prob_matrix(labels, label_to_index_map):
        num_types_local = len(label_to_index_map)
        counts = np.zeros((num_types_local, num_types_local), dtype=float)
        for i in range(len(labels) - 1):
            a = labels[i]
            b = labels[i + 1]
            if a in label_to_index_map and b in label_to_index_map:
                counts[label_to_index_map[a], label_to_index_map[b]] += 1.0
        total = counts.sum()
        return counts / total if total > 0 else counts


    W_shot_stats = compute_transition_prob_matrix(view_type, shot_type_dict)
    W_movement_stats = compute_transition_prob_matrix(mv_type, camera_movement_dict)


    # Here, a TransitionOnlyModel model instance is created, but its core function is not "training," but rather as an "energy calculator."
    transition_only_model = TransitionOnlyModel(num_shot_types, num_camera_movements).to(device)
    with torch.no_grad():
        transition_only_model.transition_model_shot.W.copy_(torch.tensor(W_shot_stats, dtype=torch.float32, device=device))
        transition_only_model.transition_model_movement.W.copy_(torch.tensor(W_movement_stats, dtype=torch.float32, device=device))

    learned_W_shot = W_shot_stats
    learned_W_movement = W_movement_stats
    plot_transition_heatmap(
        shot_type_grammar_score_matrix, learned_W_shot if learned_W_shot is not None else np.zeros_like(shot_type_grammar_score_matrix),
        camera_movement_grammar_score_matrix, learned_W_movement if learned_W_movement is not None else np.zeros_like(camera_movement_grammar_score_matrix),
        title1="Original Grammar Matrix (Shot Types)",
        title2="Stat Transition Matrix W (Shot Types)",
        title3="Original Grammar Matrix (Camera Movements)",
        title4="Stat Transition Matrix W (Camera Movements)",
        filename_prefix='app/lang/dataset/over/png/ipartment_cooking_ours'
    )
    

    print("******************Statistics completed******************")


    try:
        with open('app/lang/dataset/candidate_video/view_type.json', 'r', encoding='utf-8') as f:
            view_data = json.load(f)
    except FileNotFoundError:
        print("Error: The view_type.json file could not be found. Please ensure that the file exists in the correct path.")
        return

    
    try:
        with open('app/lang/dataset/candidate_video/mv_type.json', 'r', encoding='utf-8') as f:
            mv_data = json.load(f)
    except FileNotFoundError:
        print("Error: The mv_type.json file could not be found. Please ensure that the file exists in the correct path.")
        return

    
    view_dict = {item['img']: item['result'][0]['cls_res']['view'][0][0] for item in view_data}
    mv_dict = {item['img']: item['camera_motion'][0]['motion'][0][0] for item in mv_data}

    theoretical_max_score = 1.0

    # Get the shot size and camera movement type from viewdictionary and mvdictionary according to the image name, and convert them into numbers
    # num_select = 8
    # num_candidate_video = 34
    candidate_shot_indices = [shot_type_to_index_gan.get(view_dict.get(f"segment_{i+1:03d}"), -1)
     for i in range(num_candidate_video)]
    candidate_movement_indices = [movement_type_to_index_gan.get(mv_dict.get(f"segment_{i+1:03d}"), -1) for i in range(num_candidate_video)]

    # Build the list of video clips
    video_clips = []
    for i in range(len(candidate_shot_indices)):
        shot_index = candidate_shot_indices[i]
        movement_index = candidate_movement_indices[i]
            
        shot_type = next((k for k, v in shot_type_to_index_gan.items() if v == shot_index), "未知景别")
        movement_type = next((v for k, v in movement_type_to_index_gan.items() if v == movement_index), "未知运镜")
            
        video_clips.append({
            'id': i, 
            'type_shot': shot_type, 
            'type_movement': movement_type,
            "name": f"segment_{i+1:03d}"
        })

    #========================================
    # Data read completed, start test
    # If the number of candidate fragments is small, you can use the exhaustive method to calculate the theoretical maximum score
    all_possible_scores = []
    all_sequences = []  
    sequences_for_max_score = []
    if len(video_clips) <= 10:
        
        all_combinations = combinations(range(len(video_clips)), num_select)
        for combination in all_combinations:
            
            all_permutations = permutations(combination)
            for perm in all_permutations:
                
                sequence_shots = [video_clips[i]['type_shot'] for i in perm]
                shot_score = compute_shot_type_grammar_score(sequence_shots) if MODE in ["shot", "both"] else 0.0
                
                
                sequence_movements = [video_clips[i]['type_movement'] for i in perm]
                movement_score = compute_camera_movement_grammar_score(sequence_movements) if MODE in ["movement", "both"] else 0.0
                
                
                total_score = shot_score + movement_score
                
                
                all_possible_scores.append(total_score)
                all_sequences.append((total_score, sequence_shots, sequence_movements))

        if not all_possible_scores:
            print(f"Test case : There is no valid sequence available for computation.")
            return

        # Calculate the actual theoretical maximum score
        actual_theoretical_max_score = max(all_possible_scores)

        # Find all the sequences that score the highest theoretical points
        sequences_for_max_score = [
            (sequence_shots, sequence_movements) for score, sequence_shots, sequence_movements in all_sequences 
            if score == actual_theoretical_max_score
        ]

        
        print(f"Test case: All sequences corresponding to the theoretical maximum score (shot type -> camera movement):")
        for seq_shot, seq_movement in sequences_for_max_score:
            print(f"{seq_shot} ; {seq_movement}")

    else:
        actual_theoretical_max_score = theoretical_max_score

    # ================================
    # Set weight parameters and select which scoring methods to use
    # ================================
    include_shot_type_nn = MODE in ["shot", "both"]
    include_camera_movement_nn = MODE in ["movement", "both"]  # Keep as True or adjust based on the mode
    include_shot_type_grammar = False  
    include_camera_movement_grammar = False  
    shot_type_model = transition_only_model if include_shot_type_nn else None  
    movement_model = transition_only_model if include_camera_movement_nn else None  

    weight_shot_type_nn = 1.0  # Adjust the weights to ensure the influence of the neural network score.
    weight_camera_movement_nn = 1.0
    weight_shot_type_grammar = 1.0
    weight_camera_movement_grammar = 1.0

    # ================================
    # Run optimization.
    # ================================
    print("\nStart Langevin optimization...")
    start_time = time.time()
    best_sequences, max_energy_score, total_iterations = langevin_with_genetic_optimization(
        video_clips=video_clips, num_select=num_select,
        shot_type_model=shot_type_model if include_shot_type_nn else None,
        movement_model=movement_model if include_camera_movement_nn else None,
        device=device,
        population_size=100, num_iterations=100,
        epsilon=0.5, temperature=1.0,
        crossover_rate=0.9, mutation_rate=0.1,
        target_best_sequences_count=10,
        include_shot_type_nn=include_shot_type_nn,
        include_camera_movement_nn=include_camera_movement_nn,
        include_shot_type_grammar=include_shot_type_grammar,
        include_camera_movement_grammar=include_camera_movement_grammar,
        weight_shot_type_nn=weight_shot_type_nn,
        weight_camera_movement_nn=weight_camera_movement_nn,
        weight_shot_type_grammar=weight_shot_type_grammar,
        weight_camera_movement_grammar=weight_camera_movement_grammar,
        max_score=1.0,
        text_list=text_list
    )
    end_time = time.time()

    
    execution_time = end_time - start_time
    print(f"\nLangevin optimization runtime: {execution_time:.2f} sec")
    print(f"Maximum energy score: {max_energy_score:.4f}")
    print(f"Total number of iterations: {total_iterations}")


    best_sequences_data = {
        "execution_time": execution_time,
        "max_energy_score": max_energy_score,
        "total_iterations": total_iterations,
        "sequences": []
    }

    
    print("\n" + "="*80)
    print("Semantic similarity analysis of the best sequence.")
    print("="*80)
    
    for i, (sequence_indices, score) in enumerate(best_sequences):
        sequence_shots = [video_clips[idx]['type_shot'] for idx in sequence_indices]
        sequence_movements = [video_clips[idx]['type_movement'] for idx in sequence_indices]
        sequence_names = [video_clips[idx]['name'] for idx in sequence_indices]
        
        
        video_paths = []
        base_video_path = "app/lang/dataset/candidate_video/"
        for name in sequence_names:
            video_paths.append(os.path.join(base_video_path, f"{name}.mp4"))
        
        
        semantic_similarity = optimized_model.calculate_similarity_optimized(video_paths, text_list) if text_list else 0.0
        
        individual_similarities = []
        for j, (video_path, text) in enumerate(zip(video_paths, text_list)):
            individual_sim = optimized_model._compute_video_text_similarity(video_path, text)
            individual_similarities.append(individual_sim)
        
    
        print(f"\nSequence {i+1}:")
        print(f"  Total energy score: {score:.4f}")
        print(f"  Average semantic similarity: {semantic_similarity:.4f}")
        print(f"  Video sequence: {sequence_names}")
        print(f"  Shot sequence: {sequence_shots}")
        print(f"  Camera movement sequence.: {sequence_movements}")
        print(f"  Semantic similarity of each segment:")
        for j, (name, sim) in enumerate(zip(sequence_names, individual_similarities)):
            print(f"    {name}: {sim:.4f} (Corresponding text: {text_list[j][:30]}...)")
        
        sequence_data = {
            'id': i,
            'score': score,
            'semantic_similarity': semantic_similarity,
            'individual_similarities': individual_similarities,
            'indices': sequence_indices,
            'shot_types': sequence_shots,
            'movement_types': sequence_movements,
            'names': sequence_names
        }
        best_sequences_data["sequences"].append(sequence_data)
    
    # 计算并打印统计信息
    semantic_similarities = [seq_data['semantic_similarity'] for seq_data in best_sequences_data["sequences"]]
    avg_semantic_similarity = sum(semantic_similarities) / len(semantic_similarities)
    max_semantic_similarity = max(semantic_similarities)
    min_semantic_similarity = min(semantic_similarities)
    
    print("\n" + "-"*80)
    print("Semantic similarity statistics:")
    print(f"  Average semantic similarity: {avg_semantic_similarity:.4f}")
    print(f"  Highest semantic similarity: {max_semantic_similarity:.4f}")
    print(f"  Lowest semantic similarity: {min_semantic_similarity:.4f}")
    print(f"  Semantic similarity range: {max_semantic_similarity - min_semantic_similarity:.4f}")
    print("-"*80)
    
    
    sorted_sequences = sorted(enumerate(semantic_similarities), key=lambda x: x[1], reverse=True)
    print("\Semantic similarity ranking:")
    for rank, (seq_idx, sim_score) in enumerate(sorted_sequences, 1):
        seq_names = best_sequences_data["sequences"][seq_idx]['names']
        print(f"  No.{rank}: sequences{seq_idx+1}, similarity: {sim_score:.4f}, video: {seq_names}")
    
    
    best_sequences_data["semantic_similarity_stats"] = {
        "average_semantic_similarity": avg_semantic_similarity,
        "max_semantic_similarity": max_semantic_similarity,
        "min_semantic_similarity": min_semantic_similarity,
        "semantic_similarity_range": max_semantic_similarity - min_semantic_similarity
    }
    
    print("\n" + "="*80)

    from datetime import datetime
    
    now = datetime.now()
    formatted_time = now.strftime("%m%d_%H_%M")
    
    output_file = f"app/lang/code/result/best_sequences_{formatted_time}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(best_sequences_data, f, ensure_ascii=False, indent=4)

    print(f"Best sequence has been saved to: {output_file}")
    print(f"A total of {len(best_sequences)} best sequences were saved.")

    view_gan(sequence_names)
    video_paths = [f'app/lang/dataset/candidate_video/{i}.mp4' for i in sequence_names]
    print(f"The number of video paths returned by Langevin optimization.: {len(video_paths)}")
    print(f"Video path list: {video_paths}")
    return video_paths

def view_gan(sequence_names):
    import subprocess
    from datetime import datetime
    
    
    video_paths = []
    for seq_name in sequence_names:
        video_paths.append(f'app/lang/dataset/candidate_video/{seq_name}.mp4')

    
    if not video_paths:
        print("Error: No valid video paths found.")
        return None
    
    now = datetime.now()
    formatted_time = now.strftime("%m%d_%H_%M")
    output_file = f"app/lang/code/result/rendered_video_{formatted_time}.mp4"
    
    temp_list_file = f"app/lang/code/temp_video_list_{formatted_time}.txt"
    
    try:
        with open(temp_list_file, 'w') as f:
            for video_path in video_paths:
                f.write(f"file '{video_path}'\n")
        
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_list_file,
            '-c', 'copy',
            '-y',  
            output_file
        ]
        
        print(f"Start merging videos...")
        print(f"Input video: {sequence_names}")
        print(f"Output file: {output_file}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Video merge successful! Output file: {output_file}")
            return output_file
        else:
            print(f"Video merge failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
        return None
    
    finally:
        if os.path.exists(temp_list_file):
            os.remove(temp_list_file)


def read_json_files():
    import json


    json_file_path = 'app/lang/dataset/reference_video/view_type.json'#15
    json_file_path_m = 'app/lang/dataset/reference_video/mv_type.json'#15

    angle_type = []
    method_type = []
    view_type = []

    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        for item in data:
            for result in item.get("result", []):
                cls_res = result.get("cls_res", {})
                
                if "angle" in cls_res:
                    angle_type.append(cls_res["angle"][0][0])  

                if "method" in cls_res:
                    method_type.append(cls_res["method"][0][0])  

                if "view" in cls_res:
                    data_temp=cls_res["view"][0][0]
                    if data_temp=='远景':
                        data_temp='ELS'
                    elif data_temp=='全景':
                        data_temp='LS'
                    elif data_temp=='中景':
                        data_temp='MS'
                    elif data_temp=='近景':
                        data_temp='CU'
                    elif data_temp=='特写':
                        data_temp='ECU'
                    view_type.append(data_temp)  

    mv_type = []

    test_item=0

    with open(json_file_path_m, 'r', encoding='utf-8') as file:
        data = json.load(file)


        for item in data:

            for result in item.get("camera_motion", []):
                cls_res = result.get("motion", {})
                if cls_res[0][0]=='PAN-R':
                    data_temp='PAN-R'
                elif cls_res[0][0]=='PAN-L':
                    data_temp='PAN-L'
                elif cls_res[0][0]=='tilt_down':
                    data_temp='TILT-D'
                elif cls_res[0][0]=='tilt_up':
                    data_temp='TILT-U'
                elif cls_res[0][0]=='zoom_in':
                    data_temp='ZOOM-I'
                elif cls_res[0][0]=='zoom_out':
                    data_temp='ZOOM-O'
                elif cls_res[0][0]=='stable':
                    data_temp='STATIC'
                else:
                    data_temp='un'
                    data_temp=data_temp
                    test_item=test_item+1
                mv_type.append(data_temp)  

    print("angle_type =", angle_type)
    print("method_type =", method_type)
    print("view_type =", view_type)
    print("mv_type =", mv_type)
    return angle_type,method_type,view_type,mv_type



if __name__=="__main__":
    langevin_()

