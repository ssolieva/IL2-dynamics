import pyemma as py
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import itertools

def generate_distances(which_system, IL_resin, V_resin, save_path, directory_path):
    f = open(f"{directory_path}IL2_simulations_{which_system}/FAST_RMSD/msm/traj_list.txt", "r")
    traj_list = f.read().splitlines()
    print(f'Here is an example of a line in traj_list.txt: \n {traj_list[0]}')
    pdb_path = f'{directory_path}IL2_simulations_{which_system}/{which_system}-prot-masses.pdb'
    top_file = md.load(pdb_path) # load in the pdb file
    atom_indices_IL = top_file.topology.select(f'residue {IL_resin} and name CD1')
    atom_indices_V = top_file.topology.select(f'residue {V_resin} and name CG1')
    atom_indices1 = np.unique(np.concatenate([atom_indices_IL, atom_indices_V])) # put in order
    atom_indices = np.unique(atom_indices1)
    atom_pairs = list(itertools.combinations(atom_indices, 2))
    print(len(atom_pairs))
    resi_n_names = []
    for i in range(len(atom_indices1)):
        resi_n_names.append(top_file.topology.atom(atom_indices[i]).residue)
    print(resi_n_names)
    distances_feat = py.coordinates.featurizer(top_file)
    distances_feat.add_distances(atom_pairs)
    print("number of pairs (number of features):", len(distances_feat.describe()))
    distances_data = py.coordinates.load(traj_list, features=distances_feat) # calculate all features
    np.save(f'{save_path}data/system_{which_system}_distances_cpmg_ILV.npy', distances_data) # save the features
    np.save(f'{save_path}data/system_{which_system}_distances_cpmg_ILV_resis.npy', resi_n_names)



save_path = '../'
directory_path = '/Users/ssolieva/Desktop/bowman_lab/projects/IL2/simulations/'

IL_WT_S15_resin = '12 14 17 18 19 21 24 25 36 40 53 59 63 66 70 89 94 96 114 118 122 128 129 132'
V_WT_S15_resin = '91 93'

IL_S1_resin =     '12 14 17 18 19 21 24 25 36 40 53 59 63 66 70 87 92 94 112 116 120 126 127 130'
V_S1_resin = '89 91'

generate_distances( '1m47_c125s', IL_WT_S15_resin, V_WT_S15_resin, save_path, directory_path)
generate_distances('seq15_c125s', IL_WT_S15_resin, V_WT_S15_resin, save_path, directory_path)
generate_distances( 'seq1_c123s', IL_S1_resin, V_S1_resin, save_path, directory_path)
generate_distances( 'seq1_c123s_L56A', IL_S1_resin, V_S1_resin, save_path, directory_path)
generate_distances( 'seq1_c123s_V84A', IL_S1_resin, V_S1_resin, save_path, directory_path)
