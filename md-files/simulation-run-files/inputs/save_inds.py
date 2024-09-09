import numpy as np

import mdtraj as md 

pdb = md.load("7raa_seq15_C125S-prot-masses.pdb")
atom_inds = pdb.topology.select("protein and name N CA CB C O")
for i in atom_inds:
    print(i)

