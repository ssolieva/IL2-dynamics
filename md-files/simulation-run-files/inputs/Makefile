tag = 7raa_seq15_C125S
pdb = $(tag).pdb

ff = charmm36-adp-feb2021
water = tip3p
box_geometry = dodecahedron
box_edge_distance = 1.0 # usually 1.0 # changed

# use the following pbc approach
pbc = cluster

#sets solvent NaCl concentration in M otherwise ionize to neutral charge
molar_nacl = 0.1
pname = NA
nname = CL

# hacky solution to get sed to delete final line
d = d

#uncomment to use vsites
#vsite_atoms = hydrogens

#sshport = 8555

# .PHONY : packed
# packed : $(tag).tar.gz

.PHONY : prot-masses
prot-masses : $(tag)-prot-masses.pdb

.PHONY : npt_equilibrated
npt_equilibrated : $(tag)_npt.gro

.PHONY : nvt_equilibrated
nvt_equilibrated : $(tag)_nvt.gro

.PHONY : minimized
minimized : $(tag)_min.gro

.PHONY : solvated
solvated : $(tag)_solv.gro

.PHONY : clean
clean :
	find . ! -name "*.pdb" ! -name "Makefile" ! -name "em.mdp" ! -name "pr.mdp" ! -name "README*" ! -name "*.zip" -delete

#$(tag)_submission.log : $(tag)_equil.gro
#	ssh -L$(sshport):localhost:$(sshport) bowmore -N & \
#	python ~/projects/gromppery/client/new_project.py --gro $(tag)_equil.gro --top $(tag).top --mdp ../npt-500ns.mdp --project-name $(tag) --gromppery http://localhost:$(sshport) > $(tag)_submission.log

# $(tag).tar.gz : $(tag)-start.gro $(tag).top
# 	tar -cvzf $(tag).tar.gz $(tag)-start.gro $(tag).top $(tag)*itp README.txt

$(tag)-prot-masses.pdb : $(tag)-start.gro
	        echo '1 10' | gmx trjconv -f $(tag)_npt.gro -o $(tag)-prot-masses.pdb -s $(tag)_npt.tpr -pbc $(pbc)

$(tag)-start.gro : $(tag)_npt.gro
	        echo '1 1 0' | gmx trjconv -f $(tag)_npt.gro -o $(tag)-start.gro -s $(tag)_npt.tpr -pbc $(pbc) -center


$(tag)_npt.gro : $(tag)_npt.tpr
	bash -c "gmx mdrun -v -deffnm $(tag)_npt 2>&1 | tee npt_equilibration.log"

$(tag)_npt.tpr : $(tag)_nvt.gro npt-charmm.mdp
	gmx grompp -f npt-charmm.mdp -c $(tag)_nvt.gro -p $(tag).top -t $(tag)_nvt.cpt -o $(tag)_npt.tpr -maxwarn 1 -r $(tag)_nvt.gro

$(tag)_nvt.gro : $(tag)_nvt.tpr
	bash -c "gmx mdrun -v -deffnm $(tag)_nvt 2>&1 | tee nvt_equilibration.log"

$(tag)_nvt.tpr : $(tag)_min.gro nvt-charmm.mdp
	gmx grompp -f nvt-charmm.mdp -c $(tag)_min.gro -p $(tag).top -o $(tag)_nvt.tpr -maxwarn 1 -r $(tag)_min.gro

$(tag)_min.gro : $(tag)_min.tpr
	bash -c "gmx mdrun -v -deffnm $(tag)_min 2>&1 | tee minimization.log" && grep "Steepest Descents converged to Fmax" minimization.log

$(tag)_min.tpr : $(tag)_ions.gro minim.mdp
	gmx grompp -f minim.mdp -c $(tag)_ions.gro -p $(tag).top -o $(tag)_min.tpr

$(tag)_ions.gro : $(tag)_solv.tpr
ifdef molar_nacl
	echo "SOL" | gmx genion -s $(tag)_solv.tpr -o $(tag)_ions.gro -p $(tag).top -pname $(pname) -nname $(nname) -conc $(molar_nacl) -neutral
else
	echo "SOL" | gmx genion -s $(tag)_solv.tpr -o $(tag)_ions.gro -p $(tag).top -neutral
endif

$(tag)_solv.tpr : $(tag)_solv.gro ions.mdp
	gmx grompp -f ions.mdp -c $(tag)_solv.gro -p $(tag).top -o $(tag)_solv.tpr

$(tag)_solv.gro : $(tag)_box.gro
	gmx solvate -cp $(tag)_box.gro -cs -o $(tag)_solv.gro -p $(tag).top

$(tag)_box.gro : $(tag).gro
	bash -c "gmx editconf -c -f $(tag).gro -o $(tag)_box.gro -bt $(box_geometry) -d $(box_edge_distance) 2>&1" > editconf.log

$(tag).gro $(tag).top : $( pdb )
ifdef vsite_atoms
	bash -c "gmx pdb2gmx -f  $(pdb) -o $(tag).gro -p $(tag).top -ff $(ff) -ignh -vsite $(vsite_atoms) -water $(water) 2>&1" > prot.info || cat prot.info
else
	bash -c "gmx pdb2gmx -f  $(pdb) -o $(tag).gro -p $(tag).top -ff $(ff) -ignh -water $(water) 2>&1" > prot.info || cat prot.info
endif

# Equilibration will occur at 2 fs timestep
