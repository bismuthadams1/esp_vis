from molesp.models import ESPMolecule, Surface
from molesp.cli._cli import compute_surface
from openff.toolkit.topology import Molecule
from openff.units import unit
from typing import Union
from rdkit import Chem
from rdkit.Chem import rdmolfiles
import numpy as np
from point_charge_esp import calculate_esp
from openff.recharge.utilities.toolkits import VdWRadiiType, compute_vdw_radii
import json
from molesp.gui import launch
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import AllChem

class ESPtoSDF:
    
    def __init__(self) -> None:
        """
        
        Parameters
        ----------

        Attributes
        ----------
        grid
        
        vertices
        
        indices
        
        """
        self.grid = None
        self.vertices = None
        self.indices = None    
        self.openff_molecule = None
        self.rdkit_molecule = None


    def _sdf_to_openff(self, sdf_file: str) -> Molecule:
        """
        Convert RDKit molecule to OpenFF Molecule.
        """
        # Read the molecule using RDKit
        supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)
        rdkit_molecules = [mol for mol in supplier if mol is not None]
        rdkit_molecule = rdkit_molecules[0]
        self.rdkit_molecule = rdkit_molecule  # Store RDKit molecule
        print(f"Number of atoms in RDKit molecule: {rdkit_molecule.GetNumAtoms()}")

        # Convert RDKit molecule to OpenFF Molecule
        openff_molecule = Molecule.from_rdkit(rdkit_molecule, allow_undefined_stereo=True, hydrogens_are_explicit=True)
        # openff_molecule = Molecule.from_file(rdkit_molecule, allow_undefined_stereo=True)

        print(f"Number of atoms in OpenFF molecule: {openff_molecule.n_atoms}")

        # Get RDKit conformer coordinates
        rdkit_conf = rdkit_molecule.GetConformer()
        rdkit_coords = np.array(rdkit_conf.GetPositions()) * unit.angstrom

        # Assign the RDKit coordinates to the OpenFF molecule
        openff_molecule._conformers = [rdkit_coords]

        # Center the conformer
        centroid = np.mean(rdkit_coords, axis=0)
        openff_molecule._conformers[0] = rdkit_coords - centroid

        return openff_molecule

    def _compute_surface(self, 
                         radii: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        """"
            Compute the surface to project the ESP on
        Parameters
        ----------
        openff_molecule
            openff molecule from tagged smiles in db
        molecule_props
            list of molecule properties associated with openff_molecule
        radii
            VdW radii 
        Returns
        -------
            Tuple of vertices and indices associated with the surface.
        """
        
        vertices, indices = compute_surface(
            molecule=self.openff_molecule,
            conformer=self.openff_molecule.conformers[-1],
            radii=radii,
            radii_scale=1.1,
            spacing=0.2 * unit.angstrom,
        )
        return vertices, indices
    
    def _generate_on_atom_esp(self,
                              charge_list: list[float],
                              charge_sites: np.ndarray | None = None) -> unit.Quantity:
        """"
        takes in a list of on atom charges and produces an ESP for them
        Parameters
        ----------
        on_atom_charges
            list of on atom charges
        Returns
        -------
        on_atom_esp
            on atom esp formed from the conformer and on atom chargers
        """
        if charge_sites is None :
            charge_sites = self.openff_molecule.conformers[-1] 
              
        on_atom_esp =  calculate_esp(
            self.grid,
            charge_sites,
            charge_list * unit.e,
            with_units= True
        ).to(unit.hartree/unit.e)

        on_atom_esp = on_atom_esp.magnitude.reshape(-1, 1)
        on_atom_esp = on_atom_esp * unit.hartree/unit.e

        return on_atom_esp
    
    def _compute_charge_models(self, sdf_file: str) -> list[float]:
        """Compute partial charges for openff molecule
        
        """
        supplier = Chem.SDMolSupplier(sdf_file, removeHs=False) #, sanitize=False
        molecules = [mol for mol in supplier if mol is not None]
        molecule = molecules[0]
        # rdDetermineBonds.DetermineConnectivity(molecule)
        # Chem.SanitizeMol(molecule)
        molblock = rdmolfiles.MolToMolBlock(molecule)
        
        print(f'molblock is {molblock}')
        charge_request = module_version.handle_charge_request(
            conformer_mol=molblock,
            charge_model='MBIS_WB_GAS_ESP_DEFAULT',
            batched=False
        )
        print('charge request errors:')
        print(charge_request['error'])
        charges = json.loads(charge_request['charge_result'])
        
        return charges

    def _create_esp_molecule(self, 
                             esp: unit.Quantity) -> ESPMolecule:
        """
        creates an ESPMolecule class containing all the visualisation information
        Parameter
        ---------
        openff_molecule
            openff molecule from tagged smiles in db
        molecule_props
            list of molecule properties associated with openff_molecule
        vertices
            vertices associated with ESP surface
        indices
            indices associated with ESP surface
        esp
            ESP associated with the molecule
        Returns
        -------
            ESP molecule object
        """
        esp_molecule = ESPMolecule(
            atomic_numbers=[atom.atomic_number for atom in self.openff_molecule.atoms],
            conformer=self.openff_molecule.conformers[-1].m.flatten().tolist(),
            surface=Surface(
                vertices=self.vertices.flatten().tolist(),
                indices=self.indices.flatten().tolist(),
            ),
            esp={"Charge_model ESP": np.round(esp,7).m_as(unit.hartree / unit.e).flatten().tolist()},
        )
        return esp_molecule

    def _compute_vdw_radii(self) -> np.ndarray:
        """
            compute the VdW radii  
        Parameters
        ----------
   
        Returns
        -------
           Returns VdW radii array.
        """
        vdw_radii = compute_vdw_radii(self.openff_molecule, radii_type=VdWRadiiType.Bondi) 
        radii = np.array([[r] for r in vdw_radii.m_as(unit.angstrom)]) * unit.angstrom
        return radii
    
    def add_dummy_atoms_to_molecule(self, rdkit_mol, dummy_coords, dummy_charges):
        """
        Adds dummy atoms (atomic number 0) with known charges and 3D coordinates 
        to an existing RDKit molecule. Returns a new RDKit Mol object.

        Parameters
        ----------
        rdkit_mol : rdkit.Chem.Mol
            The starting molecule (already has some atoms).
        dummy_coords : list of (float, float, float)
            Coordinates for each new dummy atom.
        dummy_charges : list of int
            Formal charges for each new dummy atom.

        Returns
        -------
        new_mol : rdkit.Chem.Mol
            A new molecule containing the old atoms plus new dummy atoms.
        """
        print(f'length of dummy_coords is {len(dummy_coords)}')
        print(f'length of dummy_charges is {len(dummy_charges)}')
        if len(dummy_coords) != len(dummy_charges):
            raise ValueError("dummy_coords and dummy_charges must have the same length")

        # Convert to editable RWMol
        rw_mol = Chem.RWMol(rdkit_mol)

        # --- 1) Get the existing conformer (if any) ---
        if rdkit_mol.GetNumConformers() == 0:
            raise ValueError("Input molecule has no conformers. Provide coordinates or embed first.")

        old_conf = rdkit_mol.GetConformer(0)
        n_old_atoms = rdkit_mol.GetNumAtoms()

        # --- 2) Create a new conformer with enough room for old + new atoms ---
        new_conf = Chem.Conformer(n_old_atoms + len(dummy_coords))

        # Copy over the old coordinates
        for i in range(n_old_atoms):
            pos_i = old_conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(i, pos_i)

        # --- 3) Add new dummy atoms ---
        for i, (coords, chg) in enumerate(zip(dummy_coords, dummy_charges)):
            atom = Chem.Atom("*")  # atomic number 0 => "Du" (dummy)
            atom.SetDoubleProp('_TriposPartialCharge',int(chg.m[0]))
            new_index = rw_mol.AddAtom(atom)
            new_conf.SetAtomPosition(new_index, coords)

        # --- 4) Replace the old conformer with the new one ---
        # Remove the old conformers and add the new expanded one
        while rw_mol.GetNumConformers():
            rw_mol.RemoveConformer(0)
        rw_mol.AddConformer(new_conf)

        # Convert back to a normal Mol
        new_mol = rw_mol.GetMol()
        return new_mol

    def write_pdb_with_charges(self, mol, partial_charges, filename):
        """
        Write a minimal PDB for `mol`, storing partial charges in the B-factor column.
        The partial_charges list must match the number of atoms in `mol`.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            RDKit molecule with at least one conformer.
        partial_charges : list of float
            One partial charge per atom in the same order as mol.GetAtoms().
        filename : str
            Output PDB filename.
        """
        print(mol.GetNumAtoms())
        print(len(partial_charges))
        if mol.GetNumAtoms() != len(partial_charges):
            raise ValueError("Number of partial charges must match the number of atoms.")
        if mol.GetNumConformers() == 0:
            raise ValueError("Molecule has no conformers (no 3D coordinates).")

        conf = mol.GetConformer()
        lines = []
        atom_counter = 0

        # For each atom, build a PDB "HETATM" line or "ATOM" line
        for i, atom in enumerate(mol.GetAtoms()):
            atom_counter += 1
            symbol = atom.GetSymbol()
            # If you need a unique atom name, pad/truncate to 4 chars. Just a simple example:
            atom_name = f"{symbol}{i}".ljust(4)[:4]

            # Retrieve partial charge
            charge = partial_charges[i]
            if isinstance(charge, unit.Quantity):
                charge = charge.m[0]
            # Coordinates
            x, y, z = conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z

            # Occupancy = 1.00, B-factor = partial charge (up to 2 decimals for clarity)
            occupancy = 1.00
            b_factor = charge

            # Build the line (widths are fixed in the PDB format).
            # Example format (fields):
            # HETATM  atomNo  atomName  resName  chainID  resSeq   x       y       z    occupancy Bfactor
            # Columns: 
            #  1-6  Record name "HETATM" or "ATOM  "
            #  7-11 Integer atom serial number
            # 13-16 Atom name
            # 17    Alternate location indicator (A, etc.)
            # 18-20 Residue name
            # 22    Chain ID
            # 23-26 Residue sequence number
            # 31-38 x
            # 39-46 y
            # 47-54 z
            # 55-60 occupancy
            # 61-66 tempFactor
            # 77-78 element symbol
            line = (
                f"HETATM{atom_counter:5d} {atom_name:4s} UNL A   1    "  # residue name "UNL", chain 'A', resSeq=1
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{occupancy:6.2f}{b_factor:6.2f}          "  # two fields of 6.2
                f"{symbol:>2s}\n"
            )
            lines.append(line)

        lines.append("END\n")

        # Write to file
        with open(filename, "w") as f:
            f.writelines(lines)
            
    def make_charge_sdf(self,
                        sdf_file: str,
                        port: int = 8000
                        ) -> None:
        """
        Produce QM ESP using the supplied ESPSettings, Molecule, Conformer
        Paramters
        ---------
        sdf_file: str
            sdf file path in .sdf format. Currently only computes single molecule / sdf
        
        port: int
            port in which the local host will be launched
            
        Return
        ------
        
        
        
        """
        self.openff_molecule = self._sdf_to_openff(sdf_file=sdf_file)
        # Validate counts
        charges = self._compute_charge_models(sdf_file=sdf_file)
        
        num_atoms = self.openff_molecule.n_atoms
        num_charges = len(charges)
        num_coords = len(self.openff_molecule.conformers[-1])

        print(f"Number of atoms: {num_atoms}")
        print(f"Number of charges: {num_charges}")
        print(f"Number of coordinates: {num_coords}")

        assert num_atoms == num_charges == num_coords, "Mismatch in atom counts, charges, or coordinates."

        print('computing radii')
        radii = self._compute_vdw_radii()
        print('radii computed')
        print('computing surface')
        # vertices, indices = self._compute_surface(radii)
        
        rdkit_mol = self.openff_molecule.to_rdkit()
        

        # self.vertices = vertices
        # self.indices = indices
        # self.grid = vertices * unit.angstrom
        # esp = self._generate_on_atom_esp(charge_list=charges)
        
        # n_real = rdkit_mol.GetNumAtoms()  # original atoms
        # n_dummy = len(vertices)
        # partial_charges_full = [0.0] * n_real + [0.0] * n_dummy

        # # Now fill in the last part with the actual dummy-atom charges:
        # for i in range(n_dummy):
        #     partial_charges_full[n_real + i] = esp[i]
            
        # print('esp generated')
        # print(esp)        
        # print('creating esp molecule with dummy atoms')
        # dummy_mol = self.add_dummy_atoms_to_molecule(
        #     rdkit_mol=rdkit_mol,
        #     dummy_coords=vertices,
        #     dummy_charges=esp
        # )
        self.write_pdb_with_charges(
            mol=rdkit_mol,
            partial_charges=charges,
            filename=f"{sdf_file.strip('.sdf')}_cloud.pdb",
        )
            

        # print('writing file')
        # rdmolfiles.MolToMolFile(new_mol, f"{sdf_file.strip('.sdf')}_cloud.mol")


