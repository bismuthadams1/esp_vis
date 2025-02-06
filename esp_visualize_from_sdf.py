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
from naglmbis.models import load_charge_model

#there are more advanced models available, explore the naglmbis repo if you want to use them
nagl_mbis = load_charge_model(charge_model="nagl-v1-mbis-dipole")

class ESPFromSDF:
    
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
        
        charges = nagl_mbis.compute_properties(molecule)["mbis-charges"]
        charges = charges.detach().numpy().flatten()
        print("charges are:")
        print(charges)
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

    def process_and_launch_esp(self,
                                  sdf_file: float,
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
        """
        print('sdf to openff')
        self.openff_molecule = self._sdf_to_openff(sdf_file=sdf_file)
        # Validate counts
        print('compute charge models')
        charges = self._compute_charge_models(sdf_file=sdf_file)
        
        num_atoms = self.openff_molecule.n_atoms
        num_charges = len(charges)
        num_coords = len(self.openff_molecule.conformers[-1])

        assert num_atoms == num_charges == num_coords, "Mismatch in atom counts, charges, or coordinates."

        print('compute vdw radii')

        radii = self._compute_vdw_radii()
        
        print('compute surface')

        vertices, indices = self._compute_surface(radii)

        self.vertices = vertices
        self.indices = indices
        self.grid = vertices * unit.angstrom
        print('generate on atom esp')
        esp = self._generate_on_atom_esp(charge_list=charges)


        print('create esp molecule')

        #create esp molecule object for visualization
        esp_molecule = self._create_esp_molecule(esp)
        self.esp_molecule = esp_molecule
        print('launch')

        launch(esp_molecule, port)

        return esp, self.grid, self.esp_molecule
