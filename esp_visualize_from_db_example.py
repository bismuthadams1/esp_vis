from esp_from_conformer import ESPProcessor
import sys
import numpy as np

from chargecraft.storage.storage import MoleculePropRecord, MoleculePropStore
from point_charge_esp import calculate_esp
from openff.toolkit.topology import Molecule
from openff.units import unit
from molesp.gui import launch
from collections import OrderedDict

"""
Smiles list in prop store:
['OCC(O)CO',
 'C#CC',
 'C1CN1',
 'C1COC1',
 'CC#N',
 'CC(=O)[O-]',
 'CC(C)=O',
 'CCCC',
 'CCNCC',
 'CN(C)C',
 'CN=[N+]=[N-]',
 'CNC',
 'COC',
 'CSC',
 'Fc1ccccc1',
 'NCO',
 'Nc1ccccc1',
 'O=[NH+][O-]',
 'Oc1ccccc1',
 'CCl',
 'CF',
 'CO',
 'CS',
 'C1COCO1',
 'c1ccccc1',
 'c1ccncc1',
 'c1ccsc1']
"""
MOLECULE_STR = 'CO'
CONFORMER = 0
#load the prop_store so we can 
prop_store_path = "properties_store.db"
prop_store_2 = MoleculePropStore(prop_store_path)
print(prop_store_2.list())
#retrieve the first conformer
conformer = prop_store_2.retrieve(MOLECULE_STR)[CONFORMER]
#retrieve the mapped smiles
mapped_smiles = conformer.tagged_smiles
partial = prop_store_2.retrieve_partial(
    smiles=mapped_smiles,
    conformer=conformer.conformer
)
#add all the partial charges into two lists
charges_names = list(partial.keys())
print(charges_names)
charges = list(partial.values())
charges_dict = OrderedDict((key, value) for key, value in zip(charges_names,charges))
on_atom_charges = list(charges_dict.values())
on_atom_charge_names = list(charges_dict.keys())
#add the charge partitioning charges to the list
on_atom_charges.extend([conformer.mulliken_charges,conformer.lowdin_charges,conformer.mbis_charges])
on_atom_charge_names.extend(['mulliken','lowdin','mbis'])
#start the ESPProcesser class
qm_esp = ESPProcessor(prop_store_path = prop_store_path,
                      port = 8200,
                      molecule = MOLECULE_STR,
                      conformer = CONFORMER,
                      display_difference=True) 
#this will first generate the qm esp, ctrl + c to break the subprocess to generate the on-atom esps and refresh the localhost:8100
print(on_atom_charges)
esp, grid, esp_molecule = qm_esp.process_and_launch_qm_esp()
qm_esp.add_on_atom_esps(on_atom_charges= on_atom_charges, labels = on_atom_charge_names)
qm_esp.riniker_esp()





