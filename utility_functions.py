import os
# Uncomment if you don't want to use GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from openff.units import unit
import numpy as np
from chargecraft.storage.storage import MoleculePropRecord, MoleculePropStore
from openff.toolkit.topology import Molecule

from rdkit import Chem
import json
from rdkit.Chem import Draw
from typing import Literal, Optional
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from openff.units import unit
from openff.toolkit.topology import Molecule
from typing import Union, Optional
from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper

import matplotlib.pyplot as plt
import numpy as np

HA_TO_KCAL_P_MOL = 627.509391  # Hartrees to kilocalories per mole


AU_ESP = unit.atomic_unit_of_energy / unit.elementary_charge

OPEN_FF_AVAILABLE_CHARGE_MODELS = ['zeros','formal_charge','espaloma-am1bcc']

NAGL_GNN_MODELS = []

def scan_toolkit_registry():
    """Scan the toolkit registry of openff
    
    """
    from openff.toolkit.utils import toolkits

    toolkits = str(toolkits.GLOBAL_TOOLKIT_REGISTRY)

    if 'RDKIT' in toolkits:
        OPEN_FF_AVAILABLE_CHARGE_MODELS.extend(['gasteiger', 'mmff94'])
    elif 'AmberTools' in toolkits:
        OPEN_FF_AVAILABLE_CHARGE_MODELS.extend(['am1bcc', 'am1-mulliken', 'gasteiger'])
    elif 'OpenEye' in toolkits:
        OPEN_FF_AVAILABLE_CHARGE_MODELS.extend(['am1bccnosymspt','am1elf10','am1bccelf10'])
    try:

        NAGL_GNN_MODELS.extend(['nagl-small','nagl-small-weighted'])
    except Exception as e:
        print("could not load GNN nagl models")

scan_toolkit_registry()