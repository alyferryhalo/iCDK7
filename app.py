import pandas as pd
from pandas import array
from pandas import DataFrame

import joblib

import numpy as np
from numpy import zeros, array

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors, AllChem
import rdkit.Chem.AllChem as AllChem

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import import_ipynb
from descriptors_calculation import scaler

import warnings
warnings.filterwarnings('ignore')


user_smiles = input('Введите молекулу в формате SMILES: ')
user_mol = Chem.MolFromSmiles(str(user_smiles))

aromatic_rings = Chem.Lipinski.NumAromaticRings(user_mol)
heteroatoms = Chem.Lipinski.NumHeteroatoms(user_mol)
aliphatic_heterocycles = Chem.Lipinski.NumAliphaticHeterocycles(user_mol)
h_acceptors = Chem.Lipinski.NumHAcceptors(user_mol)
h_donors = Chem.Lipinski.NumHAcceptors(user_mol)
mw = Descriptors.MolWt(user_mol)
logp = Descriptors.MolLogP(user_mol)
tpsa = Descriptors.TPSA(user_mol)

mol_desc = [aromatic_rings, heteroatoms, aliphatic_heterocycles, h_acceptors, h_donors, mw, logp, tpsa]
columns = ['aromatic_rings', 'heteroatoms', 'aliphatic_heterocycles', 'h_acceptors', 'h_donors', 'mw', 'logp', 'tpsa']

df = pd.DataFrame([mol_desc], columns=columns)

def calc_morgan_one(mol):
    for_df = []
    arr = zeros((1,), dtype='float32')
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048), arr)
    for_df.append(arr)
    return DataFrame(for_df)

morgan_transformer = FunctionTransformer(calc_morgan_one, validate=False)
M = morgan_transformer.transform(user_mol)
user_data = pd.concat([df,M],axis=1)
user_smiles_X = DataFrame(scaler.transform(user_data.values), index=user_data.index, columns=user_data.columns)

model = joblib.load('gbc4_model.sav')
pred = model.predict_proba(user_smiles_X)[:,1]

print(f'Вероятность того, что {user_smiles} является ингибитором CDK7: ', pred)
