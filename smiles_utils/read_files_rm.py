import os, json
import sys, csv
import gzip, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore", module="mordred")

__all__ = ['sdf_to_df']

def read_csv2array(file_path):

    file_ = pd.read_csv(file_path)
    file_ = np.array(file_)

    return file_

def read_csv2dict(file_path):

    read_dict = {}

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames
        i=0
        for row in reader:
            read_dict[i] = row
            i+=1

    return read_dict, headers

def load_dict_pickle(input_filename):
    
    with open(input_filename,'rb') as f:
        dict_name = pickle.load(f)

    return dict_name

def line_count_csv_file(filename, chunksize=1000):

    print('Counting number of data point in: "{}"' .format(filename))
    count = 0
    for chunk in tqdm(pd.read_csv(filename, usecols=[0], chunksize=chunksize)):
        count += len(chunk)

    return count

# Function to load a json file :-
def load_json(path):

    with open(path, 'r') as fp:
        data = json.loads(fp.read())

    return data

def read_csv2list(file_path):
    
    '''
    - Reads a .csv file in the given filepath and returns it in a list format.
    '''

    with open(file_path, mode ='r')as file:
        csv_reader = csv.reader(file)

        headers = next(csv_reader)
        rows = []
        # displaying the contents of the CSV file
        for line in csv_reader:
            rows.append(line)

    return rows, headers

def read_data_from_excel(file_path: str, sheet_num: int = None):
    
    # To read all sheets into a dictionary of DataFrames
    sheets_dict = pd.read_excel(file_path, sheet_name=None)
    sheets = []
    # dataframes = []
    # Access each sheet by name
    for sheet_name in sheets_dict.keys():
        sheets.append(sheet_name)
        # dataframes.append(df)  # Display the first few rows of each sheet
    
    if not isinstance(sheet_num, int): 
        print(f'Keys: {list(sheets_dict.keys())}')
        return sheets_dict
    else:   
         return sheets_dict[sheets[sheet_num]]

def sdf_to_df(filepath):
    
    if filepath.endswith('.gz'):
        with gzip.open(filepath, "rb") as f:
            suppl = Chem.ForwardSDMolSupplier(f)
            mols = [mol for mol in suppl if mol is not None]
    else:
        suppl = Chem.ForwardSDMolSupplier(filepath)
        mols = [mol for mol in suppl if mol is not None]

    # # Step 2: Extract data
    records = []
    for mol in mols:
        data = {}
        data["SMILES"] = Chem.MolToSmiles(mol)
        # data["Name"] = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        for prop in mol.GetPropNames():
            data[prop] = mol.GetProp(prop)
        records.append(data)

    # Step 3: Save to CSV
    df = pd.DataFrame(records)
    
    return df, mols