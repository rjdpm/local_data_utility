import pandas as pd
import time

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from data_utils_local.data_utils_smiles import mol_to_features_batch
from data_utils_local.Generalised_data_utils import attach_ids, read_data_from_excel, create_folder

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from typing import Optional, Dict, Any, Tuple
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
import logging
logging.getLogger('mordred').setLevel(logging.CRITICAL)

class SMILES_PRUNING:
    
    def __init__(self,
                 smiles_list,
                 labels,
                 smiles_column_name='smiles',
                 filters = None,
                 all_atoms={'Br', 'Cl', 'P', 'I', 'F', 'H', 'S', 'N', 'O', 'C', 'B', 'Si', 'Na', 'K'},
                 prefix='',
                 sanitize=True,
                 strip_salts: bool = False,
                salt_remover_def: Optional[str] = None,
                keep_largest_fragment: bool = True,
                require_organic_fragment: bool = True,
                disconnect_metals: bool = True,
                normalize_functional_groups: bool = True,
                reionize: bool = True,
                uncharge: bool = True,
                canonicalize_tautomer: bool = True,
                remove_explicit_h: bool = True,
                sanitize_final: bool = True,
                kekulize: bool = False,
                clear_stereo: bool = False,
                canonical: bool = True,
                return_mol: bool = False,
                log_flag: bool = False
                 ):
        '''
            Supported columns for filter: ['MolWt', 'MolLogP', 'MolMR', 'qed', 'TPSA', 'NumHDonors']
        '''
        self.labels = labels
        self.all_atoms = all_atoms
        self.prefix = prefix
        self.filters = filters
        self.smiles_column_name = smiles_column_name
        self.strip_salts = strip_salts
        self.salt_remover_def = salt_remover_def
        self.keep_largest_fragment = keep_largest_fragment
        self.keep_largest_fragment = keep_largest_fragment
        self.require_organic_fragment = require_organic_fragment
        self.disconnect_metals = disconnect_metals
        self.normalize_functional_groups = normalize_functional_groups
        self.reionize = reionize
        self.uncharge = uncharge
        self.canonicalize_tautomer = canonicalize_tautomer
        self.remove_explicit_h = remove_explicit_h
        self.sanitize_final = sanitize_final
        self.kekulize = kekulize
        self.clear_stereo = clear_stereo
        self.canonical = canonical
        self.return_mol = return_mol
        self.log_flag = log_flag
        self.smiles_list = [Chem.MolFromSmiles(s, sanitize=sanitize) for s in smiles_list]
        self.log = []
        
        if isinstance(labels, str) and labels == 'eval':
            self.labels = [None]*len(self.smiles_list)
        else:
            assert len(self.smiles_list) == len(self.labels), f'ERROR: smiles (n_items - {len(self.smiles_list)}) and labels (n_items - {len(self.labels)}) should be of same lengths.'
            
        self.df = pd.DataFrame({self.smiles_column_name:self.smiles_list, 'labels':self.labels})
        self.df = self.df[~self.df[self.smiles_column_name].isna()]
        if self.prefix:
            self.df = attach_ids(self.df, self.prefix)
        if self.filters is not None:
            self.df['MolWt'] = self.df[self.smiles_column_name].apply(Chem.Descriptors.MolWt)
            self.df['MolLogP'] = self.df[self.smiles_column_name].apply(Chem.Descriptors.MolLogP)
            self.df['qed'] = self.df[self.smiles_column_name].apply(Chem.Descriptors.qed)
            self.df['TPSA'] = self.df[self.smiles_column_name].apply(Chem.Descriptors.TPSA)
            self.df['NumHDonors'] = self.df[self.smiles_column_name].apply(Chem.Descriptors.NumHDonors)
        
    def _safe_mol_from_smiles(self, smiles: str, sanitize: bool = True) -> Tuple[Optional[Chem.Mol], Optional[str]]:
        """
        Try to parse a SMILES safely. Returns (mol, error_message).
        """
        if not smiles or not isinstance(smiles, str):
            return None, "empty_or_nonstring_input"

        try:
            # allow deferred sanitization if needed later
            m = Chem.MolFromSmiles(smiles, sanitize=sanitize)
            if m is None:
                return None, "rdkit_failed_to_parse"
            return m, None
        except Exception as e:
            return None, f"parse_exception:{str(e)}"


    def _select_largest_fragment(self, mol: Chem.Mol, require_organic: bool = True) -> Chem.Mol:
        """
        Select the largest fragment by heavy atom count. If require_organic is True,
        prefer fragments containing C (a simple organic heuristic).
        """
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if not frags:
            return mol

        # filter organic fragments if requested
        if require_organic:
            organic_frags = [f for f in frags if any(a.GetSymbol() == 'C' for a in f.GetAtoms())]
            if organic_frags:
                frags = organic_frags

        # choose fragment with maximum heavy atoms
        return max(frags, key=lambda m: m.GetNumHeavyAtoms())


    def sanitize_smiles_master(self,
                            smiles: str|Chem.Mol,
                            *,
                            strip_salts: bool = True,
                            salt_remover_def: Optional[str] = None,
                            keep_largest_fragment: bool = True,
                            require_organic_fragment: bool = True,
                            disconnect_metals: bool = True,
                            normalize_functional_groups: bool = True,
                            reionize: bool = True,
                            uncharge: bool = True,
                            canonicalize_tautomer: bool = True,
                            remove_explicit_h: bool = True,
                            sanitize_final: bool = True,
                            kekulize: bool = False,
                            clear_stereo: bool = False,
                            canonical: bool = True,
                            return_mol: bool = False,
                            log_flag: bool = False
                            ) -> Dict[str, Any]:
        """
        Comprehensive step-by-step SMILES sanitization pipeline.

        Returns a dictionary with:
        - 'input_smiles' : original input
        - 'status' : 'success' or 'failed'
        - 'final_smiles' : resulting canonical SMILES (or None on failure)
        - 'mol' : RDKit Mol if return_mol True and success
        - 'steps' : list of step logs (each is dict with 'step', 'smiles', 'note')
        - 'error' : error message if any

        Parameters are self-explanatory; tweak as required.
        """

        
        original = smiles
        if isinstance(smiles, str):
            mol, err = self._safe_mol_from_smiles(smiles, sanitize=True)
        else:
            mol=smiles
        if mol is None:
            if log_flag:
                self.log.append({"step": "parse", "smiles": None, "note": err})
            return {
                "input_smiles": original,
                "status": "failed",
                "final_smiles": None,
                "steps": self.log,
                "error": err,
            }
        if log_flag:
            self.log.append({"step": "parse", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "parsed"})

        try:
            # Step 1: Strip salts/solvents (SaltRemover)
            if strip_salts:
                if salt_remover_def:
                    remover = SaltRemover(defnData=salt_remover_def)
                else:
                    # default definition removes common salts (RDKit built-in)
                    remover = SaltRemover()
                mol_stripped = remover.StripMol(mol, dontRemoveEverything=True)
                if log_flag:
                    self.log.append({
                    "step": "strip_salts",
                    "smiles": Chem.MolToSmiles(mol_stripped, canonical=False) if mol_stripped else None,
                    "note": "salt_removal_applied"
                })
                mol = mol_stripped

            # Step 2: Keep largest fragment (prefer organic)
            if keep_largest_fragment:
                mol_before = Chem.MolToSmiles(mol, canonical=False) if mol else None
                mol = self._select_largest_fragment(mol, require_organic=require_organic_fragment)
                if log_flag:
                    self.log.append({
                    "step": "largest_fragment",
                    "smiles": Chem.MolToSmiles(mol, canonical=False) if mol else None,
                    "note": f"selected_largest_fragment (organic_required={require_organic_fragment}) from {mol_before}"
                })

            # Step 3: Disconnect metals (break coordinate bonds)
            if disconnect_metals:
                md = rdMolStandardize.MetalDisconnector()
                mol = md.Disconnect(mol)
                if log_flag:
                    self.log.append({
                    "step": "disconnect_metals",
                    "smiles": Chem.MolToSmiles(mol, canonical=False),
                    "note": "metal_disconnected"
                })

            # Step 4: Normalize functional groups (Normalizer)
            if normalize_functional_groups:
                normalizer = rdMolStandardize.Normalizer()
                mol = normalizer.normalize(mol)
                if log_flag:
                    self.log.append({"step": "normalize", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "normalizer_applied"})

            # Step 5: Reionize (preferred protonation)
            if reionize:
                reionizer = rdMolStandardize.Reionizer()
                mol = reionizer.reionize(mol)
                if log_flag:
                    self.log.append({"step": "reionize", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "reionizer_applied"})

            # Step 6: Uncharge (neutralize where sensible)
            if uncharge:
                uncharger = rdMolStandardize.Uncharger()
                mol = uncharger.uncharge(mol)
                if log_flag:
                    self.log.append({"step": "uncharge", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "uncharger_applied"})

            # Step 7: Tautomer canonicalization
            if canonicalize_tautomer:
                taut_enum = rdMolStandardize.TautomerEnumerator()
                mol = taut_enum.Canonicalize(mol)
                if log_flag:
                    self.log.append({"step": "tautomer", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "tautomer_canonicalized"})

            # Step 8: Clear stereo if requested
            if clear_stereo:
                Chem.RemoveStereochemistry(mol)
                if log_flag:
                    self.log.append({"step": "clear_stereo", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "stereo_removed"})

            # Step 9: Remove explicit Hs if requested
            if remove_explicit_h:
                mol = Chem.RemoveHs(mol)
                if log_flag:
                    self.log.append({"step": "remove_explicit_h", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "explicit_H_removed"})

            # Step 10: Final sanitize and optional kekulize
            if sanitize_final:
                Chem.SanitizeMol(mol)
                if log_flag:
                    self.log.append({"step": "final_sanitize", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "sanitized"})
            if kekulize:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                    if log_flag:
                        self.log.append({"step": "kekulize", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "kekulized"})
                except Exception:
                    # some molecules cannot be kekulized; not fatal
                    if log_flag:
                        self.log.append({"step": "kekulize", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "kekulize_failed_or_not_applicable"})

            # Final canonicalization
            final_smiles = Chem.MolToSmiles(mol, canonical=canonical)
            if log_flag:
                self.log.append({"step": "final_canonical", "smiles": final_smiles, "note": "done"})

            out = {
                "input_smiles": original,
                "status": "success",
                "final_smiles": final_smiles,
                "steps": self.log,
                "error": None,
            }
            if return_mol:
                out["mol"] = mol
            return out

        except Exception as e:
            # Fail gracefully and include logs for debugging
            if log_flag:
                self.log.append({"step": "exception", "smiles": None, "note": str(e)})
            return {
                "input_smiles": original,
                "status": "failed",
                "final_smiles": None,
                "steps": self.log,
                "error": str(e),
            }
            
    def sanitize_smiles_list(self, smiles_list):
        
        cleaned_smiles=[None]*len(smiles_list)
        for i, s in enumerate(smiles_list):
            cleaned_smiles[i] = self.sanitize_smiles_master(smiles=s,
                                                            strip_salts = self.strip_salts,
                                                            salt_remover_def = self.salt_remover_def,
                                                            keep_largest_fragment = self.keep_largest_fragment,
                                                            require_organic_fragment = self.require_organic_fragment,
                                                            disconnect_metals = self.disconnect_metals,
                                                            normalize_functional_groups = self.normalize_functional_groups,
                                                            reionize = self.reionize,
                                                            uncharge = self.uncharge,
                                                            canonicalize_tautomer = self.canonicalize_tautomer,
                                                            remove_explicit_h = self.remove_explicit_h,
                                                            sanitize_final = self.sanitize_final,
                                                            kekulize = self.kekulize,
                                                            clear_stereo = self.clear_stereo,
                                                            canonical = self.canonical,
                                                            return_mol = self.return_mol,
                                                            log_flag = self.log_flag
                                                            )
        # Extract the final SMILES only (None if failed)
        cleaned_smiles = [r["final_smiles"] for r in cleaned_smiles]
        
        return cleaned_smiles

    def get_cleaned_smiles(self):
        
        self.df[self.smiles_column_name] = self.sanitize_smiles_list(self.df[self.smiles_column_name])
        # self.df[self.smiles_column_name] = self.df[self.smiles_column_name].apply(Chem.MolToSmiles)
        # if self.filters:
        #     self.df = apply_column_constraints(self.df, self.filters)
        
        return self.df

def calculate_descriptors(df, smiles_column_name):
        
    desc_df = mol_to_features_batch(df[smiles_column_name].tolist(), smiles_column_name=smiles_column_name)
    df = pd.merge(df, desc_df, how='inner', on=smiles_column_name)
    
    return df

## Use for single dataset
# if __name__=='__main__':
#     start = time.time()
#     path_init = "/home/rkmvu/Dataset/Fraction_unbound/all_datasets"
#     create_folder(path_init)
#     source_datapath = f"{path_init}/all_data_merged_source.xlsx"
#     data_savepath = f'{path_init}/comparison_datasets/Fu_cleaned_data_091125.csv'
#     smiles_column_name='Canonicalised_SMILES'
#     target_col_name = 'Fu_mean'
#     id_col = 'Unique_ID'
#     prefix='FU091125'
#     pruning = True#False
    
#     df = read_data_from_excel(source_datapath, 8)
#     df = attach_ids(df, prefix=prefix, id_col=id_col)
    
#     if pruning:
#         smiles_pruning = SMILES_PRUNING(smiles_list=df[smiles_column_name].tolist(),
#                                     labels=df[id_col],
#                                     # filters=[('MolWt', '<', 1000)],
#                                     smiles_column_name=smiles_column_name,
#                                     prefix=''
#                                     )
#         df_temp = smiles_pruning.get_cleaned_smiles()[['labels', smiles_column_name]]
#         df_temp = df_temp.rename(columns={'labels':id_col})
        
#         df = pd.merge(df_temp, df.drop(columns=[smiles_column_name]), how='inner', on=id_col) 
#     df = calculate_descriptors(df, smiles_column_name)

#     # Append as a new sheet
#     df.to_csv(data_savepath, index=False)
    
#     avg_time =  (time.time()-start)/len(df)
#     print(f'Time -> Total: {time.time()-start} sec | average: {avg_time} sec | Items: {len(df)}')
    
## Use for multiple dataset
if __name__=='__main__':
    start = time.time()
    smiles_column_name='Canonicalised_SMILES'
    target_col_name = 'Fu_mean'
    id_col = 'Unique_ID'
    pruning = True#False
    datapath = "/home/rkmvu/Dataset/Fraction_unbound/all_datasets/all_data_merged_source_updated_04012026.xlsx"
    df = pd.read_excel(datapath, sheet_name=None)
    all_keys = df.keys()
    key = 'Xulian_2025'
    prefix_list = {'Watanabe_2018':'W18',
                   'Ingle_2016':'I16',
                   'Lombardo_2018':'L18',
                   'Mulpuru_2021':'M21',
                   'Drug3D_database':'D3D',
                   'Kunal_2024':'K24',
                   'Krumpholz_2024':'Kz24',
                   'Xulian_2025':'X25',
                   'Hiroaki_2022':'H22'}
    prefix=f'{prefix_list[key]}040126'
    
    data_savepath = f"/home/rkmvu/Dataset/Fraction_unbound/all_datasets/comparison_datasets/{key}.csv"
    df = df[key]
    df = attach_ids(df, prefix=prefix, id_col=id_col)
    
    if pruning:
        smiles_pruning = SMILES_PRUNING(smiles_list=df[smiles_column_name].tolist(),
                                    labels=df[id_col],
                                    # filters=[('MolWt', '<', 1000)],
                                    smiles_column_name=smiles_column_name,
                                    prefix=''
                                    )
        df_temp = smiles_pruning.get_cleaned_smiles()[['labels', smiles_column_name]]
        df_temp = df_temp.rename(columns={'labels':id_col})
        
        df = pd.merge(df_temp, df.drop(columns=[smiles_column_name]), how='inner', on=id_col) 
    df = calculate_descriptors(df, smiles_column_name)

    # Append as a new sheet
    df.to_csv(data_savepath, index=False)
    
    avg_time =  (time.time()-start)/len(df)
    print(f'Time -> Total: {time.time()-start} sec | average: {avg_time} sec | Items: {len(df)}')