import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KDTree, NearestNeighbors
from typing import Any, List, Dict, Tuple, Union, Set, Callable, Optional

import torch
import torch.nn as nn

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore", module="mordred")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from calc_fps_rm import calculate_similarities_distances

__all__ = [  
    'closest_neighbour_smiles'
    'nearest_neighbours',
    'nearest_neighbours_smiles',
]

def closest_neighbour_smiles(smiles: str, smiles_list: List[str], similarity_metric: str = 'Tanimoto',
                              distance_metric: str = 'Soergel', fingerprint_name: str = 'RDKit') -> Tuple[str, str]:
    max_sim, min_dist = -1e6, 1e6
    best_sim_smi, best_dist_smi = '', ''

    for smi in smiles_list:
        sims, dists = calculate_similarities_distances(smiles, smi, fingerprint_name)
        sim = sims.get(similarity_metric)
        dist = dists.get(distance_metric)

        if sim is None or dist is None:
            continue
        if sim > max_sim:
            best_sim_smi, max_sim = smi, sim
        if dist < min_dist:
            best_dist_smi, min_dist = smi, dist

    return best_sim_smi, best_dist_smi, max_sim, min_dist       
        
def nearest_neighbours_smiles(model: (nn.Module | RandomForestRegressor),
                       X_train: (pd.DataFrame | torch.Tensor),
                       X_test: (pd.DataFrame | torch.Tensor),
                       y_train: (pd.DataFrame | torch.Tensor),
                       y_test: (pd.DataFrame | torch.Tensor),
                       smiles_list_test: Any,
                       smiles_list_train: Any,
                       n_neighbors: int = 1,
                       check_fingerprint: bool = False,
                       fingerprint_names: Any = ['RDKit', 'Pattern', 'TopologicalTorsion', 'MACCSKeys', 'Morgan', 'AtomPair'],
                       distance_metric: str = 'Euclidean',
                       similarity_metric: str = 'Tanimoto',
                       device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                       ) -> Tuple[List[str], List[str], pd.DataFrame]:
    
    predicted_rr = []
    n_neighbour_original = []
    n_neighbour_predicted = []
    original_rr = []
    
    rdkit_fp = []
    pattern_fp = []
    topologicaltorsion_fp = []
    maccskeys_fp = []
    morgan_fp = []
    atompair_fp = []
    fp_lists = [rdkit_fp, pattern_fp, topologicaltorsion_fp, maccskeys_fp, morgan_fp, atompair_fp]

    smi_list_rr_then_train = []
    smi_list_rr = []
    smi_list_train = []
    legends = []
    fps_result = defaultdict(list)

    for i in tqdm(range(X_test.shape[0])):
                
        if isinstance(model, nn.Module):
            # Build the KDTree
            nn_tree = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')
            nn_tree.fit(X_train)
            
            target_point = X_test[i]
            ind = nn_tree.kneighbors(X = target_point.reshape(1, -1), n_neighbors=n_neighbors, return_distance=False)
            nn_idx = ind.item()
            nearest_point = X_train[nn_idx]
            
            model, target_point, nearest_point = model.to(device), target_point.to(device), nearest_point.to(device)
            rr_point_pred_val = model(target_point)
            near_pts_pred_val = model(nearest_point.reshape(*target_point.shape))
            rr_point_pred_val, near_pts_pred_val = rr_point_pred_val.cpu().detach().numpy(), near_pts_pred_val.cpu().detach().numpy()
        
        elif isinstance(model, (RandomForestClassifier, RandomForestRegressor, SVC, SVR)):
            
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)
                # y_train = pd.DataFrame(y_train)
            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test)
                # y_test = pd.DataFrame(y_test)
            
            # Build the KDTree
            target_point = X_test.iloc[i].values.reshape(1, -1)
            nn_tree = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(X_train)
            target_point = pd.DataFrame(target_point, columns=X_train.columns)
            
            ind = nn_tree.kneighbors(X = target_point, n_neighbors=n_neighbors, return_distance=False)
            nn_idx = ind.item()
            nearest_point = X_train.iloc[nn_idx].values.reshape(1, -1)
            nearest_point = pd.DataFrame(nearest_point, columns=X_train.columns)
            
            rr_point_pred_val = model.predict(target_point)
            near_pts_pred_val = model.predict(nearest_point)
            
        else:
            raise TypeError( 'Model type unknown. Expected a neural network or random forest model.')
            
        if isinstance(y_test, pd.core.series.Series):
            y_test = y_test.to_numpy()
        if isinstance(y_train, pd.core.series.Series):
            y_train = y_train.to_numpy()  
            
        predicted_rr.append(round(rr_point_pred_val.item(), 4))
        n_neighbour_predicted.append(round(near_pts_pred_val.item(), 4))
        original_rr.append(round(y_test[i].item(), 4))
        n_neighbour_original.append(round(y_train[nn_idx].item(), 4))
        
        smi_list_rr_then_train.append(smiles_list_test[i])
        smi_list_rr.append(smiles_list_test[i])
        legends.append(f'T:{original_rr[i]}-P:{predicted_rr[i]}(In)')
        
        smi_list_rr_then_train.append(smiles_list_train[nn_idx])
        smi_list_train.append(smiles_list_train[nn_idx])
        legends.append(f'T:{n_neighbour_original[i]}-P:{n_neighbour_predicted[i]}(Tr)')
        
        fps_dict = OrderedDict({})
        
        if check_fingerprint:
            for name in fingerprint_names:
                smiles_sim, _ = closest_neighbour_smiles(smiles_list_test[i],
                                                        smiles_list_train,
                                                        fingerprint_name=name,
                                                        distance_metric=distance_metric,
                                                        similarity_metric=similarity_metric
                                                        )
                smi_list_rr_then_train.append(smiles_sim)
                
                idx = list(smiles_list_train).index(smiles_sim)
                
                if isinstance(X_train, pd.DataFrame):
                    smiles_sim_point = X_train.iloc[idx].values.reshape(1, -1)
                    smiles_sim_point = pd.DataFrame(smiles_sim_point, columns=X_train.columns)
                else:
                    smiles_sim_point = X_train[idx].reshape(1, -1)
                    # smiles_sim_point = pd.DataFrame(smiles_sim_point, columns=[f'dim-{i}' for i in range(X_train.shape[1])])
                    
                if isinstance(model, nn.Module):
                    pred_val = model(smiles_sim_point.to(device)).item()
                elif isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
                    pred_val = model.predict(smiles_sim_point).item()
                else:
                    raise TypeError( 'Model type unknown. Expected a neural network or random forest model.')
                
                true_val = y_train[idx].item()
                legends.append(f'T:{round(true_val, 4)}-P:{round(pred_val, 4)}({name})')
                
                fps_dict[name] = smiles_sim
                
            for key, value in fps_dict.items():
                fps_result[key].append(value)

    df = pd.DataFrame({'SMILES': smi_list_rr,
                       'Nearest SMILES': smi_list_train,
                       'Target Original': original_rr,
                       'Target Predicted': predicted_rr,
                       'NN Original': n_neighbour_original,
                       'NN Predicted': n_neighbour_predicted,
                       **fps_result
                       })
        
    
    return smi_list_rr_then_train, legends, df


# def nearest_neighbours_smiles(
#     model: (nn.Module | RandomForestRegressor),
#     X_train: (pd.DataFrame | torch.Tensor),
#     X_test: (pd.DataFrame | torch.Tensor),
#     y_train: (pd.DataFrame | torch.Tensor),
#     y_test: (pd.DataFrame | torch.Tensor),
#     smiles_list_rr: Any,
#     smiles_list_train: Any,
#     n_neighbors: int = 1,
#     check_fingerprint: bool = False,
#     fingerprint_names: Any = ['RDKit', 'Pattern', 'TopologicalTorsion', 'MACCSKeys', 'Morgan', 'AtomPair'],
#     distance_metric: str = 'Euclidean',
#     similarity_metric: str = 'Tanimoto',
#     device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ) -> Tuple[List[str], List[str], pd.DataFrame]:

#     # --- Convert inputs ---
#     if isinstance(y_test, pd.Series): 
#         y_test = y_test.to_numpy()
#     if isinstance(y_train, pd.Series): 
#         y_train = y_train.to_numpy()

#     if isinstance(model, (RandomForestClassifier, RandomForestRegressor, SVC, SVR)):
#         if not isinstance(X_train, pd.DataFrame):
#             X_train = pd.DataFrame(X_train)
#         if not isinstance(X_test, pd.DataFrame):
#             X_test = pd.DataFrame(X_test, columns=X_train.columns)

#     # --- Build KDTree once ---
#     nn_tree = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(X_train)

#     # --- Query all test points at once ---
#     distances, indices = nn_tree.kneighbors(X_test, n_neighbors=n_neighbors, return_distance=True)

#     # --- Collect nearest neighbors ---
#     nn_indices = indices[:, 0]   # take first neighbor
#     nearest_points = (X_train.iloc[nn_indices].values if isinstance(X_train, pd.DataFrame) else X_train[nn_indices])
#     target_points = (X_test.values if isinstance(X_test, pd.DataFrame) else X_test)

#     # --- Predict in batch ---
#     if isinstance(model, nn.Module):
#         model = model.to(device)
#         target_tensor = torch.tensor(target_points, dtype=torch.float).to(device)
#         neighbor_tensor = torch.tensor(nearest_points, dtype=torch.float).to(device)

#         rr_point_pred_val = model(target_tensor).cpu().detach().numpy()
#         near_pts_pred_val = model(neighbor_tensor).cpu().detach().numpy()

#     else:  # Sklearn model
#         rr_point_pred_val = model.predict(target_points)
#         near_pts_pred_val = model.predict(nearest_points)

#     # --- Round + Collect results ---
#     predicted_rr = np.round(rr_point_pred_val.flatten(), 4).tolist()
#     n_neighbour_predicted = np.round(near_pts_pred_val.flatten(), 4).tolist()
#     original_rr = np.round(y_test.flatten(), 4).tolist()
#     n_neighbour_original = np.round(y_train[nn_indices].flatten(), 4).tolist()

#     # --- Build SMILES + Legends ---
#     smi_list_rr = [smiles_list_rr[i] for i in range(len(X_test))]
#     smi_list_train = [smiles_list_train[idx] for idx in nn_indices]

#     smi_list_rr_then_train = []
#     legends = []
#     for i in range(len(smi_list_rr)):
#         smi_list_rr_then_train.append(smi_list_rr[i])
#         legends.append(f'T:{original_rr[i]}-P:{predicted_rr[i]}(In)')

#         smi_list_rr_then_train.append(smi_list_train[i])
#         legends.append(f'T:{n_neighbour_original[i]}-P:{n_neighbour_predicted[i]}(Tr)')

#     fps_result = defaultdict(list)

#     # --- Fingerprint similarity (still needs loop) ---
#     if check_fingerprint:
#         for i, smi in enumerate(smi_list_rr):
#             fps_dict = OrderedDict({})
#             for name in fingerprint_names:
#                 smiles_sim, _ = closest_neighbour_smiles(
#                     smi, smiles_list_train,
#                     fingerprint_name=name,
#                     distance_metric=distance_metric,
#                     similarity_metric=similarity_metric
#                 )
#                 smi_list_rr_then_train.append(smiles_sim)
#                 idx = list(smiles_list_train).index(smiles_sim)

#                 sim_point = (X_train.iloc[idx].values.reshape(1, -1) 
#                              if isinstance(X_train, pd.DataFrame) 
#                              else X_train[idx].reshape(1, -1))

#                 if isinstance(model, nn.Module):
#                     pred_val = model(torch.tensor(sim_point, dtype=torch.float).to(device)).item()
#                 else:
#                     pred_val = model.predict(sim_point).item()

#                 true_val = y_train[idx].item()
#                 legends.append(f'T:{round(true_val,4)}-P:{round(pred_val,4)}({name})')
#                 fps_dict[name] = smiles_sim

#             for key, value in fps_dict.items():
#                 fps_result[key].append(value)

#     # --- Build DataFrame ---
#     df = pd.DataFrame({
#         'SMILES': smi_list_rr,
#         'Nearest SMILES': smi_list_train,
#         'Target Original': original_rr,
#         'Target Predicted': predicted_rr,
#         'NN Original': n_neighbour_original,
#         'NN Predicted': n_neighbour_predicted,
#         **fps_result
#     })

#     return smi_list_rr_then_train, legends, df


def nearest_neighbours(X_train, X_query, n_neighbors=1):
    
    nn_tree = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')
    nn_tree.fit(X_train)
    distances, indices = nn_tree.kneighbors(X_query, n_neighbors=n_neighbors)
    indices = indices[:, 0]
    
    return indices, distances, nn_tree

