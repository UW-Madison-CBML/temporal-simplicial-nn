"""Adapted DyGLib data utilities for temporal_HTGNN."""

from __future__ import annotations

import copy
import os
from typing import Tuple

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from config import DATA_ROOT
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from tgb.utils.info import PROJ_DIR


class Data:
  def __init__(
      self,
      src_node_ids: np.ndarray,
      dst_node_ids: np.ndarray,
      node_interact_times: np.ndarray,
      edge_ids: np.ndarray,
      labels: np.ndarray,
      interact_types: np.ndarray | None = None,
      node_label_times: np.ndarray | None = None,
  ):
    self.src_node_ids = src_node_ids
    self.dst_node_ids = dst_node_ids
    self.node_interact_times = node_interact_times
    self.edge_ids = edge_ids
    self.labels = labels
    self.interact_types = interact_types
    self.node_label_times = node_label_times
    self.num_interactions = len(src_node_ids)
    self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
    self.num_unique_nodes = len(self.unique_node_ids)


class CustomizedDataset(Dataset):
  def __init__(self, indices_list: list):
    super().__init__()
    self.indices_list = indices_list

  def __getitem__(self, idx: int):
    return self.indices_list[idx]

  def __len__(self):
    return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
  return DataLoader(
      dataset=CustomizedDataset(indices_list=indices_list),
      batch_size=batch_size,
      shuffle=shuffle,
      drop_last=False,
  )


def _tgb_root_arg(data_root: str) -> str:
  data_root = os.path.abspath(data_root)
  proj_dir = os.path.abspath(PROJ_DIR)
  return os.path.relpath(data_root, proj_dir)


def _build_node_classification_arrays(
    dataset_name: str,
    data_root: str,
    cache_dir: str,
) -> Tuple[np.ndarray, np.ndarray, Data, Data, Data, Data, str, int]:
  tgb_root = _tgb_root_arg(data_root)
  dataset = PyGNodePropPredDataset(name=dataset_name, root=tgb_root, download=True)
  data = dataset.dataset.full_data

  src_node_ids = data["sources"].astype(np.longlong)
  dst_node_ids = data["destinations"].astype(np.longlong)
  node_interact_times = data["timestamps"].astype(np.float64)
  edge_ids = data["edge_idxs"].astype(np.longlong)
  edge_raw_features = data["edge_feat"].astype(np.float64)
  if len(edge_raw_features.shape) == 1:
    edge_raw_features = edge_raw_features[:, np.newaxis]

  num_edges = edge_raw_features.shape[0]
  num_nodes = len(set(src_node_ids) | set(dst_node_ids))

  if edge_ids.min() == 1:
    edge_ids = edge_ids - 1

  train_mask = dataset.train_mask.numpy()
  val_mask = dataset.val_mask.numpy()
  test_mask = dataset.test_mask.numpy()
  eval_metric_name = dataset.eval_metric
  num_classes = dataset.num_classes

  labels = np.zeros((num_edges, num_classes))
  interact_types = np.array(["just_update"] * num_edges)
  node_label_times = copy.deepcopy(node_interact_times)
  label_dict = dataset.dataset.label_dict

  converted_label_dict = {}
  for node_label_time in label_dict.keys():
    for src_node_id in label_dict[node_label_time].keys():
      converted_label_dict[(node_label_time, src_node_id)] = label_dict[node_label_time][src_node_id]

  os.makedirs(cache_dir, exist_ok=True)
  cache_path = os.path.join(cache_dir, f"{dataset_name}.npy")
  if os.path.exists(cache_path):
    labeled_node_interaction_indices = np.load(cache_path, allow_pickle=True).tolist()
  else:
    labeled_node_interaction_indices = {}
    for node_label_time, src_node_id in tqdm(
        converted_label_dict.keys(), desc=f"preprocess {dataset_name}"
    ):
      mask = (src_node_ids == src_node_id) & (node_interact_times <= node_label_time)
      if len(edge_ids[mask]) > 0:
        nodes_most_recent_interaction_idx = edge_ids[mask][-1]
      else:
        nodes_most_recent_interaction_idx = 0
      labeled_node_interaction_indices[(node_label_time, src_node_id)] = nodes_most_recent_interaction_idx
    np.save(cache_path, labeled_node_interaction_indices)

  min_val_time = node_interact_times[val_mask].min()
  min_test_time = node_interact_times[test_mask].min()

  for node_label_time, src_node_id in converted_label_dict.keys():
    interaction_idx = labeled_node_interaction_indices[(node_label_time, src_node_id)]
    labels[interaction_idx] = converted_label_dict[(node_label_time, src_node_id)]
    node_label_times[interaction_idx] = node_label_time
    if min_val_time <= node_label_time < min_test_time:
      interact_types[interaction_idx] = "validate"
    elif node_label_time >= min_test_time:
      interact_types[interaction_idx] = "test"
    else:
      interact_types[interaction_idx] = "train"

  src_node_ids = src_node_ids + 1
  dst_node_ids = dst_node_ids + 1
  edge_ids = edge_ids + 1

  if "node_feat" not in data:
    node_raw_features = np.zeros((num_nodes + 1, 1))
  else:
    node_raw_features = data["node_feat"].astype(np.float64)
    if len(node_raw_features.shape) == 1:
      node_raw_features = node_raw_features[:, np.newaxis]

  node_raw_features = np.vstack([np.zeros(node_raw_features.shape[1])[np.newaxis, :], node_raw_features])
  edge_raw_features = np.vstack([np.zeros(edge_raw_features.shape[1])[np.newaxis, :], edge_raw_features])

  full_data = Data(
      src_node_ids, dst_node_ids, node_interact_times, edge_ids,
      labels, interact_types, node_label_times,
  )
  train_data = Data(
      src_node_ids[train_mask], dst_node_ids[train_mask], node_interact_times[train_mask],
      edge_ids[train_mask], labels[train_mask], interact_types[train_mask], node_label_times[train_mask],
  )
  val_data = Data(
      src_node_ids[val_mask], dst_node_ids[val_mask], node_interact_times[val_mask],
      edge_ids[val_mask], labels[val_mask], interact_types[val_mask], node_label_times[val_mask],
  )
  test_data = Data(
      src_node_ids[test_mask], dst_node_ids[test_mask], node_interact_times[test_mask],
      edge_ids[test_mask], labels[test_mask], interact_types[test_mask], node_label_times[test_mask],
  )

  return (
      node_raw_features,
      edge_raw_features,
      full_data,
      train_data,
      val_data,
      test_data,
      eval_metric_name,
      num_classes,
  )


def get_node_classification_tgb_data(
    dataset_name: str,
    data_root: str = DATA_ROOT,
    cache_dir: str | None = None,
):
  if cache_dir is None:
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "saved_labeled_node_interaction_indices")
  cache_dir = os.path.abspath(cache_dir)
  data_root = os.path.abspath(data_root)
  os.makedirs(data_root, exist_ok=True)
  return _build_node_classification_arrays(dataset_name, data_root, cache_dir)
