import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from tgb.nodeproppred.evaluate import Evaluator as NodeClassificationEvaluator
from dyglib.utils.utils import NeighborSampler
from dyglib.utils.DataLoader import Data

def evaluate_model_node_classification(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                       evaluate_data: Data, eval_stage: str, eval_metric_name: str, evaluator: NodeClassificationEvaluator,
                                       loss_func: nn.Module, num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the node classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param eval_stage: str, specifies the evaluate stage, can be 'val' or 'test'
    :param eval_metric_name: str, name of the evaluation metric
    :param evaluator: NodeClassificationEvaluator, node classification evaluator
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        # store the results for each timeslot, and finally compute the metric for each timeslot
        # dictionary of list, key is the timeslot, value is a list, where each element is a prediction, np.ndarray with shape (num_classes, )
        evaluate_predicts_per_timeslot_dict = defaultdict(list)
        # dictionary of list, key is the timeslot, value is a list, where each element is a label, np.ndarray with shape (num_classes, )
        evaluate_labels_per_timeslot_dict = defaultdict(list)
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels, batch_interact_types, batch_node_label_times = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], \
                evaluate_data.labels[evaluate_data_indices], evaluate_data.interact_types[evaluate_data_indices], \
                evaluate_data.node_label_times[evaluate_data_indices]

            # split the batch data based on interaction types
            # train_idx = torch.tensor(np.where(batch_interact_types == 'train')[0])
            if eval_stage == 'val':
                eval_idx = torch.tensor(np.where(batch_interact_types == 'validate')[0])
                # other_idx = torch.tensor(np.where(batch_interact_types == 'test')[0])
            else:
                assert eval_stage == 'test', f"Wrong setting of eval_stage {eval_stage}!"
                eval_idx = torch.tensor(np.where(batch_interact_types == 'test')[0])
                # other_idx = torch.tensor(np.where(batch_interact_types == 'validate')[0])
            # just_update_idx = torch.tensor(np.where(batch_interact_types == 'just_update')[0])
            # assert len(train_idx) == len(other_idx) == 0 and len(eval_idx) + len(just_update_idx) == len(batch_interact_types), "The data are mixed!"

            # for memory-based models, we should use all the interactions to update memories (including eval_stage and 'just_update'),
            # while other memory-free methods only need to compute on eval_stage
            if model_name in ['JODIE', 'DyRep', 'TGN']:
                # get temporal embedding of source and destination nodes, note that the memories are updated during the forward process
                # two Tensors, with shape (batch_size, output_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            else:
                if len(eval_idx) > 0:
                    if model_name in ['TGAT', 'CAWN', 'TCL']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times,
                                                                              num_neighbors=num_neighbors)

                    elif model_name in ['GraphMixer']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times,
                                                                              num_neighbors=num_neighbors,
                                                                              time_gap=time_gap)
                    elif model_name in ['DyGFormer']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times)
                    else:
                        raise ValueError(f"Wrong value for model_name {model_name}!")
                else:
                    batch_src_node_embeddings = None

            if len(eval_idx) > 0:
                # get predicted probabilities, shape (batch_size, num_classes)
                predicts = model[1](x=batch_src_node_embeddings).squeeze(dim=-1)
                labels = torch.from_numpy(batch_labels).float().to(predicts.device)

                loss = loss_func(input=predicts[eval_idx], target=labels[eval_idx])

                evaluate_losses.append(loss.item())
                # append the predictions and labels to evaluate_predicts_per_timeslot_dict and evaluate_labels_per_timeslot_dict
                for idx in eval_idx:
                    evaluate_predicts_per_timeslot_dict[batch_node_label_times[idx]].append(predicts[idx].softmax(dim=0).cpu().detach().numpy())
                    evaluate_labels_per_timeslot_dict[batch_node_label_times[idx]].append(labels[idx].cpu().detach().numpy())

                evaluate_idx_data_loader_tqdm.set_description(f'{eval_stage} for the {batch_idx + 1}-th batch, loss: {loss.item()}')

        # compute the evaluation metric for each timeslot
        for time_slot in tqdm(evaluate_predicts_per_timeslot_dict):
            time_slot_predictions = np.stack(evaluate_predicts_per_timeslot_dict[time_slot], axis=0)
            time_slot_labels = np.stack(evaluate_labels_per_timeslot_dict[time_slot], axis=0)
            # compute metric
            input_dict = {
                "y_true": time_slot_labels,
                "y_pred": time_slot_predictions,
                "eval_metric": [eval_metric_name],
            }
            evaluate_metrics.append({eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})

    return evaluate_losses, evaluate_metrics

