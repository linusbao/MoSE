import logging
import os.path as osp
import time
from functools import partial

import numpy as np
import torch
import torch_geometric.transforms as T
from numpy.random import default_rng
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import (Actor, GNNBenchmarkDataset, Planetoid,
                                      TUDataset, WebKB, WikipediaNetwork, ZINC, LRGBDataset)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_pyg, load_ogb, set_dataset_attr
from torch_geometric.graphgym.register import register_loader

from graphgps.loader.dataset.aqsol_molecules import AQSOL
from graphgps.loader.dataset.coco_superpixels import COCOSuperpixels
from graphgps.loader.dataset.malnet_tiny import MalNetTiny
from graphgps.loader.dataset.voc_superpixels import VOCSuperpixels
from graphgps.loader.split_generator import (prepare_splits,
                                             set_dataset_splits)
from graphgps.transform.posenc_stats import compute_posenc_stats
from graphgps.transform.task_preprocessing import task_specific_preprocessing
from graphgps.transform.transforms import (pre_transform_in_memory,
                                           typecast_x, concat_x_and_pos,
                                           clip_graphs_to_size)
from torch_geometric.transforms import VirtualNode

from pathlib import Path
import sys
import os
sys.path.append(os.path.join(Path(__file__).parent.parent.parent.parent, 'hombasis-gt','hombasis-bench'))
from data import get_data

sys.path.append(os.path.join(Path(__file__).parent.parent.parent.parent, 'hombasis-gt')) #FLAGother
from image-datasets import add_data

sys.path.append(os.path.join(Path(__file__).parent.parent.parent.parent, 'hombasis-gt', 'pcqm'))
import get_pcqm_data

sys.path.append(os.path.join(Path(__file__).parent.parent.parent.parent, 'hombasis-gt','qm9','data_GraphGym_QM9'))
from CustomDataset import CustomDataset

sys.path.append(os.path.join(Path(__file__).parent.parent.parent.parent, 'hombasis-gt', 'peptides'))#FLAGpep
import get_pep_data #FLAGpep


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: "
                 f"{total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if isinstance(dataset.data.y, list):
            # A special case for ogbg-code2 dataset.
            logging.info(f"  num classes: n/a")
        elif dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info(f"  num classes: (appears to be a regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")
    elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data, 'edge_label'):
        # Edge/link prediction task.
        if hasattr(dataset.data, 'train_edge_label'):
            labels = dataset.data.train_edge_label  # Transductive link task
        else:
            labels = dataset.data.edge_label  # Inductive link task
        if labels.numel() == labels.size(0) and \
                torch.is_floating_point(labels):
            logging.info(f"  num edge classes: (probably a regression task)")
        else:
            logging.info(f"  num edge classes: {len(torch.unique(labels))}")

    ## Show distribution of graph sizes.
    # graph_sizes = [d.num_nodes if hasattr(d, 'num_nodes') else d.x.shape[0]
    #                for d in dataset]
    # hist, bin_edges = np.histogram(np.array(graph_sizes), bins=10)
    # logging.info(f'   Graph size distribution:')
    # logging.info(f'     mean: {np.mean(graph_sizes)}')
    # for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    #     logging.info(
    #         f'     bin {i}: [{start:.2f}, {end:.2f}]: '
    #         f'{hist[i]} ({hist[i] / hist.sum() * 100:.2f}%)'
    #     )


@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'Actor':
            if name != 'none':
                raise ValueError(f"Actor class provides only one dataset.")
            dataset = Actor(dataset_dir)

        elif pyg_dataset_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(dataset_dir, name)

        elif pyg_dataset_id == 'MalNetTiny':
            dataset = preformat_MalNetTiny(dataset_dir, feature_set=name)

        elif pyg_dataset_id == 'Planetoid':
            dataset = Planetoid(dataset_dir, name)

        elif pyg_dataset_id == 'TUDataset':
            dataset = preformat_TUDataset(dataset_dir, name)

        elif pyg_dataset_id == 'WebKB':
            dataset = WebKB(dataset_dir, name)

        elif pyg_dataset_id == 'WikipediaNetwork':
            if name == 'crocodile':
                raise NotImplementedError(f"crocodile not implemented")
            dataset = WikipediaNetwork(dataset_dir, name,
                                       geom_gcn_preprocess=True)

        elif pyg_dataset_id == 'ZINC':
            dataset = preformat_ZINC(dataset_dir, name)
        elif pyg_dataset_id.startswith('ZINC-'):
            postfix = pyg_dataset_id.split('-', 1)[1]
            dataset = preformat_ZINC(dataset_dir, name, postfix)
            
        elif pyg_dataset_id == 'AQSOL':
            dataset = preformat_AQSOL(dataset_dir, name)

        elif pyg_dataset_id == 'VOCSuperpixels':
            dataset = preformat_VOCSuperpixels(dataset_dir, name,
                                               cfg.dataset.slic_compactness)

        elif pyg_dataset_id == 'COCOSuperpixels':
            dataset = preformat_COCOSuperpixels(dataset_dir, name,
                                                cfg.dataset.slic_compactness)

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    elif format == 'OGB':
        if name.startswith('ogbg'):
            dataset = preformat_OGB_Graph(dataset_dir, name.replace('_', '-'))

        elif name.startswith('PCQM4Mv2-'):
            subset = name.split('-', 1)[1]
            dataset = preformat_OGB_PCQM4Mv2(dataset_dir, subset)

        elif name.startswith('PCQM4Mv2+All5'):
            subset = name.split('-', 1)[1]
            dataset = preformat_OGB_PCQM4Mv2(dataset_dir, subset, 'All5')
            
        elif name.startswith('PCQM4Mv2+Spasm'):
            subset = name.split('-', 1)[1]
            dataset = preformat_OGB_PCQM4Mv2(dataset_dir, subset, 'Spasm')

        elif name.startswith('peptides-'):
            dataset = preformat_Peptides(dataset_dir, name)

        ### Link prediction datasets.
        elif name.startswith('ogbl-'):
            # GraphGym default loader.
            dataset = load_ogb(name, dataset_dir)
            # OGB link prediction datasets are binary classification tasks,
            # however the default loader creates float labels => convert to int.
            def convert_to_int(ds, prop):
                tmp = getattr(ds.data, prop).int()
                set_dataset_attr(ds, prop, tmp, len(tmp))
            convert_to_int(dataset, 'train_edge_label')
            convert_to_int(dataset, 'val_edge_label')
            convert_to_int(dataset, 'test_edge_label')

        elif name.startswith('PCQM4Mv2Contact-'):
            dataset = preformat_PCQM4Mv2Contact(dataset_dir, name)

    elif format.startswith('EmJ-'):#FLAGsyn
        emj_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, "QM9-GraphHC") if emj_dataset_id.startswith('QM9') else osp.join(dataset_dir, "SYNTH-") #I am loading the GraphHC dataset for everything in QM9
        if emj_dataset_id == 'QM9':
            dataset = preformat_QM9(dataset_dir, name)
        elif emj_dataset_id.startswith('QM9-'):
            postfix = emj_dataset_id.split('-', 1)[1]
            dataset = preformat_QM9(dataset_dir,name, postfix)

        elif emj_dataset_id == 'Synth':
            dataset = preformat_Synth(dataset_dir,name)
        elif emj_dataset_id.startswith('Synth-'):
            postfix = emj_dataset_id.split('-', 1)[1]
            dataset = preformat_Synth(dataset_dir,name, postfix)

        else:
            raise ValueError(f"Unsupported OGB(-derived) dataset: {name}")
    else:
        raise ValueError(f"Unknown data format: {format}")

    pre_transform_in_memory(dataset, partial(task_specific_preprocessing, cfg=cfg))

    log_loaded_dataset(dataset, format, name)

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable:
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: "
                     f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        pre_transform_in_memory(dataset,
                                partial(compute_posenc_stats,
                                        pe_types=pe_enabled_list,
                                        is_undirected=is_undirected,
                                        cfg=cfg),
                                show_progress=True
                                )
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)

    # Precompute in-degree histogram if needed for PNAConv.
    if cfg.gt.layer_type.startswith('PNA') and len(cfg.gt.pna_degrees) == 0:
        cfg.gt.pna_degrees = compute_indegree_histogram(
            dataset[dataset.data['train_graph_index']])
        # print(f"Indegrees: {cfg.gt.pna_degrees}")
        # print(f"Avg:{np.mean(cfg.gt.pna_degrees)}")

    return dataset


def compute_indegree_histogram(dataset):
    """Compute histogram of in-degree of nodes needed for PNAConv.

    Args:
        dataset: PyG Dataset object

    Returns:
        List where i-th value is the number of nodes with in-degree equal to `i`
    """
    from torch_geometric.utils import degree

    deg = torch.zeros(1000, dtype=torch.long)
    max_degree = 0
    for data in dataset:
        d = degree(data.edge_index[1],
                   num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, d.max().item())
        deg += torch.bincount(d, minlength=deg.numel())
    return deg.numpy().tolist()[:max_degree + 1]


def preformat_GNNBenchmarkDataset(dataset_dir, name): #FLAGother
    """Load and preformat datasets from PyG's GNNBenchmarkDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    data_dir = os.path.join(Path(__file__).parent.parent.parent.parent, 'hombasis-gt','image-datasets', 'data')
    homcount_name = None
    if name in ['MNIST', 'CIFAR10']:
        tf_list = [concat_x_and_pos]  # concat pixel value and pos. coordinate
        tf_list.append(partial(typecast_x, type_str='float'))
    elif name in ['PATTERN', 'CLUSTER', 'CSL']:
        tf_list = []
    elif name in ['MNIST-Spasm', 'CIFAR10-Spasm', 'MNIST-All5', 'CIFAR10-All5']:
        tf_list = [concat_x_and_pos]  # concat pixel value and pos. coordinate
        tf_list.append(partial(typecast_x, type_str='float'))
        dataset_name, homcount_name = name.split('-', 1)
        if dataset_name == 'MNIST':
            data_dir = os.path.join(data_dir, 'MNIST')
            if homcount_name == 'Spasm':
                hom_files = ['mnist_c7.json','mnist_c8.json']
            elif homcount_name == 'All5':
                hom_files = ['mnist_v5.json']
        elif dataset_name == 'CIFAR10':
            data_dir = os.path.join(data_dir, 'CIFAR')
            if homcount_name == 'Spasm':
                hom_files = ['cifar_c7.json','cifar_c8.json']
            elif homcount_name == 'All5':
                hom_files = ['cifar_v5.json']
    else:
        raise ValueError(f"Loading dataset '{name}' from "
                         f"GNNBenchmarkDataset is not supported.")

    if name in ['MNIST', 'CIFAR10', 'PATTERN', 'CLUSTER']:
        dataset = join_dataset_splits(
            [GNNBenchmarkDataset(root=dataset_dir, name=name, split=split)
            for split in ['train', 'val', 'test']]
        )
        pre_transform_in_memory(dataset, T.Compose(tf_list))
    elif homcount_name != None:
        dataset = join_dataset_splits(
            [GNNBenchmarkDataset(root=dataset_dir, name=dataset_name, split=split)
            for split in ['train', 'val', 'test']]
        )
        pre_transform_in_memory(dataset, T.Compose(tf_list))
        dataset = add_data.add_hom(hom_files=hom_files,idx_list=[],root=data_dir,dataset=dataset)
    elif name == 'CSL':
        dataset = GNNBenchmarkDataset(root=dataset_dir, name=name)

    return dataset


def preformat_MalNetTiny(dataset_dir, feature_set):
    """Load and preformat Tiny version (5k graphs) of MalNet

    Args:
        dataset_dir: path where to store the cached dataset
        feature_set: select what node features to precompute as MalNet
            originally doesn't have any node nor edge features

    Returns:
        PyG dataset object
    """
    if feature_set in ['none', 'Constant']:
        tf = T.Constant()
    elif feature_set == 'OneHotDegree':
        tf = T.OneHotDegree()
    elif feature_set == 'LocalDegreeProfile':
        tf = T.LocalDegreeProfile()
    else:
        raise ValueError(f"Unexpected transform function: {feature_set}")

    dataset = MalNetTiny(dataset_dir)
    dataset.name = 'MalNetTiny'
    logging.info(f'Computing "{feature_set}" node features for MalNetTiny.')
    pre_transform_in_memory(dataset, tf)

    split_dict = dataset.get_idx_split()
    dataset.split_idxs = [split_dict['train'],
                          split_dict['valid'],
                          split_dict['test']]

    return dataset


def preformat_OGB_Graph(dataset_dir, name):
    """Load and preformat OGB Graph Property Prediction datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific OGB Graph dataset

    Returns:
        PyG dataset object
    """
    dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'valid', 'test']]

    if name == 'ogbg-ppa':
        # ogbg-ppa doesn't have any node features, therefore add zeros but do
        # so dynamically as a 'transform' and not as a cached 'pre-transform'
        # because the dataset is big (~38.5M nodes), already taking ~31GB space
        def add_zeros(data):
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)
            return data
        dataset.transform = add_zeros
    elif name == 'ogbg-code2':
        from graphgps.loader.ogbg_code2_utils import idx2vocab, \
            get_vocab_mapping, augment_edge, encode_y_to_arr
        num_vocab = 5000  # The number of vocabulary used for sequence prediction
        max_seq_len = 5  # The maximum sequence length to predict

        seq_len_list = np.array([len(seq) for seq in dataset.data.y])
        logging.info(f"Target sequences less or equal to {max_seq_len} is "
            f"{np.sum(seq_len_list <= max_seq_len) / len(seq_len_list)}")

        # Building vocabulary for sequence prediction. Only use training data.
        vocab2idx, idx2vocab_local = get_vocab_mapping(
            [dataset.data.y[i] for i in s_dict['train']], num_vocab)
        logging.info(f"Final size of vocabulary is {len(vocab2idx)}")
        idx2vocab.extend(idx2vocab_local)  # Set to global variable to later access in CustomLogger

        # Set the transform function:
        # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
        # encode_y_to_arr: add y_arr to PyG data object, indicating the array repres
        dataset.transform = T.Compose(
            [augment_edge,
             lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len)])

        # Subset graphs to a maximum size (number of nodes) limit.
        pre_transform_in_memory(dataset, partial(clip_graphs_to_size,
                                                 size_limit=1000))

    return dataset


def preformat_OGB_PCQM4Mv2(dataset_dir, name, homcount_type):
    """Load and preformat PCQM4Mv2 from OGB LSC.

    OGB-LSC provides 4 data index splits:
    2 with labeled molecules: 'train', 'valid' meant for training and dev
    2 unlabeled: 'test-dev', 'test-challenge' for the LSC challenge submission

    We will take random 150k from 'train' and make it a validation set and
    use the original 'valid' as our testing set.

    Note: PygPCQM4Mv2Dataset requires rdkit

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of the training set

    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from ogb.lsc import PygPCQM4Mv2Dataset
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2Dataset, '
                      'make sure RDKit is installed.')
        raise e


    dataset = PygPCQM4Mv2Dataset(root=dataset_dir)
    split_idx = dataset.get_idx_split()

    rng = default_rng(seed=42)
    train_idx = rng.permutation(split_idx['train'].numpy())
    train_idx = torch.from_numpy(train_idx)

    # Leave out 150k graphs for a new validation set.
    valid_idx, train_idx = train_idx[:150000], train_idx[150000:]
    if name == 'full':
        split_idxs = [train_idx,  # Subset of original 'train'.
                      valid_idx,  # Subset of original 'train' as validation set.
                      split_idx['valid']  # The original 'valid' as testing set.
                      ]

    elif name == 'subset':
        print('in PCQM subset loader!!')
        # Further subset the training set for faster debugging.
        subset_ratio = 0.1
        subtrain_idx = train_idx[:int(subset_ratio * len(train_idx))]
        subvalid_idx = valid_idx[:50000]
        subtest_idx = split_idx['valid']  # The original 'valid' as testing set.

        dataset = dataset[torch.cat([subtrain_idx, subvalid_idx, subtest_idx])]
        
        print(type(dataset))
        
        data_list = [data for data in dataset]
        print(data_list[0])
        
        print('adding homcounts!!')
        data_dir = os.path.join(Path(__file__).parent.parent.parent.parent, 'hombasis-gt','pcqm', 'data')
        
        if homcount_type == 'All5':
            homcount_file = 'pcqm_v5.json'
        elif homcount_type == 'Spasm':
            homcount_file = 'pcqm_c78.json'
        else:
            raise ValueError(f'Invalid homcount type')
        
        data_list = get_pcqm_data.add_pcqm_hom(homcount_file, data_dir, data_list)

        print('done adding homcounts!!')
        print(data_list[0])

        
        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = dataset.collate(data_list)
        
        print('after coallating!!')
        print(type(dataset.data))
        # assert False
        
        
        n1, n2, n3 = len(subtrain_idx), len(subvalid_idx), len(subtest_idx)
        split_idxs = [list(range(n1)),
                      list(range(n1, n1 + n2)),
                      list(range(n1 + n2, n1 + n2 + n3))]

    elif name == 'inference':
        split_idxs = [split_idx['valid'],  # The original labeled 'valid' set.
                      split_idx['test-dev'],  # Held-out unlabeled test dev.
                      split_idx['test-challenge']  # Held-out challenge test set.
                      ]

        dataset = dataset[torch.cat(split_idxs)]
        data_list = [data for data in dataset]
        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = dataset.collate(data_list)
        n1, n2, n3 = len(split_idxs[0]), len(split_idxs[1]), len(split_idxs[2])
        split_idxs = [list(range(n1)),
                      list(range(n1, n1 + n2)),
                      list(range(n1 + n2, n1 + n2 + n3))]
        # Check prediction targets.
        assert(all([not torch.isnan(dataset[i].y)[0] for i in split_idxs[0]]))
        assert(all([torch.isnan(dataset[i].y)[0] for i in split_idxs[1]]))
        assert(all([torch.isnan(dataset[i].y)[0] for i in split_idxs[2]]))

    else:
        raise ValueError(f'Unexpected OGB PCQM4Mv2 subset choice: {name}')
    dataset.split_idxs = split_idxs
    return dataset


def preformat_PCQM4Mv2Contact(dataset_dir, name):
    """Load PCQM4Mv2-derived molecular contact link prediction dataset.

    Note: This dataset requires RDKit dependency!

    Args:
       dataset_dir: path where to store the cached dataset
       name: the type of dataset split: 'shuffle', 'num-atoms'

    Returns:
       PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary
        from graphgps.loader.dataset.pcqm4mv2_contact import \
            PygPCQM4Mv2ContactDataset, \
            structured_neg_sampling_transform
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2ContactDataset, '
                      'make sure RDKit is installed.')
        raise e

    split_name = name.split('-', 1)[1]
    dataset = PygPCQM4Mv2ContactDataset(dataset_dir, subset='530k')
    # Inductive graph-level split (there is no train/test edge split).
    s_dict = dataset.get_idx_split(split_name)
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    if cfg.dataset.resample_negative:
        dataset.transform = structured_neg_sampling_transform
    return dataset


def preformat_Peptides(dataset_dir, name): #FLAGpep
    """Load Peptides dataset, functional or structural.

    Note: This dataset requires RDKit dependency!

    Args:
        dataset_dir: path where to store the cached dataset
        name: the type of dataset split:
            - 'peptides-functional' (10-task classification)
            - 'peptides-structural' (11-task regression)

    Returns:
        PyG dataset object
    """
    save_dir = os.path.join(Path(__file__).parent.parent.parent.parent, 'datasets')

    dataset_type = name.split('-', 1)[1]

    if '-' in dataset_type:
        dataset_type, hom_type = dataset_type.split('-', 1)
    else:
        hom_type = None

    if dataset_type == 'functional':
        dataset_type = 'peptides-func'
    elif dataset_type == 'structural':
        dataset_type = 'peptides-struct'

    print(f'DATASET NAME : {dataset_type}')
    dataset = join_dataset_splits(
        [LRGBDataset(root=save_dir, name=dataset_type, split=split)
        for split in ['train', 'val', 'test']]
    )

    if hom_type != None:
        data_dir = os.path.join(Path(__file__).parent.parent.parent.parent, 'hombasis-gt','peptides', 'data')
        if hom_type == 'Spasm' or hom_type == 'spasm':
            hom_file = "peptides_c78.json"
        elif hom_type == 'All5' or hom_type == 'all5':
            hom_file = 'peptides_v5c6.json'
        else:
            raise Exception('Specify dataset to load by giving dataset.name as peptides-X-Y where X can be \"functional\" or \"structural\" and Y can be \"Spasm\", \"All5\", or nothing at all.')

        idx_list = [] #assuming using all counts in JSON
        dataset = get_pep_data.add_peptide_hom(hom_file, idx_list, data_dir, dataset)

    return dataset


def preformat_TUDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's TUDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    if name in ['DD', 'NCI1', 'ENZYMES', 'PROTEINS', 'TRIANGLES']:
        func = None
    elif name.startswith('IMDB-') or name == "COLLAB":
        func = T.Constant()
    else:
        raise ValueError(f"Loading dataset '{name}' from "
                         f"TUDataset is not supported.")
    dataset = TUDataset(dataset_dir, name, pre_transform=func)
    return dataset

#EDITED
def preformat_ZINC(dataset_dir, name, postfix=None):
    """Load and preformat ZINC datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of ZINC

    Returns:
        PyG dataset object
    """
    data_dir = os.path.join(Path(__file__).parent.parent.parent.parent, 'hombasis-gt','hombasis-bench', 'data', 'zinc-data')
    dataset = None
    if name not in ['subset', 'full']:
        raise ValueError(f"Unexpected subset choice for ZINC dataset: {name}")
    dataset = join_dataset_splits(
        [ZINC(root=dataset_dir, subset=(name == 'subset'), split=split)
        for split in ['train', 'val', 'test']]
    )

    if postfix != None and "Spasm" in postfix and name == 'subset':
        count_files = ['zinc_with_homs_c7.json', 'zinc_with_homs_c8.json']
        idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 15, 20, 21, 22, 24, 25, 27, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
        sub_file = 'zinc_3to8C_multhom.json'
        if hasattr(cfg.dataset, 'virtual_node') and cfg.dataset.virtual_node == True:
            dataset = get_data.add_zinc_subhom_vn(name='ZINC', hom_files=count_files, idx_list=idx_list, sub_file=sub_file, root=data_dir, dataset=dataset)
        else:
            dataset = get_data.add_zinc_subhom(name='ZINC', hom_files=count_files, idx_list=idx_list, sub_file=sub_file, root=data_dir, dataset=dataset)

    return dataset

def preformat_QM9(dataset_dir,name, postfix=None, de_normalize=False):
    """Load and preformat qm9 datasets from Emily Jin's repo.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    targets = ['MU', 'ALPHA', 'HOMO', 'LUMO', 'GAP', 'R2', 'ZPVE', 'U0', 'U', 'H', 'G', 'Cv', 'OMEGA']
    norms = [0.066513725, 0.012235489, 0.071939046, 0.033730778, 0.033486113, 0.004278493, 0.001330901, 0.004165489, 0.004128926, 0.00409976, 0.004527465, 0.012292586, 0.037467458]
    assert name in targets #So far only implemented the 13 tagets used by Emily
    idx = targets.index(name)
    c = norms[idx] if de_normalize else 1

    load_dir = osp.join(dataset_dir,"processed","joined.pt")

    if postfix in ["GraphHC","NodeHC", None]: #Right now I am just loading Qm9 with node level counts for everything, and then pooling for graphHC, and ignoring for None (requires u to set proper model arch in cfg, the postfix is just for wandb readibility)    
        dataset = torch.load(load_dir)
        dataset.data['y'] = dataset.data['y'][:,idx] / c
    else:
        raise Exception('Dataset not implemented')

        #print(f'unique edge features: {set(dataset.data.edge_attr.tolist())}')#HOW MANY EDGE TYPES ARE THERE
    return dataset

def preformat_Synth(pre_dataset_dir,name, postfix=None): #FLAGsyn
    """Load and preformat Sythetic graph dataset.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """

    if postfix == None:
        feats = 'none'
    else:
        feats = postfix
    print(f'Loading SYNTH with node_feats {feats} and target {name}')

    if feats == 'none' or feats == 'All5':#we load all5 for none node-features, and simply ignore the counts
        dataset_dir = f'{pre_dataset_dir}All5_log10'
    elif feats == 'Spasm':
        dataset_dir = f'{pre_dataset_dir}Spasm'
    elif feats == "Spasm10" or feats == "spasm10": #FLAGfd
        dataset_dir = f'{pre_dataset_dir}Spasm_log10'
    elif feats == "All510" or feats == "all510": #FLAGfd
        dataset_dir = f'{pre_dataset_dir}All5_log10'
    else:
        raise Exception('Choose valid node features: none or Spasm or All5 or Spasm10 or All510') #FLAGfd
    print(f'Loading from directory: {dataset_dir}')

    load_dir = osp.join(dataset_dir,"processed","joined.pt")
    dataset = torch.load(load_dir)

    if feats == 'none':#ignore counts (use torch.emedding nodeEncoder to encoder the one trivial node type)
        dataset.data['x'] = torch.zeros_like(dataset.data['x'], dtype=torch.int32)#x is already set as blank ones tensor, just have to cast to int tensor of zeros for torch.embeddings
    else: #FLAGfd
        dataset.data['x'] = dataset.data['homcounts'] #otherwise, treat the homcounts as the node features (use MLPnodeEncoder to encode counts)

    if name == 'SpecRad': #set y as the appropriate target
        dataset.data['y'] = dataset.data['spectral_radius']
    elif name == 'AvgClust':
        dataset.data['y'] = dataset.data['avg_clustering']
    elif name == "FracDom": #FLAGfd
        dataset.data['y'] = dataset.data['fractional_domination']
    else:
        raise Exception('Choose a valid target: SpecRad or AvgClust or FracDom')

    return dataset

def preformat_AQSOL(dataset_dir):
    """Load and preformat AQSOL datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [AQSOL(root=dataset_dir, split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_VOCSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat VOCSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [VOCSuperpixels(root=dataset_dir, name=name,
                        slic_compactness=slic_compactness,
                        split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_COCOSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat COCOSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [COCOSuperpixels(root=dataset_dir, name=name,
                         slic_compactness=slic_compactness,
                         split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])

    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]
