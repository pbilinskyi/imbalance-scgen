import os
import logging

import numpy as np

from experiment_leave_one_out import train_leave_one_out
from utils import get_adata, plot_cell_type_distribution


logging.config.fileConfig(os.path.join('config', 'logging.conf'))
logger = logging.getLogger('root')


def balance_mixed_under_over_sample(adata,
                                    n,
                                    class_key='cell_type',
                                    verbose=False):
    """
        Perform balancing of classes.
    """
    class_name_count = adata.obs.groupby(class_key).size().to_dict()
    class_names, class_counts = list(class_name_count.keys()), list(class_name_count.values())
    n_total = np.sum(class_counts)
    # n_min = np.min(class_counts)
    # n_max = np.max(class_counts)

    if verbose:
        print("Before balancing:")
        for class_name in class_names:
            n_class = class_name_count[class_name]
            print('{:<12} | {:>5} instances, {:,.0%}'.format(class_name, n_class, n_class/n_total))    

    indexes_classes_balanced = []
    for class_name in class_names:
        index_cls = adata[adata.obs[class_key] == class_name].obs.index.values
        if class_name_count[class_name] >= n:  # if class larger then n, then undersample
            index_balanced = np.random.choice(index_cls, n)
        elif class_name_count[class_name] < n:  # if class smaller then n - oversample
            n_add = n - len(index_cls)
            index_add = np.random.choice(index_cls, n_add)
            index_balanced = np.concatenate([index_cls, index_add])
        indexes_classes_balanced.append(index_balanced)

    index_balanced = np.concatenate(indexes_classes_balanced)
    adata_balanced = adata[index_balanced].copy()
    n_total = adata_balanced.obs.shape[0]

    if verbose:
        print("\nAfter balancing:")
        for class_name in class_names:
            n_class = np.sum(adata_balanced.obs[class_key] == class_name)
            print('{:<12} | {:>5} instances, {:,.0%}'.format(class_name, n_class, n_class/n_total))

    return adata_balanced, index_balanced


def balance_oversample(adata,
                     balance_rate_threshold=1,
                     class_key='cell_type',
                     verbose=False):
    """
        Perform balancing of classes.
    """
    # np.random.seed(43)

    class_name_count = adata.obs.groupby(class_key).size().to_dict()
    class_names, class_counts = list(class_name_count.keys()), list(class_name_count.values())
    n_total = np.sum(class_counts)
    n_max = np.max(class_counts)
    n_size_oversampled = int(n_max*balance_rate_threshold)

    if verbose:
        print("Before balancing:")
        for class_name in class_names:
            n_class = class_name_count[class_name]
            print('{:<12} | {:>5} instances, {:,.0%}'.format(class_name, n_class, n_class/n_total))    

    indexes_classes_balanced = []
    for class_name in class_names:
        index_cls = adata[adata.obs[class_key] == class_name].obs.index.values
        if class_name_count[class_name] < n_size_oversampled:
            n_add = n_size_oversampled - len(index_cls)   # for classes with less instances that we need, add more instances by sampling
            index_add = np.random.choice(index_cls, n_add)
            index_balanced = np.concatenate([index_cls, index_add])
        else:
            index_balanced = index_cls
        indexes_classes_balanced.append(index_balanced)

    index_balanced = np.concatenate(indexes_classes_balanced)
    adata_balanced = adata[index_balanced].copy()
    n_total = adata_balanced.obs.shape[0]

    if verbose:
        print("\nAfter balancing:")
        for class_name in class_names:
            n_class = np.sum(adata_balanced.obs[class_key] == class_name)
            print('{:<12} | {:>5} instances, {:,.0%}'.format(class_name, n_class, n_class/n_total))

    return adata_balanced, index_balanced


def balance_undersample(adata,
                        balance_rate_threshold=1,
                        class_key='cell_type',
                        verbose=False):
    """
        Perform balancing of classes.
    """
    # np.random.seed(43)

    class_name_count = adata.obs.groupby(class_key).size().to_dict()
    class_names, class_counts = list(class_name_count.keys()), list(class_name_count.values())
    n_total = np.sum(class_counts)
    n_min = np.min(class_counts)
    n_size_undersampled = int(n_min/balance_rate_threshold)

    if verbose:
        print("Before balancing:")
        for class_name in class_names:
            n_class = class_name_count[class_name]
            print('{:<12} | {:>5} instances, {:,.0%}'.format(class_name, n_class, n_class/n_total))    

    indexes_classes_balanced = []
    for class_name in class_names:
        index_cls = adata[adata.obs[class_key] == class_name].obs.index.values
        if class_name_count[class_name] > n_size_undersampled:
            sample_size = n_size_undersampled
            index_balanced = np.random.choice(index_cls, sample_size)
        else:
            index_balanced = index_cls
        indexes_classes_balanced.append(index_balanced)

    index_balanced = np.concatenate(indexes_classes_balanced)
    adata_balanced = adata[index_balanced].copy()
    n_total = adata_balanced.obs.shape[0]

    if verbose:
        print("\nAfter balancing:")
        for class_name in class_names:
            n_class = np.sum(adata_balanced.obs[class_key] == class_name)
            print('{:<12} | {:>5} instances, {:,.0%}'.format(class_name, n_class, n_class/n_total))

    return adata_balanced, index_balanced


def balance_control_and_stimulated(adata, balance_func, **kwargs):
    adata_balanced_control, _ = balance_func(adata[adata.obs['condition'] == 'control'], **kwargs)
    adata_balanced_stimulated, _ = balance_func(adata[adata.obs['condition'] == 'stimulated'], **kwargs)
    adata_balanced = adata_balanced_control.concatenate(adata_balanced_stimulated)
    return adata_balanced


def run_balanced_experiment(method, results_dir, iteration, **balancing_kwargs):
    if method == 'undersampling':
        balance_func = balance_undersample
    elif method == 'oversampling':
        balance_func = balance_oversample
    elif method == 'mixed':
        balance_func = balance_mixed_under_over_sample

    np.random.seed(43)

    adata = get_adata('train_kang')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    adata_balanced = balance_control_and_stimulated(adata, balance_func=balance_func, **balancing_kwargs)
    plot_cell_type_distribution(adata=adata_balanced, save_to=os.path.join(results_dir, 'dataset.png'))

    validation_cell_types = ['NK', 'Dendritic', 'FCGR3A+Mono', 'CD4T']
    for cell_type in validation_cell_types:
        logging.info(f'\t\t\t\t Leaving out cell type {cell_type}')
        train_leave_one_out(cell_type=cell_type,
                            adata=adata_balanced,
                            save_to_dir=os.path.join(results_dir, f'leave_out_{cell_type}')
                            )


def run():

    iteration = 2
    logging.info(f'Iteration {iteration}')

    method = 'undersampling'
    logger.info('\t[STARTED] Experiment %s' % method)
    for balance_rate_threshold in [.2, .4, .5, .6, .8, 1]:
        logger.info(f'\t\tFor balance_rate_threshold = {balance_rate_threshold}')
        results_dir = os.path.join('saved_models', f'experiment_{method}', f'balance_rate_threshold___{balance_rate_threshold}', f'Iteration {iteration}')
        run_balanced_experiment(method, results_dir, iteration, balance_rate_threshold=balance_rate_threshold)

    method = 'oversampling'
    logger.info('\t[STARTED] Experiment %s' % method)
    for balance_rate_threshold in [.2, .4, .5, .6, .8, 1]:
        logger.info(f'\t\tFor balance_rate_threshold = {balance_rate_threshold}')
        results_dir = os.path.join('saved_models', f'experiment_{method}', f'balance_rate_threshold___{balance_rate_threshold}', f'Iteration {iteration}')
        run_balanced_experiment(method, results_dir, iteration, balance_rate_threshold=balance_rate_threshold)

    method = 'mixed'
    logger.info('\t[STARTED] Experiment %s' % method)
    for class_size in [600, 800, 1000, 1200, 1400, 1800, 2000]:
        logger.info(f'\t\tFor class_size = {class_size}')
        results_dir = os.path.join('saved_models', f'experiment_{method}', f'class_size___{class_size}', f'Iteration {iteration}')
        run_balanced_experiment(method, results_dir, iteration, n=class_size)


if __name__ == '__main__':
    logger.info('-'*70)
    logger.info('\n'*5)
    logger.info('[STARTED] experiments_balance.py launched.')
    run()
