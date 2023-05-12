import os
import multiprocessing as mp

import scanpy as sc

from scgen_code.scgen._scgen import SCGEN
# import warnings
# warnings.filterwarnings("ignore")


cell_types = ['NK', 'Dendritic', 'CD4T', 'B', 'FCGR3A+Mono', 'CD14+Mono', 'CD8T']


def train_leave_one_out(cell_type):
    print(f'Leave-one-out experiment, leaving out cell type {cell_type}')
    # Read the dataset
    dataset = sc.read(os.path.join('data','train_kang.h5ad'),
                      backup_url='https://drive.google.com/uc?id=1r87vhoLLq6PXAYdmyyd89zG90eJOFYLk')

    # prepare train and evaluation datasets
    mask_is_cell_type = dataset.obs["cell_type"] == cell_type
    train_adata = dataset[~(mask_is_cell_type & (dataset.obs["condition"] == "stimulated"))]
    train_adata = train_adata.copy()
    # eval_adata_control = dataset[mask_is_cell_type & (dataset.obs["condition"] == "control")]
    # eval_adata_stimulated = dataset[mask_is_cell_type & (dataset.obs["condition"] == "stimulated")]

    # setup data
    SCGEN.setup_anndata(train_adata, batch_key="condition", labels_key="cell_type")

    # create and save initial model
    model = SCGEN(train_adata)

    # train model
    model.train(
        max_epochs=100,
        batch_size=32,
        early_stopping=True,
        early_stopping_patience=25
    )

    # save trained model
    save_dir = f"saved_models/leave_one_out/leave_out_{cell_type}"
    model.save(save_dir, overwrite=True)


def run_experiment():
    f = train_leave_one_out
    for cell_type in cell_types:
        f(cell_type)
    # inputs = cell_types
    # with mp.Pool(len(inputs)) as p:
    #     outputs = p.map(f, inputs)


if __name__ == '__main__':
    run_experiment()
