# MRCGNN: Multi-relational Contrastive Learning Graph Neural Network for Drug-Drug Interaction Event Prediction

This repository contains the source code, datasets, and supplementary material associated with the MRCGNN project for drug-drug interaction (DDI) event prediction.

The repository preserves the directory structure and file paths in the provided project archive. No source-code files, dataset files, or existing paths have been modified.

## Repository Structure

```text
SGDL/
├── README.md
├── requirement.txt
├── Appendix.pdf
│
├── Deng's dataset/
│   ├── 0/
│   │   ├── ddi_training1xiao.csv
│   │   ├── ddi_validation1xiao.csv
│   │   └── ddi_test1xiao.csv
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── drug_listxiao.csv
│   ├── drug_smiles.csv
│   └── newddixiao-1.csv
│
├── Ryu's dataset/
│   ├── 0/
│   │   ├── ddi_training1.csv
│   │   ├── ddi_validation1.csv
│   │   └── ddi_test1.csv
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   └── DDI_event.csv
│
└── codes for MRCGNN/
    ├── main.py
    ├── train.py
    ├── data_preprocess.py
    ├── instantiation.py
    ├── layer.py
    ├── nodeformer.py
    ├── parms_setting.py
    ├── utils.py
    ├── sub.py
    ├── 5timesrun.py
    ├── drugfeature_fromMG.py
    ├── data/
    │   ├── 0/
    │   ├── 1/
    │   ├── 2/
    │   ├── 3/
    │   ├── 4/
    │   ├── DDI_event.csv
    │   ├── drug_brics_results.csv
    │   ├── drug_brics_results_more.csv
    │   ├── drug_listxiao.csv
    │   └── drug_smiles.csv
    └── feature/
        ├── blip2.py
        ├── blip2opt.py
        ├── dataset.py
        ├── feature.py
        ├── gin.py
        ├── nasmodel.py
        ├── nas/
        └── autogllight/
```

The `data/` directory under `codes for MRCGNN/` contains the data files referenced by the preprocessing code through relative paths. The separate `Deng's dataset/` and `Ryu's dataset/` directories are also included in the repository as dataset collections.

## Datasets

Two DDI datasets are included:

- **Deng's dataset**: Includes five data splits (`0`–`4`). Each split contains training, validation, and test files. The dataset also includes drug ID, SMILES, and DDI-related files.
- **Ryu's dataset**: Includes five data splits (`0`–`4`). Each split contains training, validation, and test files, together with the DDI event definition file.

The `codes for MRCGNN/data/` directory contains the data files used by the current preprocessing implementation. In particular, `data_preprocess.py` reads the split files and drug information using paths relative to the `codes for MRCGNN/` working directory.

## Code Overview

The main code directory is `codes for MRCGNN/`.

| File or directory | Description |
|---|---|
| `main.py` | Main program entry point. Loads configuration and data, creates the model, and starts training. |
| `parms_setting.py` | Defines command-line arguments and model/training hyperparameters. |
| `data_preprocess.py` | Loads DDI data, constructs data loaders, and prepares molecular and BRICS features. |
| `instantiation.py` | Creates the MRCGNN model and optimizer. |
| `train.py` | Contains training and evaluation routines. |
| `layer.py` | Contains graph neural network/model layer implementations. |
| `nodeformer.py` | Contains NodeFormer-related graph model components. |
| `utils.py` | Utility functions used by the project. |
| `5timesrun.py` | Launches five runs using split indices `0`–`4`. |
| `feature/` | Contains molecular feature extraction, graph representation, and related components. |
| `data/` | Contains the data files accessed by the preprocessing code. |

## Environment

The provided environment specification is:

- Python 3.7.10
- NumPy 1.18.5
- PyTorch 1.7.1 with CUDA 11.0
- PyTorch Geometric 2.0.0
- NetworkX 2.6.3
- scikit-learn 0.24.2

The full package list and version pins are recorded in [`requirement.txt`](requirement.txt).

### Installation

Create and activate a Python environment compatible with the versions listed above, then install the dependencies according to `requirement.txt`.

```bash
pip install -r requirement.txt
```

**Dependency note:** The supplied `requirement.txt` contains local wheel-file references for `torch-scatter` and `torch-sparse` that point to the original development machine. Those wheel files are not included in this archive. Consequently, the dependency file is preserved as supplied, but the installation command may require those two packages to be installed separately with versions compatible with the local PyTorch, CUDA, Python, and PyG versions.

## Running the Experiments

The scripts are located in `codes for MRCGNN/`. The existing `5timesrun.py` script invokes `main.py` five times with `--zhongzi` set to `0`, `1`, `2`, `3`, and `4`, respectively.

The intended command from the code directory is:

```bash
cd "codes for MRCGNN"
python 5timesrun.py
```

Alternatively, a single split can be selected through the main program's arguments:

```bash
python main.py --out_file test.txt --zhongzi 0
```

The output filename is specified by `--out_file`, and `--zhongzi` selects the split index.

### Important: Current Archive Completeness

The commands above document the entry points and invocation pattern present in the supplied source files. **The uploaded archive does not currently contain every module or feature file referenced by the entry-point code**, so the experiments cannot be confirmed as directly executable from this archive alone.

In particular:

- `main.py` imports `data_preprocess_test2.py` and `train_test2.py`, which are not present in the supplied archive.
- `instantiation.py` imports `MRCGNN` from `layer_test3.py`, which is not present in the supplied archive.
- `data_preprocess.py` loads `feature/drug_emb_feature0.npy` and `feature/drug_emb_feature0_drug_ids.npy`; these files are not present in the supplied archive.

These files have not been recreated or replaced. The original files and paths are left unchanged.

The script `drugfeature_fromMG.py` also contains an absolute path from the original development environment. It is included unchanged; its path-dependent execution is therefore specific to that environment.

## Supplementary Material

`Appendix.pdf` provides the supplementary material associated with the project.

## Citation

If you use this code or the associated datasets in your research, please cite the corresponding MRCGNN paper:

```bibtex
@inproceedings{mrcgnn_aaai2023,
  title     = {Multi-relational Contrastive Learning Graph Neural Network for Drug-drug Interaction Event Prediction},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2023}
}
```

Please verify the author list, volume, issue, and page information against the published paper before using this BibTeX entry in a manuscript.

## Acknowledgement

This repository includes code and components from third-party projects, including AutoGLight and molecular graph/feature extraction implementations. Please refer to the corresponding source files and upstream projects for their licenses and citation requirements.
