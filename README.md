# VBayesMM: Variational Bayesian microbiome multiomics

## Publication
Tung Dang, Artem Lysenko and Tatsuhiko Tsunoda. "VBayesMM: Variational Bayesian neural network to prioritize important relationships of high-dimensional microbiome multiomics data" bioRxiv (2024): 2024-10 | [https://doi.org/10.3389/fninf.2023.1266713](https://doi.org/10.1101/2023.08.18.553796)


## The workflow of the algorithm

<img src="VBayesMM_method.png" width="1000" height="500">

## Directory structure

### Data

- Please follow instructions in the [Haddad_osa github repo](https://github.com/knightlab-analyses/haddad_osa/) to get OTU tables, metabolome and microbiome phylogenetic tree for dataset A.
- Please follow instructions in the [Multiomic-cooccurences github repo](https://github.com/knightlab-analyses/multiomic-cooccurrences) to get OTU tables, metabolome and microbiome phylogenetic tree for dataset B. 
- Please follow instructions in the [Microbiome-metabolome curated data github repo](https://github.com/borenstein-lab/microbiome-metabolome-curated-data/wiki) to get OTU tables, metabolome and microbiome phylogenetic tree for datasets C and D. 

### Source code

All of the code is in the ```src/``` folder, you can use to re-make the analyses in the paper:

- ```tensorflow/VBayesMM.py```: file contains Python codes for VBayesMM method for TensorFlow User.
- ```pytorch/VBayesMM.py```: file contains Python codes for VBayesMM method for PyTorch User.

If you have any problem, please contact me via email: dangthanhtung91@vn-bml.com
