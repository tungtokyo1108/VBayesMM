# VBayesMM: Variational Bayesian microbiome multiomics

## Publication
Tung Dang, Artem Lysenko and Tatsuhiko Tsunoda. "VBayesMM: Variational Bayesian neural network to prioritize important relationships of high-dimensional microbiome multiomics data" bioRxiv (2024): 2024-


## The workflow of the algorithm

<img src="VBayesMM_method.png" width="1000" height="500">

## How To Use 

- Import packages 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import biom
from biom import load_table, Table
from scipy.stats import entropy, spearmanr
from scipy.sparse import coo_matrix

import tensorflow as tf
from VBayesMM_v1 import VBayesMM
```

## Directory structure

### Data

- The obstructive sleep apnea (OSA) in mice (dataset A). 16S rRNA gene sequencing-based microbiome and liquid chromatography-tandem mass spectrometry (LC-MS/MS)-based metabolome are obtained from [Haddad_osa github repo](https://github.com/knightlab-analyses/haddad_osa/).
- The high-fat diet (HFD) in a murine model (dataset B). 16S rRNA gene sequencing-based microbiome and liquid chromatography-tandem mass spectrometry (LC-MS/MS)-based metabolome are obtained from [Multiomic-cooccurences github repo](https://github.com/knightlab-analyses/multiomic-cooccurrences). 
- The astric cancer (GC) patients (dataset C) and colorectal cancer (CRC) patients from stage 0 to stage 4 (dataset D). Whole-genome shotgun sequencing (WGS) microbiome profiling and capillary electrophoresis time-of-flight mass spectrometry (CE-TOFMS) for metabolomics are obtained from [Microbiome-metabolome curated data github repo](https://github.com/borenstein-lab/microbiome-metabolome-curated-data/wiki).  

### Source code

All of the code is in the ```src/``` folder, you can use to re-make the analyses in the paper:

- ```tensorflow/VBayesMM.py```: file contains Python codes for VBayesMM method for TensorFlow User.
- ```pytorch/VBayesMM.py```: file contains Python codes for VBayesMM method for PyTorch User.

If you have any problem, please contact me via email: dangthanhtung91@vn-bml.com
