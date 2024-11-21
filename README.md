# VBayesMM: Variational Bayesian microbiome multiomics

## Publication
Tung Dang, Artem Lysenko, Keith A. Boroevich and Tatsuhiko Tsunoda. "VBayesMM: Variational Bayesian neural network to prioritize important relationships of high-dimensional microbiome multiomics data" 


## The workflow of VBayesMM

<img src="VBayesMM_method.png" width="1000" height="500">

## Quick start
### TensorFlow
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
from VBayesMM import VBayesMM
```

- Loading and preparing data in  ```examples/ ```
  
  Let us first load a subsampled version of the obstructive sleep apnea (OSA) in mice dataset described in Tripathi et al. (2018). VBayesMM supports for loading arbitrary ```biom```, ```tsv```, and ```csv```
  
```
microbes = load_table("microbes.biom")
metabolites = load_table("metabolites.biom")
microbes_df = microbes.to_dataframe()
metabolites_df = metabolites.to_dataframe()

microbes_df = microbes_df.astype(pd.SparseDtype("float64",fill_value=0))
metabolites_df = metabolites_df.astype(pd.SparseDtype("float64",fill_value=0))

microbes_df, metabolites_df = microbes_df.align(metabolites_df, axis=0, join='inner')

num_test = 20

sample_ids = set(np.random.choice(microbes_df.index, size=num_test))
sample_ids = np.array([(x in sample_ids) for x in microbes_df.index])

train_microbes_df = microbes_df.loc[~sample_ids]
test_microbes_df = microbes_df.loc[sample_ids]
train_metabolites_df = metabolites_df.loc[~sample_ids]
test_metabolites_df = metabolites_df.loc[sample_ids]

train_microbes_coo = coo_matrix(train_microbes_df.values)
test_microbes_coo = coo_matrix(test_microbes_df.values)
```

- Creating, training, and testing a model

```
model = VBayesMM()

config = tf.compat.v1.ConfigProto()

with tf.Graph().as_default(), tf.compat.v1.Session(config=config) as session:
    model(session, train_microbes_coo, train_metabolites_df.values,
          test_microbes_coo, test_metabolites_df.values)
    ELBO, _, SMAPE = model.fit(epoch=5000) 
```
| Train data | Test data | 
| ----------------------------------- |:---------------------------------------------:|
| <img src="examples/ELBO.png" width="500" height="300">|<img src="examples/SMAPE.png" width="500" height="300">| 

- Visualizing the posterior distributions of model outputs

```
latent_microbiome_matrix = model.U

microbial_species_selection = model.U_mean_gamma
microbial_species_selection_mean = np.sort(np.mean(microbial_species_selection, axis=1))[::-1]
```
| Latent microbiome matrix | Microbial species selection | 
| ----------------------------------- |:---------------------------------------------:|
| <img src="examples/Posterior_distribution_of_latent_microbiome_matrix.png" width="400" height="400">|<img src="examples/Posterior_distribution_of_microbial_species_selection.png" width="400" height="400">| 


### PyTorch
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

import torch
from VBayesMM import VBayesMM
```

- Loading and preparing data

```
microbes = load_table("microbes.biom")
metabolites = load_table("metabolites.biom")
microbes_df = microbes.to_dataframe()
metabolites_df = metabolites.to_dataframe()

microbes_df = microbes_df.astype(pd.SparseDtype("float64",fill_value=0))
metabolites_df = metabolites_df.astype(pd.SparseDtype("float64",fill_value=0))
microbes_df, metabolites_df = microbes_df.align(metabolites_df, axis=0, join='inner')

num_test = 20

sample_ids = set(np.random.choice(microbes_df.index, size=num_test))
sample_ids = np.array([(x in sample_ids) for x in microbes_df.index])

train_microbes_df = microbes_df.loc[~sample_ids]
test_microbes_df = microbes_df.loc[sample_ids]
train_metabolites_df = metabolites_df.loc[~sample_ids]
test_metabolites_df = metabolites_df.loc[sample_ids]

n, d1 = train_microbes_df.shape
n, d2 = train_metabolites_df.shape

train_microbes_coo = coo_matrix(train_microbes_df.values)
test_microbes_coo = coo_matrix(test_microbes_df.values)
trainY_torch = torch.tensor(train_metabolites_df.to_numpy(), dtype=torch.float32)
testY_torch = torch.tensor(test_metabolites_df.to_numpy(), dtype=torch.float32)
```

- Creating, training, and testing a model

```
model = VBayesMM(d1=d1, d2=d2, num_samples=n, batch_size=n)
ELBO, _, SMAPE = model.fit(train_microbes_coo, trainY_torch, test_microbes_coo, testY_torch, epochs=5000)
config = tf.compat.v1.ConfigProto()
```
| Train data | Test data | 
| ----------------------------------- |:---------------------------------------------:|
| <img src="examples/ELBO_torch.png" width="500" height="300">|<img src="examples/SMAPE_torch.png" width="500" height="300">| 

- Visualizing the posterior distributions of model outputs

```
latent_microbiome_matrix = np.array(model.qUmain_mean.weight.data.detach())

microbial_species_selection = np.array(model.qUmain_mean_gamma.detach())
microbial_species_selection_mean = np.sort(np.mean(microbial_species_selection, axis=1))[::-1]
```
| Latent microbiome matrix | Microbial species selection | 
| ----------------------------------- |:---------------------------------------------:|
| <img src="examples/Posterior_distribution_of_latent_microbiome_matrix_pt.png" width="400" height="400">|<img src="examples/Posterior_distribution_of_microbial_species_selection_pt.png" width="400" height="400">| 


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
