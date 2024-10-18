# FedVul

## Dataset
Two vulnerability datasets link: 
* BigVul: <https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing>
* DiverseVul: <https://drive.google.com/file/d/12IWKhmLhq7qn5B_iXgn5YerOQtkH-6RG/view?usp=sharing>


## Requirement
Our code is based on Python3. There are a few dependencies to run the code. The major libraries are listed as follows:
* torch
* torch_geometric
* dgl
* transformers

## 📥 Guide

#### Data Preprocessing


- (1) We download Joern [here](https://github.com/joernio/joern). 

- (2) Follow the Joern documentation to generate a code property graph.
  
- (3) Put the processed files into the data folder.

#### Training

```bash
python Vul_Classification.py --a fedvul_vc --b diverse_graph_vc --m Reveal
```

```bash
python Vul_Detection.py --a fedvul_vd --b diverse_graph_vd --m Reveal
```

