# LILA
A high-resolution pressure-driven Leakage Identification and Localization Algorithm
This repository contains the **identification part** of LILA.

### Underlying work
Daniel et al. (2021) "LILA: a high-resolution pressure-driven algorithm for leakage identification and localization in water distribution networks." *Journal of Water Resources Planning and Management* (submitted)

### Organization of repository
LILA operates in a sequential way, represented in the following flowchart:
![flowchart](LILA_flowchart.png)
*source: underlying work (see section above)*

First, the notebook in the folder 'LI1_LI2' performs linear regression and provides the error trajectories, then the notebook in 'LI3' performs change point detection to provide the starting times of the leaks.

The further work performing localization is available at:


Additionally, '_utils' contains all data as well as functions used for data loading, helper classes, and the definitions of the functions used for change point detection.

### Related work
- https://doi.org/10.5281/zenodo.3924632 (Version submitted to BattLeDIM)

### Dataset / Ressources
This version is applied to the dataset of the BattLeDIM.

Information on the BattLeDIM can be found at:
- https://battledim.ucy.ac.cy/ (Website hosted by comittee)
- https://zenodo.org/record/3902046 (Overview)
- https://zenodo.org/record/4017659#.X4mBaC2w1hE (Dataset)
- https://zenodo.org/record/4139603#.X8lAfbG5p04.mendeley (Competition results)
