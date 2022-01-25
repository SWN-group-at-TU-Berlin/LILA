# LILA - A high-resolution pressure-driven Leakage Identification and Localization Algorithm
This repository contains the Leakage Identification part of **LILA - the sequential pressure-based algorithm for data-driven Leakage Identification 
and model-based Localization in water distribution networks**. LILA identifies potential leakages via semisupervised linear regression of pairwise sensor pressure data and provides the location of its nearest sensor. LILA also locates leaky pipes relying on an initial set of candidate pipes and a simulation-based optimization framework with iterative linear and mixed-integer linear programming.

### Citation
If you use LILA or part of its code, please consider citing our paper that describes it: 
```
Daniel et al. (forthcoming) "A sequential pressure-based algorithm for data-driven Leakage Identification 
and model-based Localization in water distribution networks" Journal of Water Resources Planning and Management.
DOI:10.1061/(ASCE)WR.1943-5452.0001535
```
### Authors
- [Ivo Daniel](https://www.swn.tu-berlin.de/menue/team/msc_ivo_daniel/), [Andrea Cominola](https://www.swn.tu-berlin.de/menue/team/prof_dr_andrea_cominola/) - [Chair of Smart Water Networks](https://swn.tu-berlin.de) | [Technische Universität Berlin](https://tu.berlin) and [Einstein Center Digital Future, Berlin](https://digital-future.berlin) (Germany)
- Simon Letzgus - [Machine Learning group](https://www.ml.tu-berlin.de/menue/machine_learning/) | [Technische Universität Berlin](https://tu.berlin) (Germany)

### Organization of repository
LILA operates in a sequential way, represented in the following flowchart:
![flowchart](LILA_flowchart.png)
*source: underlying work (Daniel et al., 2022)*

For leakage identification, the notebook in the folder 'LI' performs linear regression analysis and change point detection to provide the starting times of the leaks, and also provides the error trajectories,.

The further work performing localization is available at:
https://github.com/jorgeps86/LeakLocalization

Additionally, [_utils](_utils/) contains all data as well as functions used for data loading, helper classes, and the definitions of the functions used for change point detection.

### Dataset
The work in this repository is applied to the dataset of the [BattLeDIM 2020](https://battledim.ucy.ac.cy/), the Battle of the Leakage Detection and Isolation Methods. The BattLeDIM dataset is open and available at the links below.
Information on the BattLeDIM can be found at the following links:
- Official website of the BattLeDIM, hosted by the BattLeDIM committee:  https://battledim.ucy.ac.cy/
- BattLeDIM problem description and rules: https://zenodo.org/record/3902046
- BattLeDIM dataset: https://zenodo.org/record/4017659#.X4mBaC2w1hE
- BattLeDIM results: https://zenodo.org/record/4139603#.X8lAfbG5p04.mendeley

### References
LILA is fully presented and tested in [Daniel et al. (2022)](DOI:10.1061/(ASCE)WR.1943-5452.0001535): 
```
Daniel et al. (forthcoming) "A sequential pressure-based algorithm for data-driven Leakage Identification 
and model-based Localization in water distribution networks" Journal of Water Resources Planning and Management.
DOI:10.1061/(ASCE)WR.1943-5452.0001535
```
LILA derives from an initial version of the algorithm presented in [Daniel et al. (2020)](https://doi.org/10.5281/zenodo.3924632) and used during the BattLeDIM competition:
```
Danieil, Ivo, Pesantez, Jorge, Letzgus, Simon, Khaksar Fasaee, Mohammad Ali, Alghamdi, Faisal, Mahinthajkumar, Kumar, Berglund, Emily, & Cominola, Andrea. (2020). A high-resolution pressure-driven method for leakage identification and localization in water distribution networks. Zenodo. https://doi.org/10.5281/zenodo.3924632 
```

### LICENSE
Copyright (C) 2021 Ivo Daniel, Simon Letzgus, Andrea Cominola. Released under the [GNU General Public License v3.0](LICENSE). The code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with STREaM. If not, see http://www.gnu.org/licenses/licenses.en.html.
