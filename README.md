# Data for 47-node Transportation System and 69-node Distribution System

This repository provides the open-source data used in the paper:

> K. Zhang, Y. Xu, Y. Zheng, D. Yang, W. Wu, and H. Sun, “Robust Renewable Charging Station Planning with Spatio-temporal Flexibility and Continuous Endogenous Uncertainty,” *IEEE Transactions on Transportation Electrification*, doi: 10.1109/TTE.2026.3699325.

The dataset is released to support academic research on electric vehicle charging station planning, coupled power-transportation systems, renewable charging stations, and decision-dependent uncertainty modeling.

## Overview

The dataset contains the transportation-network data, distribution-network data, charging-demand profiles, clustering results, electricity-market data, and detailed numerical results used for the case studies in the above paper. The test system consists of a 47-node transportation network and a modified 69-node distribution network.

These data can be used for research topics including, but not limited to:

* Electric vehicle charging station planning;
* Renewable charging station planning with PV and energy storage;
* Coupled power-transportation network modeling;
* Spatio-temporal charging demand flexibility;
* Robust optimization and decision-dependent uncertainty;
* Charging demand transfer and demand response analysis.

## Repository Structure

```text
.
├── Clustering_Partition/
│   └── Clustering and partition results of the transportation network
├── Cap_pv_cap_ess_detailed.xlsx
│   └── Detailed planning results of PV and ESS capacities
├── EVCS_scalability_test_detailed.xlsx
│   └── Detailed scalability test results for EV charging station planning
├── EVCS_ycs_cost_detailed.xlsx
│   └── Detailed cost results of EV charging station planning
├── Electricity market data of China Southern Power Grid.xlsx
│   └── Electricity price and related market data
├── Expected_charging_demand.xlsx
│   └── Expected EV charging demand profiles
├── Luohu_47bus_distance_matrix.xlsx
│   └── Distance matrix of the 47-node transportation network
├── Sensitivity_analysis_DDCDT_detailed.xlsx
│   └── Detailed sensitivity analysis results of the DDCDT model
├── Typical_load.xlsx
│   └── Typical load profiles for the distribution network
└── ieee69bus.xlsx
    └── Data of the modified IEEE 69-bus distribution system
```

## Citation

If you use this dataset in your research, please cite the following paper:

```bibtex
@article{zhang2026robust,
  author  = {Zhang, Kaizhe and Xu, Yinliang and Zheng, Yunhan and Yang, Dingtong and Wu, Wenchuan and Sun, Hongbin},
  title   = {Robust Renewable Charging Station Planning with Spatio-temporal Flexibility and Continuous Endogenous Uncertainty},
  journal = {IEEE Transactions on Transportation Electrification},
  year    = {2026},
  doi     = {10.1109/TTE.2026.3699325}
}
```

## Notes

The data are provided for academic and research purposes. Users are encouraged to properly acknowledge the original paper when using the dataset, reproducing the case studies, or developing extended models based on this test system.

Although we have carefully checked the dataset, the files are provided “as is” without any warranty. Users are responsible for verifying the data before applying them to their own research or engineering studies.

## Contact

For questions, suggestions, or potential research collaboration, please contact the repository owner or open an issue in this repository.
