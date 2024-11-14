# 🎓 Algorithms for Multiple Multidimensional Knapsack Problem with Family-Split Penalties
This repository contains the code for my Master Thesis: *Modelli e Algoritmi di Ottimizzazione per Problemi Multi Risorsa con Penalità Family-Split*. A copy of the document is available in the `docs` folder.
The thesis treats two variants of the Multiple Multidimensional Knapsack Problem with Family-Split Penalties: the first was already known in the literature while the second is proposed as a new.

## Directories and files
- `docs`: contains the documents related to the project. 📃
    - `relaxation_analysis`: contains files used to analyze the continuous relaxation of the problem.
    - `utility_scripts`: contains scripts used to analyze the instances and the continuous relaxation of the problem.
    - `thesis.pdf`: the final version of the thesis.
- `final_results`: contains relevant executions'output, which are discussed in the report. 🔬
- `instances`: contains the instances created for the second variant. 💾
- `mkfsp`: contains the codebase provided by the assignment. 🧱
- `source`: contains the source code of the algorithms used for the second variant. 👨‍💻
- `first_variant_source_and_instances`: contains instances and source code for the first variant. ⏱


## Requirements
This project needs Python 3.10+ (available on the [Python website](https://www.python.org/)).
Two external libraries are used:
- [pandas](https://pandas.pydata.org/) 🐼
- [gurobipy](https://pypi.org/project/gurobipy/)

Note that you will also need a Gurobi license to effectively use the external library.
Follow the instructions on [this page](https://www.gurobi.com/academia/academic-program-and-licenses/)
to get a Gurobi free academic licence.

## First execution
### Install requirements
Install required packages with the following command:
```bash
pip install -r requirements.txt
```

### Configure constants 🔧
There are some constants that can be changed in in the `source/solve_instances.py` file. They're explained in code's comments and also in documentation.

### Run the code 🚀
To solve all the instances in the `instances` folder, execute the following command:
```bash
python -m source.main
```
If you want to set one or more parameters, you can use the following command:
```bash
python -m source.main -t 60 -i 0 -ni 0 -k 5 -fa 5 -ks 5 -is 5 -mi 5 -r 3 -wr -wa -wi -wk -rwa -rwr -d
```

## Results
To read the results, open the `.csv` files with a spreadsheet program (e.g. Microsoft Excel, LibreOffice Calc, etc.).

It's encoded in UTF-8 and separated by semicolons.

## Logging
All the logs generated during the execution will be saved in the corresponding `/logs` folder.

## FAQ
If you don't understand something in the code, you may find the explanation in the thesis, in `docs` folder.