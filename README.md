# SuperMask — UC3M Final Degree Project

> **Author:** Marcos Caballero Cortés

> **Degree:** BSc in Computer Science  

> **University:** Universidad Carlos III de Madrid (UC3M)

---

## Overview

**SuperMask** is a research project exploring evolutionary algorithms in deep neural networks to search for Supermasks.

This repository includes experimental code and results developed as part of a **Final Degree Project** (TFG) at UC3M.

---

## Repository Structure

```
SuperMask/
├───1. Initial_resolutions/ 
|
├───2. Proving_LTH/         # Scripts for proving LTH
│
├───3. First_AE/            # First evolutionary algorithm
│
├───4. Second_AE/           # Second evolutionary algorithm
│
├───data/                   # Datasets
│
├───Scripts/                # Configuration files and first experiments
|
├── requirements.txt        # Python dependencies
|
├── LICENSE                 # MIT license        
|
└── README.md               # This file
```

---

## Installation

We recommend using a virtual environment.

```bash
git clone https://github.com/cabamarcos/SuperMask.git
cd SuperMask

python3 -m venv venv
source venv/bin/activate    # or .\venv\Scripts\activate on Windows

pip install --upgrade pip
pip install -r requirements.txt
```
---

## How to Run Experiments

All training and evaluation scripts are inside the folder of each experimental phase. 

You can run the experiments  via Jupyter notebooks:

```bash
jupyter notebook path/to/notebook.ipynb
```
The notebooks are self-contained and can be executed step-by-step from within the Jupyter interface.

---

## License

This project is licensed under the [MIT License](./LICENSE).

---

## Contact

For questions, suggestions, or collaborations:

* ✉️ \[[marcoscaballero.contacto@gmail.com](mailto:marcoscaballero.contacto@gmail.com)]
* 🐛 [Open an Issue](https://github.com/cabamarcos/SuperMask/issues)