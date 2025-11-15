# Image Super Resolution using Diffusion models

In this repository I will document my progress on the super-resolution project.

## Deliverables (in the repo)
1. A working model
2. Model report
3. [Optional] Project presentation

## How to run
I highly recommend using a virtual environment in order to avoid conflicts with pre-existing libraries of different versions.
```
python3 -m venv myenv
pip install -r requirements.txt
cd src
```
Download datasets with
```
python3 prepare_data.py
```

## Current project structure
``` bash
.
├── README.md
├── requirements.txt
└── src
    ├── model.py
    └── prepare_data.py
```