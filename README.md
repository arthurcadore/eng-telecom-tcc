# Telecom Engineering ARGOS-3 Final Project

This repository contains the materials for the Final Project of the Bachelor's Degree in Telecom Engineering at IFSC (Instituto Federal de Santa Catarina). It includes the monograph, extended abstract, and simulation code.

## Repository Structure

- `submodules/monografia`: Contains the LaTeX source files for the monograph.   
- `submodules/resumo-expandido`: Contains the LaTeX source files for the extended abstract.  
- `submodules/simulador`: Contains the source code for the simulation.
- `notebooks/`: Jupyter notebooks used for concepts and data analysis.
- `references/`: Reference materials and bibliographies used on ARGOS3 simulator and LaTeX documents. 

## Dependencies: 

- texlive-xetex (XeTeX 3.141592653-2.6-0.999995 (TeX Live 2023/Debian))
- python3 3.12.3
- python3-pip
- python3-venv

## Getting Started

1. Clone the repository with submodules:
   ```bash
   git clone --recurse-submodules https://github.com/arthurcadore/eng-telecom-tcc.git
    ```
2. Navigate to the project directory:
    ```bash
    cd eng-telecom-tcc
    ```
3. Set up a Python virtual environment and install dependencies:
    ```bash
    make
    ```

## Author: Arthur Cadore M. Barcella, IFSC, São José