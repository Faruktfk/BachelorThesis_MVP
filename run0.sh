#!/bin/bash

# Clear previous log
> log.md

for seed in 0 1 2 3; do
    echo -e "Running seed: $seed"

    echo -e "\n\n===========SEED: $seed=============\n\n" >> log.md
    echo -e "## SEED: $seed" >> log.md
        
    echo -e "Running base experiment..."
    echo -e "### Base experiment with various fault types and modes\n" >> log.md
    python main.py --fault none --seed $seed >> log.md
    echo -e "\n---\n" >> log.md
    echo "done ✅"

    echo -e "Running Label Noise, random..."
    echo -e "### 1.1 Label Noise Fault, random\n" >> log.md
    python main.py --fault label_noise --mode 0 --seed $seed >> log.md
    echo -e "\n---\n" >> log.md
    echo "done ✅"

    echo -e "Running Label Noise, hard..."
    echo -e "### 1.2 Label Noise Fault, hard\n" >> log.md
    python main.py --fault label_noise --mode 1 --seed $seed >> log.md
    echo -e "\n---\n" >> log.md
    echo "done ✅"

    echo -e "Running Data Leakage, direct..."
    echo -e "### 2.1 Data Leakage Fault, direct\n" >> log.md
    python main.py --fault data_leakage --mode 0 --seed $seed >> log.md
    echo -e "\n---\n" >> log.md
    echo "done ✅"

    echo -e "Running Data Leakage, indirect..."
    echo -e "### 2.2 Data Leakage Fault, indirect\n" >> log.md
    python main.py --fault data_leakage --mode 1 --seed $seed >> log.md
    echo -e "\n---\n" >> log.md
    echo "done ✅"

    echo -e "Running Spurious Correlation, broken..."
    echo -e "### 3.1 Spurious Correlation Fault, broken\n" >> log.md
    python main.py --fault spurious_correlation --mode 0 --seed $seed >> log.md
    echo -e "\n---\n" >> log.md
    echo "done ✅"

    echo -e "Running Spurious Correlation, inverted..."
    echo -e "### 3.2 Spurious Correlation Fault, inverted\n" >> log.md
    python main.py --fault spurious_correlation --mode 1 --seed $seed >> log.md
    echo -e "\n---\n" >> log.md
    echo "done ✅"
done