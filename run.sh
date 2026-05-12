#!/bin/bash

# Clear previous log
> log.md

echo -e "Running base experiment..."
echo -e "# Base experiment with various fault types and modes\n" >> log.md
python main.py --fault none >> log.md
echo -e "\n---\n" >> log.md
echo "done ✅"

echo -e "Running Label Noise with Mode 0..."
echo -e "# Label Noise Fault, Mode 0\n" >> log.md
python main.py --fault label_noise --mode 0 >> log.md
echo -e "\n---\n" >> log.md
echo "done ✅"

echo -e "Running Label Noise with Mode 1..."
echo -e "# Label Noise Fault, Mode 1\n" >> log.md
python main.py --fault label_noise --mode 1 >> log.md
echo -e "\n---\n" >> log.md
echo "done ✅"

echo -e "Running Data Leakage with Mode 0..."
echo -e "# Data Leakage Fault, Mode 0\n" >> log.md
python main.py --fault data_leakage --mode 0 >> log.md
echo -e "\n---\n" >> log.md
echo "done ✅"

echo -e "Running Data Leakage with Mode 1..."
echo -e "# Data Leakage Fault, Mode 1\n" >> log.md
python main.py --fault data_leakage --mode 1 >> log.md
echo -e "\n---\n" >> log.md
echo "done ✅"

echo -e "Running Spurious Correlation with Mode 0..."
echo -e "# Spurious Correlation Fault, Mode 0\n" >> log.md
python main.py --fault spurious_correlation --mode 0 >> log.md
echo -e "\n---\n" >> log.md
echo "done ✅"

echo -e "Running Spurious Correlation with Mode 1..."
echo -e "# Spurious Correlation Fault, Mode 1\n" >> log.md
python main.py --fault spurious_correlation --mode 1 >> log.md
echo -e "\n---\n" >> log.md
echo "done ✅"