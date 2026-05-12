#!/bin/bash

# Clear previous log
> log.md

echo -e "Running base experiment..."
echo -e "# Base experiment with various fault types and modes\n" >> log.md
python main.py --fault none >> log.md
echo -e "\n---\n" >> log.md
echo "done ✅"

echo -e "Running Label Noise, random..."
echo -e "# 1.1 Label Noise Fault, random\n" >> log.md
python main.py --fault label_noise --mode 0 >> log.md
echo -e "\n---\n" >> log.md
echo "done ✅"

echo -e "Running Label Noise, hard..."
echo -e "# 1.2 Label Noise Fault, hard\n" >> log.md
python main.py --fault label_noise --mode 1 >> log.md
echo -e "\n---\n" >> log.md
echo "done ✅"

echo -e "Running Data Leakage, direct..."
echo -e "# 2.1 Data Leakage Fault, direct\n" >> log.md
python main.py --fault data_leakage --mode 0 >> log.md
echo -e "\n---\n" >> log.md
echo "done ✅"

echo -e "Running Data Leakage, indirect..."
echo -e "# 2.2 Data Leakage Fault, indirect\n" >> log.md
python main.py --fault data_leakage --mode 1 >> log.md
echo -e "\n---\n" >> log.md
echo "done ✅"

echo -e "Running Spurious Correlation, broken..."
echo -e "# 3.1 Spurious Correlation Fault, broken\n" >> log.md
python main.py --fault spurious_correlation --mode 0 >> log.md
echo -e "\n---\n" >> log.md
echo "done ✅"

echo -e "Running Spurious Correlation, inverted..."
echo -e "# 3.2 Spurious Correlation Fault, inverted\n" >> log.md
python main.py --fault spurious_correlation --mode 1 >> log.md
echo -e "\n---\n" >> log.md
echo "done ✅"