#!/bin/bash

# Clear previous log
> code_log_itr_.txt

echo -e "##########################" >> code_log_itr_.txt
echo -e "# main.py" >> code_log_itr_.txt
echo -e "##########################" >> code_log_itr_.txt
cat main.py >> code_log_itr_.txt

echo -e "\n\n##########################" >> code_log_itr_.txt
echo -e "# faults.py" >> code_log_itr_.txt
echo -e "##########################" >> code_log_itr_.txt
cat faults.py >> code_log_itr_.txt

echo -e "\n\n##########################" >> code_log_itr_.txt
echo -e "# baseline_debugging.py" >> code_log_itr_.txt
echo -e "##########################" >> code_log_itr_.txt
cat baseline_debugging.py >> code_log_itr_.txt

echo -e "\n\n##########################" >> code_log_itr_.txt
echo -e "# xai_debugging.py" >> code_log_itr_.txt
echo -e "##########################" >> code_log_itr_.txt
cat xai_debugging.py >> code_log_itr_.txt

echo -e "\n\n##########################" >> code_log_itr_.txt
echo -e "# evaluation_metrics.py" >> code_log_itr_.txt
echo -e "##########################" >> code_log_itr_.txt
cat evaluation_metrics.py >> code_log_itr_.txt

echo -e "\n\n##########################" >> code_log_itr_.txt
echo -e "# log.md" >> code_log_itr_.txt
echo -e "##########################" >> code_log_itr_.txt
cat log.md >> code_log_itr_.txt


echo -e "done ✅"

start .