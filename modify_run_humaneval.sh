#!/bin/bash

echo "Modifying run_humaneval.sh to use Programming Expert + Test Analyst..."

# Backup the original file
cp run_humaneval.sh run_humaneval.sh.backup

# Modify the agent configuration
sed -i 's/AGENT_NAMES="Project_Manager Algorithm_Designer Programming_Expert"/AGENT_NAMES="Programming_Expert Test_Analyst"/' run_humaneval.sh

sed -i 's/AGENT_NUMS="1 1 1" # Corresponds to the number of agents for each name/AGENT_NUMS="1 1" # Corresponds to the number of agents for each name/' run_humaneval.sh

echo "✅ Successfully modified run_humaneval.sh"
echo "Changes made:"
echo "- AGENT_NAMES: Programming_Expert Test_Analyst"
echo "- AGENT_NUMS: 1 1"
echo ""
echo "Original file backed up as: run_humaneval.sh.backup"
echo ""
echo "You can now run: ./run_humaneval.sh"
