#!/bin/bash

# Update run_humaneval.sh to use 2 Algorithm Designers and 2 Bug Fixers
echo "Updating run_humaneval.sh to use 2 Programming Experts and 2 Bug Fixers..."

# Update AGENT_NAMES
sed -i 's/AGENT_NAMES="Programming_Expert Test_Analyst"/AGENT_NAMES="Programming_Expert Programming_Expert Bug_Fixer Bug_Fixer"/' run_humaneval.sh

# Update AGENT_NUMS
sed -i 's/AGENT_NUMS="1 1"/AGENT_NUMS="2 2"/' run_humaneval.sh

echo "✅ Updated run_humaneval.sh:"
echo "   AGENT_NAMES: Algorithm_Designer Bug_Fixer"
echo "   AGENT_NUMS: 2 2"

# Show the updated lines
echo ""
echo "Updated configuration:"
grep -E "AGENT_NAMES=|AGENT_NUMS=" run_humaneval.sh

echo ""
echo "Script is ready to run with 2 Algorithm Designers and 2 Bug Fixers!"
