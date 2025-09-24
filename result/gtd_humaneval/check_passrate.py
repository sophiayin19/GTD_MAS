#!/usr/bin/env python3
import json
import sys

def main(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    total_problems = len(data)
    attemptcodefail = 0
    solved_problems = sum(1 for item in data if item.get("Solved", 0) == 1.0)
    for item in data:
        attempt = item.get("Attempt_Code")
        firstwordattempt = attempt.split()[0]
        if firstwordattempt != "def" and firstwordattempt != "class" and firstwordattempt != "import" and firstwordattempt != "from":
            attemptcodefail +=1

    if total_problems == 0:
        print("No problems found in file.")
        return

    pass_rate = solved_problems / total_problems
    passratecode = solved_problems / (total_problems - attemptcodefail)
    print(f"Total problems: {total_problems}")
    print(f"Solved problems: {solved_problems}")
    print(f"Pass rate: {pass_rate:.4f}")
    print(f"Attempt code fail: {attemptcodefail}")
    print(f"Pass rate with code: {passratecode:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <json_file>")
        sys.exit(1)
    main(sys.argv[1])