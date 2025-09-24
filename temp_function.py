def check_correctness(prompt: str, completion: str, test: str) -> Tuple[float, str]:
    """
    Evaluates the generated code against the provided test cases with partial credit.
    Returns a decimal score (0.0 to 1.0) based on the percentage of test cases that pass.
    """
    program = f"{prompt}\n{completion}\n{test}"
    try:
        exec_globals = {}
        exec(program, exec_globals)
        return 1.0, "All tests passed"
    except AssertionError as e:
        # Try to run individual test cases to see how many pass
        try:
            import ast
            tree = ast.parse(test)
            assert_count = 0
            passed_count = 0
            
            # Count total assertions
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert):
                    assert_count += 1
            
            # Try to run individual assertions
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert):
                    try:
                        # Create a minimal test for this assertion
                        individual_test = f"def check_single():\n    {ast.unparse(node)}\ncheck_single()"
                        individual_program = f"{prompt}\n{completion}\n{individual_test}"
                        exec_globals_individual = {}
                        exec(individual_program, exec_globals_individual)
                        passed_count += 1
                    except:
                        pass
            
            # Return the percentage of tests that passed
            if assert_count > 0:
                score = passed_count / assert_count
                return score, f"Partial credit: {passed_count}/{assert_count} tests passed"
            else:
                return 0.0, "No assertions found"
        except Exception as parse_error:
            return 0.0, f"AssertionError: {e} (parse error: {parse_error})"
    except Exception as e:
        return 0.0, f"Execution failed: {type(e).__name__}: {e}"
