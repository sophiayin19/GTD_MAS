import re

def gsm_data_process(dataset):
    # This function remains unchanged.
    list_data_dict = []
    for data in dataset:
        item = {"task":data["question"]}
        raw_answer = data["answer"]
        raw_answer_list = raw_answer.split("\n####")
        item["step"] = raw_answer_list[0].strip()
        item["answer"] = raw_answer_list[-1].replace(",", "").strip()
        list_data_dict.append(item)
    return list_data_dict

def gsm_get_predict(pred_str: str) -> str:
    """
    A robust function to extract the final numerical answer from a model's response.
    It handles integers, floats, and various common formatting issues.
    """
    # Priority 1: Look for the <final_answer> tag
    match = re.search(r'<final_answer>(.*?)</final_answer>', pred_str, re.IGNORECASE)
    if match:
        pred_str = match.group(1)

    # Remove commas, dollar signs, and percentage signs for cleaner parsing
    pred_str = pred_str.replace(",", "").replace("$", "").replace("%", "").strip()

    # Regex to find the last number (integer or float) in the string
    # This pattern handles numbers like 123, 123.45, .5, -.5
    pattern = r'[-+]?\d*\.\d+|[-+]?\d+'
    matches = re.findall(pattern, pred_str)
    
    if not matches:
        return "0" # Return '0' if no number is found

    # The final number is usually the last one mentioned
    last_number_str = matches[-1]

    # Convert to float for standardization, then to int if it's a whole number
    try:
        num = float(last_number_str)
        if num.is_integer():
            return str(int(num))
        else:
            return str(num)
    except ValueError:
        return "0" # Return '0' if conversion fails

# The helper functions below are no longer directly used by the new gsm_get_predict,
# but are kept in case they are needed for other legacy parts of the codebase.

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)
        n=str(n)
        return n

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string
