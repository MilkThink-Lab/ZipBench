import re
import math
from math import sqrt, sin, cos, log, pi, factorial, exp, e
E = 2.718


def floatify(num: str):
    try:
        num = float(num)
        if num.is_integer():
            return round(num)
        else:
            return num
    except Exception:
        return None


def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.04
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False


def clean_units(pred_str: str):
    """Clean the units in the number."""
    def convert_pi_to_number(code_string):
        code_string = code_string.replace('\\pi', 'π')
        # Replace \pi or π not preceded by a digit or } with 3.14
        code_string = re.sub(r'(?<![\d}])\\?π', '3.14', code_string)
        # Replace instances where π is preceded by a digit but without a multiplication symbol, e.g., "3π" -> "3*3.14"
        code_string = re.sub(r'(\d)(\\?π)', r'\1*3.14', code_string)
        # Handle cases where π is within braces or followed by a multiplication symbol
        # This replaces "{π}" with "3.14" directly and "3*π" with "3*3.14"
        code_string = re.sub(r'\{(\\?π)\}', '3.14', code_string)
        code_string = re.sub(r'\*(\\?π)', '*3.14', code_string)
        return code_string

    pred_str = convert_pi_to_number(pred_str)
    pred_str = pred_str.replace('%', '/100')
    pred_str = pred_str.replace('$', '')
    pred_str = pred_str.replace('¥', '')
    pred_str = pred_str.replace('°C', '')
    pred_str = pred_str.replace(' C', '')
    pred_str = pred_str.replace('°', '')
    return pred_str


def number_it(num):
    from latex2sympy2_extended import latex2sympy
    if isinstance(num, (int, float)):
        return num

    if not isinstance(num, str):
        return None

    num = num.strip()
    num = clean_units(num)

    # First try to parse directly as a number
    if floatify(num) is not None:
        return floatify(num)

    # Try to parse the simple fraction form a/b
    frac_match = re.match(r'^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$', num.strip())
    if frac_match:
        try:
            result = float(frac_match.group(1)) / float(frac_match.group(2))
            return result
        except Exception:
            pass

    # Try parsing with latex2sympy
    try:
        parsed = str(latex2sympy(num))
        if floatify(parsed) is not None:
            return floatify(parsed)
    except Exception:
        pass

    # Finally fall back to eval
    try:
        result = eval(num)
        if isinstance(result, (list, tuple)):
            result = result[0]
        if floatify(result) is not None:
            return floatify(result)
        if isinstance(result, (int, float)):
            return result
    except Exception:
        pass

    return None


def compare_two_numbers(p, gt):
    try:
        if math.isnan(p):
            return False
        if isinstance(gt, int):
            return round(p) == gt
        else:
            return within_eps(pred=p, gt=gt)
    except Exception:
        return False


def compare_two_list(pred, gt):
    # If pred is a string, try to parse it as a list
    if isinstance(pred, str):
        pred = pred.strip()
        try:
            parsed = eval(pred)
            if isinstance(parsed, (list, tuple)):
                pred = list(parsed)
            else:
                return False
        except Exception:
            # Try to parse the [a, b, c] form manually
            match = re.match(r'[\[\(](.*?)[\]\)]', pred)
            if match:
                inner = match.group(1)
                elements = [e.strip() for e in inner.split(',') if e.strip()]
                pred = elements
            else:
                return False

    if not isinstance(pred, list):
        return False
    if len(pred) != len(gt):
        return False

    # Convert every element to a number
    try:
        pred_nums = []
        for x in pred:
            if isinstance(x, (int, float)):
                pred_nums.append(x)
            else:
                converted = number_it(x)
                if converted is None:
                    return False
                pred_nums.append(converted)
        pred = pred_nums
    except Exception:
        return False

    # Make sure gt is a numeric list as well
    try:
        gt_nums = []
        for x in gt:
            if isinstance(x, (int, float)):
                gt_nums.append(x)
            else:
                converted = number_it(x)
                if converted is None:
                    return False
                gt_nums.append(converted)
        gt = gt_nums
    except Exception:
        return False

    # Compare after sorting (order-insensitive)
    pred = sorted(pred)
    gt = sorted(gt)
    return all(compare_two_numbers(p, g) for p, g in zip(pred, gt))
