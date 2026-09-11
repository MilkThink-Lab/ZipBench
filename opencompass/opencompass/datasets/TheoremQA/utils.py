import re
from .number_utils import clean_units, compare_two_numbers, compare_two_list, number_it
import contextlib
import signal

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise ValueError

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _has_option(text: str) -> bool:
    return re.search(r'\([a-f]\)', text, flags=re.IGNORECASE) is not None


def _has_bool(text: str) -> bool:
    return re.search(r'\b(?:true|false|yes|no)\b', text, flags=re.IGNORECASE) is not None


def _has_digit(text: str) -> bool:
    return re.search(r'\d', text) is not None


def _is_formula_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if re.search(r'\\[a-zA-Z]+', stripped):
        return True
    if re.search(r'[\^_]', stripped):
        return True
    if re.search(r'[=<>≤≥≈]', stripped) and _has_digit(stripped):
        return True
    if re.search(r'[+\-*/]', stripped) and _has_digit(stripped) and re.search(r'[A-Za-z]', stripped):
        return True
    if re.search(r'[\(\)\[\]]', stripped) and _has_digit(stripped) and re.search(r'[=+\-*/^_]', stripped):
        return True
    if re.search(r'\b(?:det|sin|cos|tan|log|ln|exp)\b', stripped, flags=re.IGNORECASE):
        return True
    return False


def _is_simple_equation_answer(text: str) -> bool:
    cleaned = _strip_latex(text)
    if not cleaned:
        return False
    if len(cleaned) > 60:
        return False
    if re.search(r'[<>≤≥]', cleaned):
        return False
    numbers = re.findall(r'-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?', cleaned)
    if not numbers:
        return False
    if len(numbers) > 1:
        return False
    return True


def _is_valid_candidate(text: str) -> bool:
    return _has_option(text) or _has_bool(text) or _has_digit(text)


def _is_numeric_like(text: str) -> bool:
    return re.fullmatch(r'-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?', text) is not None


def _is_list_like(text: str) -> bool:
    # Stricter list detection: only match lists of plain numbers
    # e.g. [1, 2, 3], [0.5, -0.3] or (1, 2)
    # Do not match content containing letters or LaTeX commands
    text = text.strip()
    if not text:
        return False
    # Must start with [ or ( and end with ] or )
    if not ((text.startswith('[') and text.endswith(']')) or
            (text.startswith('(') and text.endswith(')'))):
        return False
    inner = text[1:-1].strip()
    # Must not contain letters (except 'e' for scientific notation) or backslashes (LaTeX commands)
    if re.search(r'[a-df-zA-DF-Z]|\\', inner):
        return False
    # Must contain several comma-separated numbers
    if ',' not in inner:
        return False
    # Verify that every element is a number
    elements = [e.strip() for e in inner.split(',')]
    if len(elements) < 2:
        return False
    for elem in elements:
        if not elem:
            continue
        # Every element should be numeric (negatives, decimals and scientific notation allowed)
        if not re.fullmatch(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', elem):
            return False
    return True


def _is_valid_final(text: str, allow_list: bool, allow_option: bool) -> bool:
    if text in ('True', 'False'):
        return True
    if allow_option and _has_option(text):
        return True
    if _is_numeric_like(text):
        return True
    if allow_list and _is_list_like(text):
        return True
    return False


def _strip_latex(text: str) -> str:
    text = text.replace('\\(', '').replace('\\)', '')
    text = text.replace('\\[', '').replace('\\]', '')
    text = text.replace('$', '')
    text = re.sub(r'\\text\{[^}]*\}', '', text)
    text = re.sub(r'\\mathrm\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = text.replace('{', '').replace('}', '')
    text = re.sub(r'(?<=\d),(?=\d{3}(?:,|$))', '', text)
    return text


def _extract_option(text: str) -> str:
    options = re.findall(r'\([a-f]\)', text, flags=re.IGNORECASE)
    if options:
        return options[-1].lower()
    return ''


def _extract_option_from_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        has_context = re.search(r'\b(option|answer|choice)\b', line, flags=re.IGNORECASE)
        standalone = re.fullmatch(r'[\(\[]?[a-f][\)\]]\.?', line.strip(), flags=re.IGNORECASE)
        if not (has_context or standalone):
            continue
        option = _extract_option(line)
        if option:
            return option
    return ''


def _extract_bool(text: str, include_yes_no: bool = True) -> str:
    tokens = r'true|false|yes|no' if include_yes_no else r'true|false'
    matches = re.findall(rf'\b(?:{tokens})\b', text, flags=re.IGNORECASE)
    if not matches:
        return ''
    token = matches[-1].lower()
    if token in ('yes', 'true'):
        return 'True'
    return 'False'


def _extract_tail_bool(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        match = _extract_bool(line, include_yes_no=False)
        if match:
            return match
        if re.fullmatch(r'[^a-zA-Z]*(yes|no)[^a-zA-Z]*', line, flags=re.IGNORECASE):
            return _extract_bool(line, include_yes_no=True)
        if re.search(r'\b(yes|no)\b[.!?]*\s*$', line, flags=re.IGNORECASE):
            return _extract_bool(line, include_yes_no=True)
    return ''


def _infer_bool_from_tail(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if _is_formula_like(line):
            continue
        cleaned = _strip_latex(line).lower()
        if not cleaned:
            continue
        if re.search(
            r'\bfalse\b'
            r'|\bno\b(?!\.)'
            r'|\bimpossible\b'
            r'|\bcannot\b'
            r"|\bcan't\b"
            r'|\bcontradiction\b'
            r'|\bcounterexample\b'
            r'|\bfails?\b'
            r'|\bviolates\b'
            r'|\bdoes\s+not\s+(?:hold|exist|apply)\b'
            r'|\bnot\s+(?:true|valid|correct|possible|feasible|linear|convex|concave|continuous|monotone|'
            r'increasing|decreasing|injective|surjective|bijective|unique)\b',
            cleaned,
        ):
            return 'False'
        if re.search(
            r'\btrue\b',
            cleaned,
        ):
            return 'True'
        if re.search(r'\b(?:correct|valid|does\s+hold|holds|satisfies)\b', cleaned) and not _has_digit(cleaned):
            return 'True'
        if not _has_digit(cleaned) and re.search(r'^(conclusion|judgment|judgement)\b', cleaned):
            return 'True'
        if not _has_digit(cleaned) and re.search(r'\bdefinition\b', cleaned):
            return 'True'
    return ''


def _has_bool_cue(text: str) -> bool:
    cleaned = _strip_latex(text).lower()
    if re.search(r'\b(?:true|false|yes|no)\b', cleaned):
        return True
    if re.search(
        r'\bimpossible\b'
        r'|\bcannot\b'
        r"|\bcan't\b"
        r'|\bcontradiction\b'
        r'|\bcounterexample\b'
        r'|\bfails?\b'
        r'|\bviolates\b'
        r'|\bdoes\s+not\s+(?:hold|exist|apply)\b'
        r'|\bdoes\s+hold\b'
        r'|\bholds\b'
        r'|\bsatisfies\b'
        r'|\bvalid\b'
        r'|\bcorrect\b'
        r'|\bconclusion\b'
        r'|\bjudgment\b'
        r'|\bjudgement\b'
        r'|\bdefinition\b'
        r'|\bnot\s+(?:true|valid|correct|possible|feasible|linear|convex|concave|continuous|monotone|'
        r'increasing|decreasing|injective|surjective|bijective|unique)\b',
        cleaned,
    ):
        return True
    return False


def _find_matching_brace(s: str, start: int) -> int:
    """Find the '}' matching the '{' at position start."""
    count = 1
    i = start + 1
    while i < len(s) and count > 0:
        if s[i] == '{':
            count += 1
        elif s[i] == '}':
            count -= 1
        i += 1
    return i - 1 if count == 0 else -1


def _extract_boxed_or_math(text: str) -> str:
    # Use recursive matching to handle nested braces
    results = []
    pattern = r'\\boxed\{'
    for m in re.finditer(pattern, text):
        start = m.end() - 1  # position of '{'
        end = _find_matching_brace(text, start)
        if end > start:
            content = text[start+1:end]
            if _is_valid_candidate(content):
                results.append(content.strip())

    if results:
        return results[-1]
    return ''


def _parse_latex_fractions(text: str) -> str:
    """Convert a LaTeX fraction to a decimal number."""
    import math

    def replace_frac(match):
        numer = match.group(1)
        denom = match.group(2)
        try:
            # Handle pi
            numer = numer.replace('\\pi', str(math.pi)).replace('pi', str(math.pi))
            denom = denom.replace('\\pi', str(math.pi)).replace('pi', str(math.pi))
            result = float(eval(numer)) / float(eval(denom))
            return f'{result:.4f}'
        except Exception:
            return match.group(0)

    # Match the \frac{a}{b} form
    text = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', replace_frac, text)
    return text


def _extract_list_answer(text: str) -> str:
    """Extract answers written in list form."""
    # Match several list forms, ordered by priority
    patterns = [
        # LaTeX vector/matrix form
        (r'\\begin\{[bp]?matrix\}([\s\S]*?)\\end\{[bp]?matrix\}', 'matrix'),
        # **[1, 2, 3]** markdown bold form
        (r'\*\*\[([^\]]+)\]\*\*', 'bold_bracket'),
        # List of LaTeX fractions \left[\frac{1}{3}, \frac{1}{4}\right]
        (r'\\left\s*\[([\s\S]*?)\]\s*\\right', 'latex_bracket'),
        # $[...]$ form
        (r'\$\s*\[([\s\S]*?)\]\s*\$', 'dollar_bracket'),
        # Plain [a, b, c] form
        (r'\[([^\[\]]*?\d[^\[\]]*?)\]', 'plain_bracket'),
    ]

    for pattern, ptype in patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            content = matches[-1].strip()
            if ptype == 'matrix':
                # Handle matrix form: 1 \\ 2 \\ 3 -> [1, 2, 3]
                elements = re.split(r'\\\\|&', content)
                elements = [e.strip() for e in elements if e.strip()]
                if elements:
                    # Try to parse fractions
                    parsed = [_parse_latex_fractions(e) for e in elements]
                    return '[' + ', '.join(parsed) + ']'
            else:
                # Handle other forms
                content = _parse_latex_fractions(content)
                # Strip extra whitespace and LaTeX leftovers
                content = re.sub(r'\\[a-zA-Z]+', '', content)  # remove \left, \right, etc.
                content = re.sub(r'\s+', ' ', content).strip()
                return f'[{content}]'
    return ''


def _extract_last_line_or_sentence(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        candidate = lines[-1]
    else:
        candidate = text.strip()
    if not candidate:
        return ''
    sentences = re.split(r'[.!?]\s+', candidate)
    for item in reversed(sentences):
        if item.strip():
            return item.strip()
    return candidate


def _extract_last_textual_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text.strip()
    for line in reversed(lines):
        if not _is_formula_like(line):
            return line
    return lines[-1]


def _extract_last_number_expr(text: str) -> str:
    matches = re.findall(r'-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?', text)
    if matches:
        return matches[-1].strip()
    return ''


def _extract_first_number_expr(text: str) -> str:
    matches = re.findall(r'-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?', text)
    if matches:
        return matches[0].strip()
    return ''


def _tail_text(text: str, max_lines: int = 10, max_chars: int = 800) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tail_lines = lines[-max_lines:] if lines else [text.strip()]
    tail = '\n'.join(tail_lines)
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def _trim_to_first_candidate_sentence(text: str) -> str:
    if not text:
        return text
    parts = re.split(r'(?:\n+|(?<!\d)[.!?](?!\d))', text)
    for part in parts:
        part = part.strip()
        if part and _is_valid_candidate(part):
            return part
    return text.strip()


def _extract_explicit_answer(text: str) -> str:
    patterns = [
        r'\bfinal answer\b\s*[:：]?\s*([^\n]+)',
        r'\banswer\b\s*(?:is)?\s*[:：]\s*([^\n]+)',
        r'\banswer\b\s*(?:is)\s*([^\n]+)',
        r'\bcorrect option\b\s*[:：]?\s*([^\n]+)',
    ]
    lines = [line for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                candidate = candidate.strip('*').strip()
                candidate = re.sub(r'<think>.*?</think>', '', candidate, flags=re.DOTALL)
                candidate = candidate.replace('</think>', '').replace('<think>', '').strip()
                candidate = _trim_to_first_candidate_sentence(candidate)
                return candidate
    return ''


def _has_answer_cue(text: str) -> bool:
    return re.search(
        r'\b(answer|final|result|conclusion|therefore|thus|hence)\b',
        text,
        flags=re.IGNORECASE,
    ) is not None


def _has_strong_answer_cue(text: str) -> bool:
    return re.search(
        r'\b(answer|final answer|result|correct option)\b',
        text,
        flags=re.IGNORECASE,
    ) is not None


def _has_numeric_cue(text: str) -> bool:
    return re.search(
        r'\b(answer|result|value|equals?)\b'
        r'|[=≈]'
        r'|\bapprox(?:imately)?\b',
        text,
        flags=re.IGNORECASE,
    ) is not None


def _extract_number_from_answer_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if not _has_numeric_cue(line):
            continue
        line = re.sub(r'^\s*\(?\d+[.)]\s*', '', line)
        candidate = _normalize_candidate(line, allow_list=False, allow_option=False)
        if _is_numeric_like(candidate):
            return candidate
    return ''


def _extract_leading_bool(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:3]:
        match = _extract_bool(line, include_yes_no=True)
        if not match:
            continue
        if re.fullmatch(r'[^a-zA-Z]*(true|false|yes|no)[^a-zA-Z]*', line, flags=re.IGNORECASE):
            return match
        if line.lower().startswith(('true', 'false', 'yes', 'no')):
            return match
        if re.search(r'\b(?:true|false|yes|no)\b[.!?]*\s*$', line, flags=re.IGNORECASE):
            return match
    return ''


def _normalize_candidate(
    candidate: str, allow_list: bool, allow_option: bool, prefer_first_number: bool = False
) -> str:
    candidate = candidate.strip()

    # Handle list form first (before options and booleans)
    if allow_list:
        # Check whether it contains a list/vector form
        list_match = re.search(r'[\[\(](.*?)[\]\)]', candidate)
        if list_match:
            inner = list_match.group(1)
            # Check whether there are several comma-separated elements
            if ',' in inner or '\\\\' in inner:
                # Try to parse LaTeX fractions
                parsed = _parse_latex_fractions(inner)
                # Strip whitespace
                parsed = re.sub(r'\s+', ' ', parsed).strip()
                return f'[{parsed}]'

    option = _extract_option(candidate)
    if option and (allow_option or len(candidate) <= 8 or re.search(r'\b(option|answer|choice)\b', candidate, flags=re.IGNORECASE)):
        return option
    bool_value = _extract_bool(candidate, include_yes_no=True)
    if bool_value:
        return bool_value
    if allow_list and _is_list_like(candidate):
        return candidate
    cleaned = _strip_latex(candidate)
    cleaned = cleaned.replace('≈', ' ')
    cleaned = cleaned.replace('~', ' ')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if '=' in cleaned:
        cleaned = cleaned.split('=')[-1].strip()
    if prefer_first_number:
        num = _extract_first_number_expr(cleaned)
    else:
        num = _extract_last_number_expr(cleaned)
    if num:
        return num
    return cleaned


def extract_theoremqa_answer(pred: str, answer_flag: bool = True):
    from latex2sympy2_extended import latex2sympy

    # If it is already in list form, return as-is (do not break the list)
    if _is_list_like(pred):
        return pred

    pred_lower = pred.lower()
    if re.search(r'\b(?:yes|true)\b', pred_lower):
        pred = 'True'
    elif re.search(r'\b(?:no|false)\b', pred_lower):
        pred = 'False'
    elif any([option in pred_lower for option in ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']]):
        pass
    else:
        if answer_flag:
            # Extract the numbers out of the string
            pred = pred.split('=')[-1].strip()
            pred = clean_units(pred)
            try:
                with time_limit(1):
                    tmp = str(latex2sympy(pred))
                    pred = eval(tmp)
                    if isinstance(pred, tuple):
                        pred = str(list(pred))
                    else:
                        pred = str(pred)

            except Exception:
                if re.match(r'-?[\d\.]+\s\D+$', pred):
                    pred = pred.split(' ')[0]
                elif re.match(r'-?[\d\.]+\s[^\s]+$', pred):
                    pred = pred.split(' ')[0]
        else:
            # desparate search over the last number
            preds = re.findall(r'-?\d*\.?\d+', pred)
            if(len(preds) >= 1):
                pred = preds[-1]
            else:
                pred = ''
    return pred

def answer_clean(direct_answer_trigger_for_fewshot: tuple, pred: str):
    raw_pred = pred
    pred = pred.strip('\n')

    # Determine if this is ICL, if so, use \n\n to split the first chunk.
    ICL = False
    for trigger in direct_answer_trigger_for_fewshot:
        if pred.count(trigger) > 1:
            ICL = True
    if ICL:
        pred = pred.split('\n\n')[0]

    # Split the trigger to find the answer.
    preds = re.split('|'.join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip('\n').rstrip('.').rstrip('/').strip(' ')
    if not answer_flag:
        fallback = _extract_explicit_answer(raw_pred)
        if fallback:
            normalized = _normalize_candidate(fallback, allow_list=True, allow_option=True)
            if _is_valid_final(normalized, allow_list=True, allow_option=True):
                pred = normalized
                answer_flag = True
        if not answer_flag:
            leading_bool = _extract_leading_bool(raw_pred)
            if leading_bool:
                pred = leading_bool
                answer_flag = True
        if not answer_flag:
            tail = _tail_text(raw_pred)
            bool_cue = _has_bool_cue(tail)
            bool_candidate = _extract_tail_bool(tail)
            if bool_candidate:
                pred = bool_candidate
                answer_flag = True
            if not answer_flag:
                option_candidate = _extract_option_from_lines(tail)
                if option_candidate:
                    pred = option_candidate
                    answer_flag = True
            if not answer_flag and bool_cue:
                inferred_bool = _infer_bool_from_tail(tail)
                if inferred_bool:
                    pred = inferred_bool
                    answer_flag = True
            if not answer_flag:
                fallback = _extract_boxed_or_math(tail)
                if fallback:
                    normalized = _normalize_candidate(fallback, allow_list=True, allow_option=True)
                    if _is_valid_final(normalized, allow_list=True, allow_option=True):
                        pred = normalized
                        answer_flag = True
            # Try to extract a list-form answer
            if not answer_flag:
                list_answer = _extract_list_answer(tail)
                if list_answer:
                    pred = list_answer
                    answer_flag = True
            if not answer_flag:
                last_line = _extract_last_textual_line(tail)
                allow_last = True
                if _is_formula_like(last_line) and not (
                    _has_answer_cue(last_line) or _is_simple_equation_answer(last_line)
                ):
                    allow_last = False
                if allow_last:
                    normalized = _normalize_candidate(last_line, allow_list=False, allow_option=False)
                    if _is_valid_final(normalized, allow_list=False, allow_option=False):
                        pred = normalized
                        answer_flag = True
            if not answer_flag:
                number_from_answer = _extract_number_from_answer_lines(tail)
                if number_from_answer:
                    pred = number_from_answer
                    answer_flag = True
            if not answer_flag and _has_strong_answer_cue(tail):
                fallback = _extract_last_number_expr(tail)
                if fallback:
                    pred = fallback
                    answer_flag = True

    pred = [extract_theoremqa_answer(pred, answer_flag)]

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last e
            pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip('.').rstrip('/')
    return pred



def compare_answer_with_groundtruth(answer: str, groundtruth_str: str, groundtruth_num = None):
    if groundtruth_str.lower() in ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']:
        return groundtruth_str.lower() in answer.lower()
    elif answer.lower() == groundtruth_str.lower():
        return True
    elif groundtruth_num is not None:
        if isinstance(groundtruth_num, (int, float)):
            return compare_two_numbers(number_it(answer), groundtruth_num)
        else:
            # groundtruth_num is a list; try list comparison
            # Supports several list forms: (a, b), [a, b]
            if (answer.startswith('(') and answer.endswith(')')) or \
               (answer.startswith('[') and answer.endswith(']')):
                try:
                    answer_list = list(eval(answer))
                    answer_list = [number_it(a) for a in answer_list]
                except Exception:
                    return False
                return compare_two_list(answer_list, groundtruth_num)
            else:
                # Try compare_two_list directly (it now supports parsing strings)
                return compare_two_list(answer, groundtruth_num)
    else:
        return False
