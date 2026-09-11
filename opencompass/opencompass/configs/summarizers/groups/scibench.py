# opencompass/opencompass/configs/summarizers/groups/scibench.py
scibench_summary_groups = []

scibench_tasks = ['atkins', 'calculus', 'chemmc', 'class', 'diff', 'fund', 'matter', 'quan', 'stat', 'thermo']

# Sample counts for each subset (for weighted average)
_scibench_weights = {
    'atkins': 107,
    'calculus': 42,
    'chemmc': 39,
    'class': 47,
    'diff': 50,
    'fund': 73,
    'matter': 49,
    'quan': 34,
    'stat': 75,
    'thermo': 67,
}

for suffix in ['', '_zs-cot', '_fs', '_fs-cot']:
    subsets = [f'scibench-{subset}{suffix}' for subset in scibench_tasks]
    
    # Simple average (naive_average)
    scibench_summary_groups.append({'name': f'scibench{suffix}', 'subsets': subsets})
    
    # Weighted average by sample count (true accuracy = correct / total)
    weights = {f'scibench-{k}{suffix}': v for k, v in _scibench_weights.items()}
    scibench_summary_groups.append({'name': f'scibench{suffix}-weighted', 'subsets': subsets, 'weights': weights})
