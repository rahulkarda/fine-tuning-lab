from typing import List, Dict, Any, Optional
import numpy as np

"""
Aggregate metrics dashboard utility for evaluation phase.
- Computes summary statistics (mean, std, min, max, count) for numeric metrics across runs/checkpoints
- Filters non-numeric values, skips missing metrics, robust to mixed result dicts
- Outputs summary report as dict or printable table

Usage Example:
    # After running evals (loss, generation quality, etc.)
    results = [
        {'loss': 1.23, 'rouge': 0.3},
        {'loss': 1.10, 'rouge': 0.32},
        {'loss': 1.05, 'rouge': 0.35}
    ]
    from src.eval_dashboard import aggregate_metrics, print_dashboard
    dashboard = aggregate_metrics(results)
    print_dashboard(dashboard)

Notes:
- Only numeric values (int, float, not bool) are aggregated.
- Non-metrics keys (outputs, prompts, diffs) are excluded by default.
- Robust to missing keys: skipped if not present.
- Designed for quick experiment reporting, not full-featured logging.
"""

def _is_numeric(val: Any) -> bool:
    """
    Returns True if val is int or float, but not bool.
    """
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def aggregate_metrics(
    results: List[Dict[str, Any]],
    metric_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Aggregates numeric metrics from a list of result dicts.
    Args:
        results: list of dicts, each with metrics (e.g. from eval_generation_quality, eval_loss)
        metric_keys: which metrics to aggregate (default: all numeric keys except outputs/prompts/diff)
    Returns:
        dict with mean, std, min, max, count per metric
    """
    if not results:
        return {}
    # Keys to exclude from aggregation (non-metrics, text fields)
    exclude_keys = {'outputs', 'prompts', 'base_output', 'tuned_output', 'diff'}
    # Collect all keys seen in any result dict
    all_keys = set()
    for res in results:
        all_keys.update(res.keys())
    
    # Determine which keys to aggregate: only those that are numeric in at least one result
    if metric_keys is not None:
        keys_to_aggregate = metric_keys
    else:
        candidate_keys = [k for k in all_keys if k not in exclude_keys]
        keys_to_aggregate = []
        for k in candidate_keys:
            for res in results:
                v = res.get(k, None)
                if v is None:
                    continue
                if isinstance(v, list):
                    if any(_is_numeric(x) for x in v):
                        keys_to_aggregate.append(k)
                        break
                elif _is_numeric(v):
                    keys_to_aggregate.append(k)
                    break
        # Remove duplicates while preserving order
        seen = set()
        keys_to_aggregate = [k for k in keys_to_aggregate if not (k in seen or seen.add(k))]

    dashboard = {}
    for key in keys_to_aggregate:
        values = []
        for res in results:
            v = res.get(key, None)
            if v is None:
                continue
            # Accept numeric scalars or numeric lists, skip bool
            if isinstance(v, list):
                values.extend([x for x in v if _is_numeric(x)])
            elif _is_numeric(v):
                values.append(v)
            # skip non-numeric types
        # Filter out nan values explicitly
        values = [x for x in values if not (isinstance(x, float) and np.isnan(x))]
        if not values:
            continue
        arr = np.array(values, dtype=np.float32)
        dashboard[key] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'count': int(len(arr))
        }
    return dashboard


def print_dashboard(dashboard: Dict[str, Any]) -> None:
    """
    Pretty-prints aggregate dashboard metrics.
    Args:
        dashboard: dict from aggregate_metrics
    """
    if not dashboard:
        print("No metrics to display.")
        return
    print("Aggregate Metrics Dashboard:")
    for key, stats in dashboard.items():
        # Use .get for each stat and handle nan/missing gracefully
        mean = stats.get('mean')
        std = stats.get('std')
        minv = stats.get('min')
        maxv = stats.get('max')
        count = stats.get('count', 0)
        def fmt(val):
            if val is None:
                return "-"
            try:
                return f"{val:.4f}"
            except Exception:
                return str(val)
        print(f"- {key}: mean={fmt(mean)}, std={fmt(std)}, min={fmt(minv)}, max={fmt(maxv)}, count={count}")
