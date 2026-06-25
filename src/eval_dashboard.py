from typing import List, Dict, Any, Optional
import numpy as np

"""
Aggregate metrics dashboard utility for evaluation phase.
- Computes summary statistics (mean, std, min, max, count) for numeric metrics across runs/checkpoints
- Filters non-numeric values, skips missing metrics, robust to mixed result dicts
- Outputs summary report as dict or printable table
"""
def aggregate_metrics(results: List[Dict[str, Any]], metric_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Aggregates numeric metrics from a list of result dicts.
    Args:
        results: list of dicts, each with metrics (e.g. from eval_generation_quality, eval_loss)
        metric_keys: which metrics to aggregate (default: all keys except string fields and outputs/prompts)
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
    # Determine which keys to aggregate
    keys = metric_keys if metric_keys is not None else [k for k in all_keys if k not in exclude_keys]
    dashboard = {}
    for key in keys:
        values = []
        for res in results:
            v = res.get(key, None)
            if v is None:
                continue
            # Accept numeric scalars (int/float, not bool) or numeric lists
            if isinstance(v, list):
                values.extend([x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)])
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                values.append(v)
            # skip non-numeric types
        if values:
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
        print(f"- {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}, count={stats['count']}")
