...
def train_val_split(
    data: List[Any],
    val_ratio: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[List[Any], List[Any]]:
    """
    Randomly split dataset into train and validation sets with seed control.
    Args:
        data: list of items
        val_ratio: fraction of items to assign to val set (0 < val_ratio < 1)
        seed: random seed for reproducibility
    Returns:
        train, val: (list, list)
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1)")
    num_items = len(data)
    val_size = max(1, int(num_items * val_ratio)) if num_items > 0 and val_ratio > 0 else 0
    indices = list(range(num_items))
    rand = random.Random(seed) if seed is not None else random
    rand.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    val = [data[i] for i in val_indices]
    train = [data[i] for i in train_indices]
    return train, val
...
