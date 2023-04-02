from typing import List, Tuple


def get_bar_locations(sequence : List[int], summary_idx: List[int]) -> List[Tuple]:
    bar_pairs = []
    bar_start = None
    for pos, token in enumerate(sequence[0]):
        if token in summary_idx:
            if bar_start is not None:
                bar_pairs.append((bar_start, pos))
            bar_start = pos + 1
    return bar_pairs
