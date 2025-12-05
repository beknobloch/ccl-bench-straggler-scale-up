from ._common import collect_comm_windows


def metric_cal(directory: str) -> float:
    """
    Calculate the normalized straggler delay (relative lag) for a communication group.

    Args:
        directory (str): Path to the directory containing kineto traces.

    Returns:
        float: Delay metric in [0, 1].
    """

    trace_files, device_windows = collect_comm_windows(directory)
    if not trace_files:
        print(f"No kineto_trace_<rank>.json files found under: {directory}")
        return 0.0

    valid_windows = [window for window in device_windows.values() if window[0] < window[1]]
    if len(valid_windows) <= 1:
        return 0.0

    global_start = min(window[0] for window in valid_windows)
    slowest_end = max(window[1] for window in valid_windows)
    fastest_end = min(window[1] for window in valid_windows)
    total_window = slowest_end - global_start

    print(f"\n{'='*40}")
    print(f"Per-Rank Communication Window Stats (Delay):")
    print(f"{'Rank/PID':<40} | {'Start (us)':<15} | {'End (us)':<15} | {'Duration (us)':<15}")
    print(f"{'-'*40}-|-{'-'*15}-|-{'-'*15}-|-{'-'*15}")
    
    for key, (start, end) in device_windows.items():
        if start >= end: continue
        rank_label = key
        if "trace_rank_" in key:
            try:
                rank_part = key.split("trace_rank_")[1].split(".json")[0]
                rank_label = f"Rank {rank_part}"
            except:
                pass
        print(f"{rank_label:<40} | {start:<15.2f} | {end:<15.2f} | {end-start:<15.2f}")
    
    print(f"{'='*40}\n")
    print(f"Global Start: {global_start:.2f}")
    print(f"Slowest End:  {slowest_end:.2f}")
    print(f"Fastest End:  {fastest_end:.2f}")
    print(f"Total Window: {total_window:.2f}")
    print(f"Delay (Slowest - Fastest): {slowest_end - fastest_end:.2f}")

    if total_window <= 0:
        return 0.0

    return (slowest_end - fastest_end) / total_window
