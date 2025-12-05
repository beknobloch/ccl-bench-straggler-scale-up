from ._common import collect_comm_windows


def metric_cal(directory: str) -> float:
    """
    Calculate the straggler slowdown as the ratio of the slowest to fastest communication window durations.

    Args:
        directory (str): Path to the directory containing kineto traces.

    Returns:
        float: Slowdown factor (>= 1 when data is available, 0 when undefined).
    """

    trace_files, device_windows = collect_comm_windows(directory)
    if not trace_files:
        print(f"No kineto_trace_<rank>.json files found under: {directory}")
        return 0.0

    durations = [window[1] - window[0] for window in device_windows.values() if window[1] > window[0]]
    if len(durations) <= 1:
        return 0.0

    print(f"\n{'='*40}")
    print(f"Per-Rank Communication Window Stats (Slowdown):")
    print(f"{'Rank/PID':<40} | {'Duration (us)':<15}")
    print(f"{'-'*40}-|-{'-'*15}")

    for key, (start, end) in device_windows.items():
        if start >= end: continue
        rank_label = key
        if "trace_rank_" in key:
            try:
                rank_part = key.split("trace_rank_")[1].split(".json")[0]
                rank_label = f"Rank {rank_part}"
            except:
                pass
        print(f"{rank_label:<40} | {end-start:<15.2f}")

    min_duration = min(durations)
    max_duration = max(durations)

    print(f"{'='*40}\n")
    print(f"Min Duration: {min_duration:.2f}")
    print(f"Max Duration: {max_duration:.2f}")
    print(f"Slowdown (Max / Min): {max_duration / min_duration:.2f}")

    if min_duration <= 0:
        return 0.0

    return max_duration / min_duration
