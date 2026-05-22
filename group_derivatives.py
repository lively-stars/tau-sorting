import matplotlib.pyplot as plt
import numpy as np


def read_grouped_column_file(filename):
    """
    Reads a one-column text file that contains group labels (strings)
    and numeric values. Every label starts a new group; all following
    numeric lines belong to that group until the next label.

    Returns
    -------
    groups : dict
        {label: np.ndarray of values}
    """
    groups = {}
    current_label = None
    current_vals = []

    with open(filename) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue  # skip empty lines

            try:
                val = float(s)
                # It's a numeric value
                if current_label is None:
                    # in case file starts with numbers and no header
                    current_label = "group_0"
                current_vals.append(val)
            except ValueError:
                # It's a label / header line
                # store previous group (if any)
                if current_label is not None and len(current_vals) > 0:
                    groups[current_label] = np.array(current_vals, dtype=float)
                current_label = s
                current_vals = []

    # store last group if present
    if current_label is not None and len(current_vals) > 0:
        groups[current_label] = np.array(current_vals, dtype=float)

    return groups


def smooth_1d(y, window=5):
    """
    Moving-average smoothing with edge padding.
    Guarantees that output has same length as input
    for both odd and even window sizes.
    """
    y = np.asarray(y, dtype=float)
    if window < 3 or len(y) < 3:
        return y

    window = int(window)
    kernel = np.ones(window, dtype=float) / window

    # asymmetric padding to keep length == len(y)
    pad_left = window // 2
    pad_right = window - 1 - pad_left

    y_pad = np.pad(y, pad_width=(pad_left, pad_right), mode="edge")
    y_smooth = np.convolve(y_pad, kernel, mode="valid")
    return y_smooth


def piecewise_linear_breaks_tails(
    x,
    y,
    min_tail_frac_low=0.01,
    max_tail_frac_low=0.20,  # e.g. up to 20% for low tail
    min_tail_frac_high=0.01,
    max_tail_frac_high=0.05,  # e.g. only up to 5% for high tail
    step_tail_frac=0.01,
    min_mid_frac=0.20,
):
    """
    Find two breakpoints that split (x, y) into three segments:
    low tail, middle, high tail.

    Low and high tails have *separate* min/max allowed fractions.

    Returns
    -------
    result : dict with keys:
        'b1', 'b2'          : break indices
        'low_len', 'mid_len', 'high_len'
        'low_frac', 'mid_frac', 'high_frac'
        'hit_max_tail_low', 'hit_max_tail_high', 'hit_max_tail_any'
        'sse_3seg'          : SSE of best 3-segment fit
        'params'            : dict of search parameters
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 10:
        raise ValueError("Too few points for 3 segments.")

    # convert fractions to absolute lengths
    min_low_tail = max(3, int(round(min_tail_frac_low * n)))
    max_low_tail = max(min_low_tail + 1, int(round(max_tail_frac_low * n)))
    min_high_tail = max(3, int(round(min_tail_frac_high * n)))
    max_high_tail = max(min_high_tail + 1, int(round(max_tail_frac_high * n)))

    step_tail = max(1, int(round(step_tail_frac * n)))
    min_mid = max(1, int(round(min_mid_frac * n)))

    best_sse = np.inf
    best = None  # will hold (b1, b2, low_len, high_len)

    for low_len in range(min_low_tail, max_low_tail + 1, step_tail):
        for high_len in range(min_high_tail, max_high_tail + 1, step_tail):
            mid_len = n - low_len - high_len
            if mid_len < min_mid:
                continue  # middle too small, skip

            b1 = low_len - 1
            b2 = n - high_len

            # segment 1: [0 .. b1]
            x1, y1 = x[: b1 + 1], y[: b1 + 1]
            c1 = np.polyfit(x1, y1, 1)
            y1_fit = np.polyval(c1, x1)
            sse = np.sum((y1 - y1_fit) ** 2)

            # segment 2: [b1 .. b2]
            x2, y2 = x[b1 : b2 + 1], y[b1 : b2 + 1]
            c2 = np.polyfit(x2, y2, 1)
            y2_fit = np.polyval(c2, x2)
            sse += np.sum((y2 - y2_fit) ** 2)

            # segment 3: [b2 .. n-1]
            x3, y3 = x[b2:], y[b2:]
            c3 = np.polyfit(x3, y3, 1)
            y3_fit = np.polyval(c3, x3)
            sse += np.sum((y3 - y3_fit) ** 2)

            if sse < best_sse:
                best_sse = sse
                best = (b1, b2, low_len, high_len)

    if best is None:
        raise RuntimeError("No valid segmentation found. Check parameters.")

    b1, b2, low_len, high_len = best
    mid_len = n - low_len - high_len

    low_frac = low_len / n
    mid_frac = mid_len / n
    high_frac = high_len / n

    hit_max_tail_low = low_len >= max_low_tail
    hit_max_tail_high = high_len >= max_high_tail
    hit_max_tail_any = hit_max_tail_low or hit_max_tail_high

    result = {
        "b1": b1,
        "b2": b2,
        "low_len": low_len,
        "mid_len": mid_len,
        "high_len": high_len,
        "low_frac": low_frac,
        "mid_frac": mid_frac,
        "high_frac": high_frac,
        "hit_max_tail_low": hit_max_tail_low,
        "hit_max_tail_high": hit_max_tail_high,
        "hit_max_tail_any": hit_max_tail_any,
        "sse_3seg": best_sse,
        "params": {
            "n": n,
            "min_tail_frac_low": min_tail_frac_low,
            "max_tail_frac_low": max_tail_frac_low,
            "min_tail_frac_high": min_tail_frac_high,
            "max_tail_frac_high": max_tail_frac_high,
            "step_tail_frac": step_tail_frac,
            "min_mid_frac": min_mid_frac,
            "min_low_tail_points": min_low_tail,
            "max_low_tail_points": max_low_tail,
            "min_high_tail_points": min_high_tail,
            "max_high_tail_points": max_high_tail,
            "min_mid_points": min_mid,
        },
    }
    return result


def refine_middle_segment(x, y, seg_info, mid_step_frac=0.02, min_mid_sub_frac=0.10):
    """
    Optional refinement: split the middle segment into two by adding
    one extra breakpoint, if it improves the SSE.

    Parameters
    ----------
    x, y : 1D arrays
        Original data (same as for the tail finder).
    seg_info : dict
        Output of piecewise_linear_breaks_tails.
    mid_step_frac : float
        Step size for candidate splits as fraction of mid length.
    min_mid_sub_frac : float
        Minimum fraction of mid points required in each subsegment.

    Returns
    -------
    seg_info_refined : dict
        Like seg_info but possibly with extra keys:
        'split_mid' : bool
        'b_mid'     : breakpoint inside middle (if split_mid == True)
        'sse_4seg'  : SSE of 4-segment fit (if used)
        and updated mid1_len, mid2_len, etc.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    b1 = seg_info["b1"]
    b2 = seg_info["b2"]
    low_len = seg_info["low_len"]
    high_len = seg_info["high_len"]
    mid_len = seg_info["mid_len"]
    sse_3seg = seg_info["sse_3seg"]

    # if middle is too small, nothing to do
    if mid_len < 3:
        seg_info["split_mid"] = False
        return seg_info

    mid_start = b1
    mid_end = b2
    n_mid = mid_end - mid_start + 1

    min_mid_sub = max(1, int(round(min_mid_sub_frac * n_mid)))
    if 2 * min_mid_sub > n_mid:
        seg_info["split_mid"] = False
        return seg_info

    mid_step = max(1, int(round(mid_step_frac * n_mid)))

    # baseline SSE for 3 segments (recompute once)
    def sse_3():
        sse = 0.0
        # low tail
        x1, y1 = x[: b1 + 1], y[: b1 + 1]
        c1 = np.polyfit(x1, y1, 1)
        y1_fit = np.polyval(c1, x1)
        sse += np.sum((y1 - y1_fit) ** 2)
        # middle
        x2, y2 = x[b1 : b2 + 1], y[b1 : b2 + 1]
        c2 = np.polyfit(x2, y2, 1)
        y2_fit = np.polyval(c2, x2)
        sse += np.sum((y2 - y2_fit) ** 2)
        # high tail
        x3, y3 = x[b2:], y[b2:]
        c3 = np.polyfit(x3, y3, 1)
        y3_fit = np.polyval(c3, x3)
        sse += np.sum((y3 - y3_fit) ** 2)
        return sse

    baseline_sse3 = sse_3()  # we could also use seg_info["sse_3seg"], but this recomputes consistently

    # candidate internal splits in the middle, ordered from centre outward
    j_min = mid_start + min_mid_sub
    j_max = mid_end - min_mid_sub
    candidates = list(range(j_min, j_max + 1, mid_step))

    centre = (mid_start + mid_end) // 2
    candidates.sort(key=lambda j: abs(j - centre))  # start near centre, then go left/right

    best_sse4 = baseline_sse3
    best_j = None

    for j in candidates:
        # four segments: [0..b1], [b1..j], [j..b2], [b2..n-1]
        sse = 0.0

        # low tail
        x1, y1 = x[: b1 + 1], y[: b1 + 1]
        c1 = np.polyfit(x1, y1, 1)
        y1_fit = np.polyval(c1, x1)
        sse += np.sum((y1 - y1_fit) ** 2)

        # mid1
        x2, y2 = x[b1 : j + 1], y[b1 : j + 1]
        c2 = np.polyfit(x2, y2, 1)
        y2_fit = np.polyval(c2, x2)
        sse += np.sum((y2 - y2_fit) ** 2)

        # mid2
        x3, y3 = x[j : b2 + 1], y[j : b2 + 1]
        c3 = np.polyfit(x3, y3, 1)
        y3_fit = np.polyval(c3, x3)
        sse += np.sum((y3 - y3_fit) ** 2)

        # high tail
        x4, y4 = x[b2:], y[b2:]
        c4 = np.polyfit(x4, y4, 1)
        y4_fit = np.polyval(c4, x4)
        sse += np.sum((y4 - y4_fit) ** 2)

        if sse < best_sse4:
            best_sse4 = sse
            best_j = j

    if (best_j is not None) and (best_sse4 < baseline_sse3):
        # accept split
        mid1_len = best_j - b1
        mid2_len = b2 - best_j
        seg_info["split_mid"] = True
        seg_info["b_mid"] = best_j
        seg_info["sse_4seg"] = best_sse4
        seg_info["mid1_len"] = mid1_len
        seg_info["mid2_len"] = mid2_len
        seg_info["mid1_frac"] = mid1_len / n
        seg_info["mid2_frac"] = mid2_len / n
    else:
        seg_info["split_mid"] = False

    return seg_info


def sse_4seg(x, y, b1, b_mid, b2):
    """
    Compute total SSE for four linear segments:
    [0..b1], [b1..b_mid], [b_mid..b2], [b2..n-1]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    sse = 0.0

    # segment 1: low tail [0..b1]
    x1, y1 = x[: b1 + 1], y[: b1 + 1]
    c1 = np.polyfit(x1, y1, 1)
    y1_fit = np.polyval(c1, x1)
    sse += np.sum((y1 - y1_fit) ** 2)

    # segment 2: mid1 [b1..b_mid]
    x2, y2 = x[b1 : b_mid + 1], y[b1 : b_mid + 1]
    c2 = np.polyfit(x2, y2, 1)
    y2_fit = np.polyval(c2, x2)
    sse += np.sum((y2 - y2_fit) ** 2)

    # segment 3: mid2 [b_mid..b2]
    x3, y3 = x[b_mid : b2 + 1], y[b_mid : b2 + 1]
    c3 = np.polyfit(x3, y3, 1)
    y3_fit = np.polyval(c3, x3)
    sse += np.sum((y3 - y3_fit) ** 2)

    # segment 4: high tail [b2..n-1]
    x4, y4 = x[b2:], y[b2:]
    c4 = np.polyfit(x4, y4, 1)
    y4_fit = np.polyval(c4, x4)
    sse += np.sum((y4 - y4_fit) ** 2)

    return sse


def refine_b1_local(x, y, seg_info, window_frac=0.05, min_mid_sub_frac=0.10):
    """
    Locally refine the first breakpoint b1 by sliding it within a window.

    Uses 4-segment SSE (low, mid1, mid2, high) with fixed b_mid and b2.
    Respects min lengths for low tail, mid1, mid2, high tail.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    params = seg_info["params"]
    b1 = seg_info["b1"]
    b2 = seg_info["b2"]
    if not seg_info.get("split_mid", False):
        # nothing to refine if we don't have a middle split
        return seg_info

    b_mid = seg_info["b_mid"]

    # minimal lengths from original tail search
    min_low_tail_points = params["min_low_tail_points"]
    max_low_tail_points = params["max_low_tail_points"]
    min_high_tail_points = params["min_high_tail_points"]
    max_high_tail_points = params["max_high_tail_points"]

    # approximate total middle length between b1 and b2
    mid_total = b2 - b1
    min_mid_sub = max(1, int(round(min_mid_sub_frac * mid_total)))

    # local search window around current b1
    win = max(1, int(round(window_frac * n)))
    cand_min = max(min_low_tail_points - 1, b1 - win)
    # must leave enough room for mid1 and mid2 before b2,
    # and not exceed the max low-tail length
    cand_max = min(b1 + win, b_mid - min_mid_sub, max_low_tail_points - 1)

    if cand_min >= cand_max:
        return seg_info  # nothing to do

    best_b1 = b1
    best_sse = np.inf

    for cand_b1 in range(cand_min, cand_max + 1):
        low_len = cand_b1 + 1
        mid1_len = b_mid - cand_b1
        mid2_len = b2 - b_mid
        high_len = n - b2

        if (
            low_len < min_low_tail_points
            or low_len > max_low_tail_points
            or mid1_len < min_mid_sub
            or mid2_len < min_mid_sub
            or high_len < min_high_tail_points
            or high_len > max_high_tail_points
        ):
            continue

        sse = sse_4seg(x, y, cand_b1, b_mid, b2)
        if sse < best_sse:
            best_sse = sse
            best_b1 = cand_b1

    # update seg_info if we found an improvement
    if best_b1 != b1 and best_sse < np.inf:
        seg_info["b1_old"] = b1
        seg_info["b1"] = best_b1
        # recompute lengths/fractions
        low_len = best_b1 + 1
        high_len = n - b2
        mid_len = n - low_len - high_len
        seg_info["low_len"] = low_len
        seg_info["high_len"] = high_len
        seg_info["mid_len"] = mid_len
        seg_info["low_frac"] = low_len / n
        seg_info["mid_frac"] = mid_len / n
        seg_info["high_frac"] = high_len / n

    return seg_info


def refine_b2_local(x, y, seg_info, window_frac=0.05, min_mid_sub_frac=0.10):
    """
    Locally refine the second breakpoint b2 by sliding it within a window.

    Uses 4-segment SSE (low, mid1, mid2, high) with fixed b1 and b_mid.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    params = seg_info["params"]
    b1 = seg_info["b1"]
    b2 = seg_info["b2"]
    if not seg_info.get("split_mid", False):
        return seg_info

    b_mid = seg_info["b_mid"]

    min_low_tail_points = params["min_low_tail_points"]
    max_low_tail_points = params["max_low_tail_points"]
    min_high_tail_points = params["min_high_tail_points"]
    max_high_tail_points = params["max_high_tail_points"]

    mid_total = b2 - b1
    min_mid_sub = max(1, int(round(min_mid_sub_frac * mid_total)))

    win = max(1, int(round(window_frac * n)))
    # must leave room for mid2 and high tail, and not exceed max high-tail length
    cand_min = max(b2 - win, b_mid + min_mid_sub, n - max_high_tail_points)
    cand_max = min(b2 + win, n - min_high_tail_points)

    if cand_min >= cand_max:
        return seg_info

    best_b2 = b2
    best_sse = np.inf

    for cand_b2 in range(cand_min, cand_max + 1):
        low_len = b1 + 1
        mid1_len = b_mid - b1
        mid2_len = cand_b2 - b_mid
        high_len = n - cand_b2

        if (
            low_len < min_low_tail_points
            or low_len > max_low_tail_points
            or mid1_len < min_mid_sub
            or mid2_len < min_mid_sub
            or high_len < min_high_tail_points
            or high_len > max_high_tail_points
        ):
            continue

        sse = sse_4seg(x, y, b1, b_mid, cand_b2)
        if sse < best_sse:
            best_sse = sse
            best_b2 = cand_b2

    if best_b2 != b2 and best_sse < np.inf:
        seg_info["b2_old"] = b2
        seg_info["b2"] = best_b2
        low_len = b1 + 1
        high_len = n - best_b2
        mid_len = n - low_len - high_len
        seg_info["low_len"] = low_len
        seg_info["high_len"] = high_len
        seg_info["mid_len"] = mid_len
        seg_info["low_frac"] = low_len / n
        seg_info["mid_frac"] = mid_len / n
        seg_info["high_frac"] = high_len / n

    return seg_info


def iterative_refine_breaks(
    x,
    y,
    seg_initial,
    n_iter=2,
    mid_step_frac=0.02,
    min_mid_sub_frac=0.10,
    window_frac=0.05,
    warn_mid_shift_frac=0.25,
):
    """
    Iteratively refine b1, b2 and b_mid.

    Steps per iteration:
    1) refine middle segment (get b_mid),
    2) locally refine b1,
    3) locally refine b2,
    4) refine middle again and check if b_mid moves a lot;
       if so, set a warning flag.

    Parameters
    ----------
    x, y : arrays
        Data in the space you are segmenting (e.g. smoothed log-values).
    seg_initial : dict
        Output of piecewise_linear_breaks_tails.
    n_iter : int
        How many refinement cycles to perform.
    mid_step_frac : float
        Step size for candidate mid splits.
    min_mid_sub_frac : float
        Minimum fraction of mid length required for each mid subsegment.
    window_frac : float
        Local search window for b1, b2 as fraction of n.
    warn_mid_shift_frac : float
        If |b_mid_new - b_mid_old| > warn_mid_shift_frac * mid_len,
        raise a warning flag in the returned dict.

    Returns
    -------
    seg : dict
        Updated segmentation info (with possible b1, b2, b_mid changes
        and 'mid_shift_warning' flag).
    """
    seg = seg_initial.copy()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    for it in range(n_iter):
        # 1) (re)compute middle split
        seg = refine_middle_segment(x, y, seg, mid_step_frac=mid_step_frac, min_mid_sub_frac=min_mid_sub_frac)

        if not seg.get("split_mid", False):
            break  # nothing more to refine

        old_b_mid = seg["b_mid"]

        # 2) refine b1 locally
        seg = refine_b1_local(x, y, seg, window_frac=window_frac, min_mid_sub_frac=min_mid_sub_frac)

        # 3) refine b2 locally
        seg = refine_b2_local(x, y, seg, window_frac=window_frac, min_mid_sub_frac=min_mid_sub_frac)

        # 4) recompute middle split after updated b1, b2
        seg = refine_middle_segment(x, y, seg, mid_step_frac=mid_step_frac, min_mid_sub_frac=min_mid_sub_frac)

        if seg.get("split_mid", False):
            new_b_mid = seg["b_mid"]
            mid_len = seg["mid_len"]
            shift = abs(new_b_mid - old_b_mid)
            seg["b_mid_old"] = old_b_mid
            seg["b_mid_shift"] = shift
            seg["mid_shift_warning"] = shift > warn_mid_shift_frac * mid_len
        else:
            seg["mid_shift_warning"] = False

    return seg


def best_weighted_polyfit(x, y, degrees=(3, 4, 5), high_frac=0.2, high_weight=10.0):
    """
    Fit polynomials of given degrees to (x, y) with weights that heavily
    emphasise the top `high_frac` fraction of y-values.

    Parameters
    ----------
    x, y : array-like
        Data points.
    degrees : iterable of int
        Polynomial degrees to try.
    high_frac : float
        Fraction (0-1) of points with the highest y-values that get a
        boosted weight.
    high_weight : float
        Multiplicative factor for the weights of the top `high_frac`.

    Returns
    -------
    best_poly : np.poly1d
        Polynomial with best weighted RMS fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(y)
    if n == 0:
        raise ValueError("Empty data for polyfit.")

    # base weights = 1
    w = np.ones_like(y, dtype=float)

    # determine threshold for top `high_frac` largest y-values
    if 0.0 < high_frac < 1.0:
        k = max(1, int(np.floor(high_frac * n)))
        # index of k-th largest value
        # sort y descending
        sort_idx = np.argsort(y)[::-1]
        top_idx = sort_idx[:k]
        w[top_idx] *= high_weight
    else:
        # if high_frac is weird, just keep uniform weights
        pass

    best_poly = None
    best_score = np.inf
    best_deg = None

    for deg in degrees:
        # weighted polynomial fit
        coefs = np.polyfit(x, y, deg=deg, w=w)
        p = np.poly1d(coefs)

        # weighted RMS of residuals
        resid = y - p(x)
        # typical weighted mean: sum(w * r^2) / sum(w)
        score = np.sqrt(np.sum(w * resid**2) / np.sum(w))

        if score < best_score:
            best_score = score
            best_poly = p
            best_deg = deg

    print(f"  -> chosen degree {best_deg} (score={best_score:.3e})")
    return best_poly


def analyze_group(
    values,
    smooth_window=7,
    min_tail_frac_low=0.01,
    max_tail_frac_low=0.20,
    min_tail_frac_high=0.01,
    max_tail_frac_high=0.05,
    step_tail_frac=0.01,
    min_mid_frac=0.20,
    refine_mid=True,
):
    """
    Smooth + segment a single 1D group of values.

    Pipeline:
      1) log10 transform (with non-positive guard).
      2) moving-average smoothing.
      3) initial 3-segment break search (piecewise_linear_breaks_tails).
      4) iterative refinement (iterative_refine_breaks) — skipped when
         refine_mid is False; the seg dict will then have split_mid=False
         and no b_mid.

    Parameters
    ----------
    values : array-like
        Raw values for one group (will be log10'd).
    smooth_window : int
        Moving-average window passed to smooth_1d.
    refine_mid : bool
        If False, return the 3-segment result from piecewise_linear_breaks_tails
        without running iterative_refine_breaks (no mid split, no b1/b2 nudging).

    Returns
    -------
    dict with keys:
        seg       : segmentation dict (b1, b2, optional b_mid, *_len, *_frac, ...)
        y_data    : log10 of input
        y_smooth  : smoothed log10
        x         : index array
    """
    values = np.asarray(values, dtype=float)

    if np.any(values <= 0):
        pos = values[values > 0]
        if len(pos) == 0:
            raise ValueError("All values are <= 0, cannot take log10.")
        min_pos = np.min(pos)
        safe = np.where(values <= 0, min_pos * 1e-6, values)
        y_data = np.log10(safe)
    else:
        y_data = np.log10(values)

    x = np.arange(len(y_data))

    y_smooth = smooth_1d(y_data, window=smooth_window)

    seg0 = piecewise_linear_breaks_tails(
        x,
        y_smooth,
        min_tail_frac_low=min_tail_frac_low,
        max_tail_frac_low=max_tail_frac_low,
        min_tail_frac_high=min_tail_frac_high,
        max_tail_frac_high=max_tail_frac_high,
        step_tail_frac=step_tail_frac,
        min_mid_frac=min_mid_frac,
    )

    if refine_mid:
        seg = iterative_refine_breaks(
            x,
            y_smooth,
            seg_initial=seg0,
            n_iter=6,
            mid_step_frac=0.02,
            min_mid_sub_frac=0.10,
            window_frac=0.05,
            warn_mid_shift_frac=0.25,
        )
    else:
        seg = dict(seg0)
        seg["split_mid"] = False

    return {"seg": seg, "y_data": y_data, "y_smooth": y_smooth, "x": x}


def process_file(
    filename=None,
    smooth_window=7,
    normalise_derivatives=True,
    max_vertical_lines=6,
    edge_cut=3,  # ignore maxima too close to boundaries
    groups=None,
):
    """
    Run analyze_group + diagnostic plot for every group.

    Source of groups (exactly one of):
      filename : str — file readable by read_grouped_column_file.
      groups   : dict[label, array-like] — pre-built in-memory groups.
    """
    if (filename is None) == (groups is None):
        raise ValueError("Provide exactly one of `filename` or `groups`.")

    if groups is None:
        groups = read_grouped_column_file(filename)

    for label, values in groups.items():
        print(f"Processing group: {label}")

        result = analyze_group(values, smooth_window=smooth_window)
        seg = result["seg"]
        y_data = result["y_data"]
        y_smooth = result["y_smooth"]
        x = result["x"]

        # 3. dump some diagnostics
        print(f"Group {label}:")
        print(f"  low  tail: {seg['low_len']} pts ({seg['low_frac']:.3f})")
        print(f"  mid  part: {seg['mid_len']} pts ({seg['mid_frac']:.3f})")
        print(f"  high tail: {seg['high_len']} pts ({seg['high_frac']:.3f})")
        print(f"  hit_max_tail_low  = {seg['hit_max_tail_low']}")
        print(f"  hit_max_tail_high = {seg['hit_max_tail_high']}")

        if seg.get("split_mid", False):
            print(f"  mid split at index {seg['b_mid']}")
            print(f"    mid1: {seg['mid1_len']} pts ({seg['mid1_frac']:.3f})")
            print(f"    mid2: {seg['mid2_len']} pts ({seg['mid2_frac']:.3f})")
            if seg.get("mid_shift_warning", False):
                print(f"  WARNING: mid break moved a lot (shift={seg['b_mid_shift']}, mid_len={seg['mid_len']})")
        else:
            print("  middle not split further.")

        # ---- PLOTTING ----
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

        ax1.plot(x, y_data, label="data (log)", linewidth=1.0, alpha=0.4)
        ax1.plot(x, y_smooth, label="smoothed (log)", linewidth=1.5)

        b1 = seg["b1"]
        b2 = seg["b2"]
        ax1.axvline(x=b1, linestyle=":", linewidth=1.0, alpha=0.8, color="red")
        ax1.axvline(x=b2, linestyle=":", linewidth=1.0, alpha=0.8, color="red")
        if seg.get("split_mid", False):
            ax1.axvline(x=seg["b_mid"], linestyle="--", linewidth=1.0, alpha=0.8, color="orange")

        ax1.set_ylabel("log10(value)")
        ax1.set_title(f"Group: {label}")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage:
    # process_file("sorted_buffer.txt", smooth_window=7)
    import sys

    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
    else:
        process_file(sys.argv[1], smooth_window=7)
