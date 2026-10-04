"""Myers' O(ND) diff.

Usage:
    main.py lines A B       Part A: minimal line diff of file A to file B

Design
------
* myers(a, b) works on ANY two indexable sequences (lists of ints for lines,
  plain strings for characters). It returns two flag arrays:
      dele[i] = 1  if a[i] is deleted
      ins[j]  = 1  if b[j] is inserted
  Every element that is not flagged is kept, in order.
* Common prefix and suffix are trimmed first (cheap, and most real diffs
  are mostly identical).
* The main algorithm is the greedy forward Myers algorithm. V[k] holds the
  furthest x reached on diagonal k = x - y. A copy of the live part of V is
  saved for every d (the "trace") so the edit path can be walked backwards.
* The trace needs O(D^2) memory. If it grows past TRACE_BUDGET entries we
  throw it away and redo the diff with the linear-space variant (middle
  snake + divide and conquer), which needs only O(N) memory.
"""

import sys
from array import array

# Maximum number of V entries kept in the trace (int32 each -> ~160 MB).
TRACE_BUDGET = 40_000_000


# --------------------------------------------------------------------------
# Core algorithm (trace version)
# --------------------------------------------------------------------------
def _greedy(a, b, dele, ins, a0, b0, budget):
    """Myers forward search with a saved trace.

    Marks deletions/insertions in dele/ins (shifted by a0/b0).
    Returns False, having changed nothing, if the trace outgrows `budget`.
    """
    n = len(a)
    m = len(b)
    if n == 0:
        for j in range(m):
            ins[b0 + j] = 1
        return True
    if m == 0:
        for i in range(n):
            dele[a0 + i] = 1
        return True

    off = n + m + 1
    v = [0] * (2 * (n + m) + 3)   # v[off + k] = furthest x on diagonal k
    trace = [None]                # trace[d] = v[-d..d] before round d
    used = 0
    final_d = -1

    for d in range(n + m + 1):
        if d:
            used += 2 * d + 1
            if used > budget:
                return False
            trace.append(array('i', v[off - d:off + d + 1]))
        # i = off + k is the index of diagonal k inside v
        lo = off - d
        hi = off + d
        for i in range(lo, hi + 1, 2):
            if i == lo or (i != hi and v[i - 1] < v[i + 1]):
                x = v[i + 1]                # step down: insert from b
            else:
                x = v[i - 1] + 1            # step right: delete from a
            y = x + off - i                 # y = x - k
            while x < n and y < m and a[x] == b[y]:   # follow the snake
                x += 1
                y += 1
            v[i] = x
            if x >= n and y >= m:
                final_d = d
                break
        if final_d >= 0:
            break

    # Walk the path backwards from (n, m) to (0, 0).
    x = n
    y = m
    for d in range(final_d, 0, -1):
        vs = trace[d]
        k = x - y
        if k == -d or (k != d and vs[k - 1 + d] < vs[k + 1 + d]):
            pk = k + 1
        else:
            pk = k - 1
        px = vs[pk + d]
        py = px - pk
        if pk == k + 1:
            ins[b0 + py] = 1         # moved down -> b[py] inserted
        else:
            dele[a0 + px] = 1        # moved right -> a[px] deleted
        x = px
        y = py
    return True


# --------------------------------------------------------------------------
# Linear-space fallback (middle snake, divide and conquer)
# --------------------------------------------------------------------------
def _middle_snake(a, al, ar, b, bl, br):
    """Return (x1, y1, x2, y2) of the middle snake, local to the sub-problem.

    The sub-problem is a[al:ar] against b[bl:br] (both non-empty).
    """
    n = ar - al
    m = br - bl
    delta = n - m
    odd = delta & 1
    half = (n + m + 1) // 2
    off = half + 1
    size = 2 * half + 3
    vf = [0] * size
    vb = [0] * size               # reverse search, in reversed coordinates
    for d in range(half + 1):
        for k in range(-d, d + 1, 2):             # forward
            if k == -d or (k != d and vf[off + k - 1] < vf[off + k + 1]):
                x = vf[off + k + 1]
            else:
                x = vf[off + k - 1] + 1
            y = x - k
            sx = x
            sy = y
            while x < n and y < m and a[al + x] == b[bl + y]:
                x += 1
                y += 1
            vf[off + k] = x
            if odd and delta - (d - 1) <= k <= delta + (d - 1):
                if x + vb[off + delta - k] >= n:
                    return sx, sy, x, y
        for k in range(-d, d + 1, 2):             # reverse
            if k == -d or (k != d and vb[off + k - 1] < vb[off + k + 1]):
                x = vb[off + k + 1]
            else:
                x = vb[off + k - 1] + 1
            y = x - k
            sx = x
            sy = y
            while x < n and y < m and a[ar - 1 - x] == b[br - 1 - y]:
                x += 1
                y += 1
            vb[off + k] = x
            if not odd and -d <= delta - k <= d:
                if x + vf[off + delta - k] >= n:
                    return n - x, m - y, n - sx, m - sy
    raise AssertionError("no middle snake found")


def _linear(a, b, dele, ins):
    """Linear-space Myers; explicit stack instead of recursion."""
    stack = [(0, len(a), 0, len(b))]
    while stack:
        al, ar, bl, br = stack.pop()
        while al < ar and bl < br and a[al] == b[bl]:     # trim prefix
            al += 1
            bl += 1
        while al < ar and bl < br and a[ar - 1] == b[br - 1]:   # trim suffix
            ar -= 1
            br -= 1
        if al == ar:
            for j in range(bl, br):
                ins[j] = 1
            continue
        if bl == br:
            for i in range(al, ar):
                dele[i] = 1
            continue
        x1, y1, x2, y2 = _middle_snake(a, al, ar, b, bl, br)
        stack.append((al, al + x1, bl, bl + y1))
        stack.append((al + x2, ar, bl + y2, br))


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def myers(a, b, budget=TRACE_BUDGET):
    """Minimal diff of two sequences -> (dele, ins) flag bytearrays."""
    n = len(a)
    m = len(b)
    dele = bytearray(n)
    ins = bytearray(m)

    lo = 0                                   # common prefix
    top = min(n, m)
    while lo < top and a[lo] == b[lo]:
        lo += 1
    hi = 0                                   # common suffix
    top -= lo
    while hi < top and a[n - 1 - hi] == b[m - 1 - hi]:
        hi += 1

    sa = a[lo:n - hi]
    sb = b[lo:m - hi]
    if not _greedy(sa, sb, dele, ins, lo, lo, budget):
        sub_d = bytearray(len(sa))
        sub_i = bytearray(len(sb))
        _linear(sa, sb, sub_d, sub_i)
        dele[lo:lo + len(sa)] = sub_d
        ins[lo:lo + len(sb)] = sub_i
    return dele, ins


# --------------------------------------------------------------------------
# File reading
# --------------------------------------------------------------------------
def read_lines(path):
    """Read a file as raw bytes and split it into lines (newline kept out)."""
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return lines


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def build_output(a, b):
    # Map every distinct line to a small int so comparisons are int compares.
    ids = {}
    ia = [ids.setdefault(line, len(ids)) for line in a]
    ib = [ids.setdefault(line, len(ids)) for line in b]
    n = len(a)
    m = len(b)

    dele, ins = myers(ia, ib)
    n = len(a)
    m = len(b)

    out = []
    i = 0
    j = 0
    while i < n or j < m:
        if (i < n and dele[i]) or (j < m and ins[j]):
            # A change block: deletions first, then insertions.
            dels = []
            adds = []
            while i < n and dele[i]:
                dels.append(a[i])
                i += 1
            while j < m and ins[j]:
                adds.append(b[j])
                j += 1
            for line in dels:
                out.append(b"-" + line + b"\n")
            for line in adds:
                out.append(b"+" + line + b"\n")
        else:
            out.append(b" " + a[i] + b"\n")
            i += 1
            j += 1
    return out


def main(argv):
    if len(argv) != 4 or argv[1] != "lines":
        sys.stderr.write("usage: main.py lines FILE_A FILE_B\n")
        return 2
    try:
        a = read_lines(argv[2])
        b = read_lines(argv[3])
    except OSError as e:
        sys.stderr.write("error: cannot read file: %s\n" % e)
        return 2
    out = build_output(a, b)
    sys.stdout.buffer.write(b"".join(out))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
