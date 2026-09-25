"""Single-root directed maximum spanning tree decoding.

Adapted from the Chu-Liu/Edmonds decoder in BaseUDParser, which credits AllenNLP.
The score matrix is indexed [head][dependent] and node 0 is ROOT.
"""


NEGATIVE = -1e9


def _cycle(parents: list[int]) -> list[int]:
    done = {0}
    for start in range(1, len(parents)):
        if start in done:
            continue
        seen = {}
        node = start
        while node not in done and node not in seen:
            seen[node] = len(seen)
            node = parents[node]
        done.update(seen)
        if node in seen:
            path = list(seen)
            return path[seen[node]:]
    return []


def _edmonds(scores: list[list[float]]) -> list[int]:
    n = len(scores)
    parents = [-1] + [max((h for h in range(n) if h != dep),
                          key=lambda h: scores[h][dep]) for dep in range(1, n)]
    cycle = _cycle(parents)
    if not cycle:
        return parents

    cycle_set = set(cycle)
    outside = [node for node in range(n) if node not in cycle_set]
    old_to_new = {old: new for new, old in enumerate(outside)}
    contracted = len(outside)
    size = contracted + 1
    reduced = [[NEGATIVE] * size for _ in range(size)]
    in_choice, out_choice = {}, {}
    cycle_weight = sum(scores[parents[dep]][dep] for dep in cycle)
    for head in outside:
        for dep in outside:
            reduced[old_to_new[head]][old_to_new[dep]] = scores[head][dep]
        entering = max(cycle, key=lambda dep: scores[head][dep] - scores[parents[dep]][dep])
        reduced[old_to_new[head]][contracted] = (
            cycle_weight + scores[head][entering] - scores[parents[entering]][entering]
        )
        in_choice[head] = entering
    for dep in outside:
        leaving = max(cycle, key=lambda head: scores[head][dep])
        reduced[contracted][old_to_new[dep]] = scores[leaving][dep]
        out_choice[dep] = leaving

    reduced_parents = _edmonds(reduced)
    result = parents[:]
    for dep in outside:
        if dep == 0:
            continue
        parent = reduced_parents[old_to_new[dep]]
        result[dep] = out_choice[dep] if parent == contracted else outside[parent]
    incoming = reduced_parents[contracted]
    if incoming < 0:
        raise ValueError("Contracted nonroot cycle has no incoming edge")
    external_head = outside[incoming]
    result[in_choice[external_head]] = external_head
    return result


def decode_single_root(scores: list[list[float]]) -> list[int]:
    """Return one head per word, with exactly one ROOT arc and no cycles."""
    n = len(scores)
    if n < 2 or any(len(row) != n for row in scores):
        raise ValueError("Expected a square score matrix with ROOT and words")
    sanitized = [row[:] for row in scores]
    for head in range(n):
        sanitized[head][0] = NEGATIVE
        sanitized[head][head] = NEGATIVE
    initial = _edmonds(sanitized)
    root_children = [dep for dep in range(1, n) if initial[dep] == 0]
    if len(root_children) == 1:
        return initial
    # The unconstrained arborescence often has only a few ROOT children.
    # Re-run with each such child as the sole permitted ROOT dependent.
    best_heads, best_score = None, float("-inf")
    for candidate in root_children:
        restricted = [row[:] for row in sanitized]
        for dep in range(1, n):
            if dep != candidate:
                restricted[0][dep] = NEGATIVE
        heads = _edmonds(restricted)
        score = sum(sanitized[heads[dep]][dep] for dep in range(1, n))
        if score > best_score:
            best_score, best_heads = score, heads
    if best_heads is None:
        raise ValueError("Could not decode a rooted tree")
    return best_heads
