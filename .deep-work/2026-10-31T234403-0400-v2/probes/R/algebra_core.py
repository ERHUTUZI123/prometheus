"""Inspected stdlib-only exploratory algebra executor; no neural acquisition."""
import itertools


def normal_form(word, rules, budget=1000):
    word = tuple(word)
    used = 0
    while True:
        changed = False
        for pos in range(len(word)):
            for left, right in rules:
                if word[pos:pos + len(left)] == left:
                    word = word[:pos] + right + word[pos + len(left):]
                    used += 1
                    if used > budget:
                        raise RuntimeError("rewrite budget exhausted")
                    changed = True
                    break
            if changed:
                break
        if not changed:
            return word, used


def stack_rules():
    operators = ("Ca", "Cb", "Da", "Db")
    rules = []
    for i in "ab":
        for j in "ab":
            rules.append((("D" + i, "C" + j), () if i == j else ("Z",)))
        rules.append((("D" + i, "E"), ("Z",)))
    for token in operators + ("E", "Z"):
        if token != "E":
            rules.append(((token, "Z"), ("Z",)))
        rules.append((("Z", token), ("Z",)))
    return tuple(rules)


def reference_stack(actions):
    contents = []
    invalid = False
    for action in actions:
        if invalid:
            continue
        symbol = action[1]
        if action[0] == "C":
            contents.insert(0, symbol)
        elif contents and contents[0] == symbol:
            contents.pop(0)
        else:
            invalid = True
    return ("Z",) if invalid else tuple("C" + x for x in contents) + ("E",)


def counter_rules():
    rules = [(("x", "X"), ()), (("X", "x"), ()),
             (("y", "Y"), ()), (("Y", "y"), ())]
    for second in ("y", "Y"):
        for first in ("x", "X"):
            rules.append(((second, first), (first, second)))
    return tuple(rules)


def reference_counters(actions):
    x = actions.count("x") - actions.count("X")
    y = actions.count("y") - actions.count("Y")
    return (("x",) * max(x, 0) + ("X",) * max(-x, 0)
            + ("y",) * max(y, 0) + ("Y",) * max(-y, 0))


def all_words(alphabet, max_length):
    for length in range(max_length + 1):
        yield from itertools.product(alphabet, repeat=length)
