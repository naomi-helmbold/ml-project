def is_even(n: int) -> bool:
    return not bool(n%2)

def is_odd(n: int) -> bool:
    return not is_even(n)
