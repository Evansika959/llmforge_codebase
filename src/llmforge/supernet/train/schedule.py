import math


def cosine_warmup(step, total, warmup, base_lr, min_lr=0.0):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(1.0, p)))
