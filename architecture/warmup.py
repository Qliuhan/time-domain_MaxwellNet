import numpy as np

def wram_up(warmup_start_lr, warmup_steps, max_iter, lr0, power):
    warmup_factor = (lr0 / warmup_start_lr) ** (1 / warmup_steps)
    def get_lr(t):
        if t <= warmup_steps:
            lr = warmup_start_lr * (warmup_factor ** t)
        else:
            factor = (1 - (t - warmup_steps) / (max_iter - warmup_steps)) ** power
            lr = lr0 * factor

        return lr

    steps = np.arange(max_iter)
    lrs = list(map(get_lr, steps))
    return lrs
