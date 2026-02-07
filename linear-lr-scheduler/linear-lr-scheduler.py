def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    if step > total_steps:
        return float(final_lr)

    if warmup_steps <= 0:
        if total_steps <= 0:
            return float(final_lr)
        return float(final_lr + (initial_lr - final_lr) * ((total_steps - step) / total_steps))

    if warmup_steps >= total_steps:
        if step < warmup_steps:
            return float(step * initial_lr / warmup_steps)
        return float(final_lr)

    if step < warmup_steps:
        return float(step * initial_lr / warmup_steps)

    return float(final_lr + (initial_lr - final_lr) * ((total_steps - step) / (total_steps - warmup_steps)))