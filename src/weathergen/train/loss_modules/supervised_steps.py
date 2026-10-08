"""Which forecast steps of a rollout carry predictions."""


def first_supervised_timestep(preds_physical: list[dict], stream_name: str) -> int | None:
    """
    Index of the first forecast step with predictions for `stream_name` (None if none).

    Without a training spin-up this is the first forecast step; with one
    (forecast.prefix_steps) it is the first step after the spin-up, whose error the dynamic
    channel weights track instead of the never-decoded first forecast step.
    """
    return next((i for i, step in enumerate(preds_physical) if step.get(stream_name)), None)
