"""The training loop collects garbage after releasing each step's batch (stopgap)."""

import inspect

from weathergen.train.trainer import Trainer


def test_train_loop_collects_garbage_after_del():
    source = inspect.getsource(Trainer.train)
    del_line = "del batch, x, preds, targets_and_auxs, loss"
    gc_line = "gc.collect()"
    assert del_line in source
    assert gc_line in source
    assert source.index(gc_line) > source.index(del_line)
