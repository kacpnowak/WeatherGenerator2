"""Where a continued run resumes: the mini-epoch and the optimizer steps already done in it."""

import inspect

from weathergen.train.trainer import Trainer, resume_istep, resume_position


def test_fresh_run():
    assert resume_position(0, 256, 10) == (0, 0)


def test_mid_epoch():
    assert resume_position(960, 256, 10) == (3, 192)


def test_epoch_boundary():
    assert resume_position(768, 256, 10) == (3, 0)


def test_complete():
    assert resume_position(2560, 256, 10) == (10, 0)


def test_beyond_complete():
    assert resume_position(2688, 256, 10) == (10, 0)


def test_world_size_changed_restarts_the_epoch():
    assert resume_position(960, 256, 10, world_size_changed=True) == (3, 0)


def test_world_size_changed_restarts_the_step_counter():
    # the restarted mini-epoch is counted from its first step, so the run still ends at 2560
    assert resume_istep(960, 256, 10, world_size_changed=True) == 768


def test_step_counter_kept_without_world_size_change():
    assert resume_istep(960, 256, 10) == 960
    # a completed run is not clamped
    assert resume_istep(2688, 256, 10) == 2688


def test_world_size_changed_after_completion():
    for istep in (2560, 2688):
        assert resume_position(istep, 256, 10, world_size_changed=True) == (10, 0)
        assert resume_istep(istep, 256, 10, world_size_changed=True) == 2560


def test_run_resumes_before_building_the_lr_scheduler():
    # the scheduler continues from the realigned step counter
    source = inspect.getsource(Trainer.run)
    assert source.index("self.cf.general.istep = resume_istep(") < source.index(
        "LearningRateScheduler("
    )


def test_run_sets_sampler_position_before_training():
    source = inspect.getsource(Trainer.run)
    set_line = "self.dataset.set_resume_position(mini_epoch_base, steps_done)"
    assert set_line in source
    assert source.index(set_line) < source.index("self.validate_before_training()")
    assert source.index(set_line) < source.index("self.train(mini_epoch)")


def test_latest_saved_after_step_count():
    # the mid-epoch _latest save records the number of completed optimizer steps
    source = inspect.getsource(Trainer.train)
    assert source.index("self.save_model(-1)") > source.index("self.cf.general.istep += 1")


def test_latest_saved_at_end_of_each_mini_epoch():
    source = inspect.getsource(Trainer.run)
    epoch_save = source.index("self.save_model(mini_epoch)")
    assert source.index("self.save_model(-1)", epoch_save) < source.index(
        "self.save_model(self.training_cfg.num_mini_epochs)"
    )
