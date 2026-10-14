"""The from-scratch training config for the per-variable FESOM streams."""

from weathergen.common import config


def test_fesom_vars_training_config_merges():
    cf = config.load_merge_configs(None, None, None, None, "config/config_fesom_vars.yml", {})
    assert set(cf.streams) == {"TEMP", "SALT", "U", "V", "SURFACE", "IFS_ATMO"}
    assert cf.freeze_modules == ""
    assert cf.healpix_level == 5 and cf.fe_num_blocks == 16 and cf.ae_global_dim_embed == 2048
    fc = cf.training_config.forecast
    assert fc.num_steps == 15 and fc.offset == 1 and fc.get("prefix_steps") is None
    assert str(cf.training_config.start_date).startswith("2000-01-01")
    assert str(cf.validation_config.start_date).startswith("2209-01-01")
