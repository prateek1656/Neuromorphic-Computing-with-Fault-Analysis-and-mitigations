"""Systematic sweep, not spot checks: every config field must either (a)
reject an out-of-physical-range value clearly, or (b) be a deliberate,
documented "anything goes" field. Silence by omission is what let a typo'd
redundancy_scheme run an entire experiment under the wrong scheme,
indistinguishable from a correctly-configured run (see
test_known_bug_regressions.py) - this file is the systematic version of that
one finding, covering every field, not just the two already caught.
"""

from __future__ import annotations

import pytest

from neurofault.config import (
    CrossbarConfig,
    DeviceConfig,
    ExperimentConfig,
    FaultConfig,
    MitigationConfig,
)


class TestDeviceConfigValidation:
    def test_rejects_non_positive_r_on(self):
        with pytest.raises(ValueError, match="r_on/r_off must be positive"):
            DeviceConfig(r_on=0.0, r_off=1000.0)

    def test_rejects_non_positive_r_off(self):
        with pytest.raises(ValueError, match="r_on/r_off must be positive"):
            DeviceConfig(r_on=100.0, r_off=-5.0)

    def test_rejects_r_on_greater_than_or_equal_to_r_off(self):
        with pytest.raises(ValueError, match="r_on must be less than r_off"):
            DeviceConfig(r_on=1000.0, r_off=1000.0)
        with pytest.raises(ValueError, match="r_on must be less than r_off"):
            DeviceConfig(r_on=2000.0, r_off=1000.0)

    def test_rejects_non_positive_time_series_resolution(self):
        with pytest.raises(ValueError, match="time_series_resolution must be positive"):
            DeviceConfig(time_series_resolution=0.0)

    def test_accepts_valid_values(self):
        DeviceConfig(r_on=1.0, r_off=1.0001, time_series_resolution=1e-12)  # must not raise


class TestCrossbarConfigValidation:
    @pytest.mark.parametrize("adc_resolution,dac_resolution", [(0, 8), (8, 0), (-1, 8)])
    def test_rejects_non_positive_adc_dac_resolution(self, adc_resolution, dac_resolution):
        with pytest.raises(ValueError, match="adc_resolution/dac_resolution must be >= 1"):
            CrossbarConfig(adc_resolution=adc_resolution, dac_resolution=dac_resolution)

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_rejects_out_of_range_redundancy_factor(self, value):
        with pytest.raises(ValueError, match="redundancy_factor must be in"):
            CrossbarConfig(redundancy_factor=value)

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_rejects_out_of_range_condemn_threshold(self, value):
        with pytest.raises(ValueError, match="redundancy_condemn_threshold must be in"):
            CrossbarConfig(redundancy_condemn_threshold=value)

    def test_rejects_devices_per_synapse_below_one(self):
        with pytest.raises(ValueError, match="devices_per_synapse must be >= 1"):
            CrossbarConfig(devices_per_synapse=0)

    @pytest.mark.parametrize("tile_shape", [(0, 10), (10, 0), (-1, 10)])
    def test_rejects_non_positive_tile_shape_dimensions(self, tile_shape):
        with pytest.raises(ValueError, match="tile_shape dimensions must be positive"):
            CrossbarConfig(tile_shape=tile_shape)

    def test_rejects_unknown_circuit_topology(self):
        with pytest.raises(ValueError, match="Unknown crossbar_config.circuit_topology"):
            CrossbarConfig(circuit_topology="bogus")

    def test_accepts_none_circuit_topology(self):
        CrossbarConfig(circuit_topology=None)  # must not raise - the documented default

    def test_rejects_unknown_redundancy_scheme(self):
        with pytest.raises(ValueError, match="Unknown crossbar_config.redundancy_scheme"):
            CrossbarConfig(redundancy_scheme="bogus")

    def test_accepts_valid_values(self):
        CrossbarConfig(
            adc_resolution=1,
            redundancy_factor=0.0,
            redundancy_condemn_threshold=1.0,
            devices_per_synapse=1,
            tile_shape=(1, 1),
            circuit_topology="1T1R",
            redundancy_scheme="row_col_granularity",
        )  # must not raise - boundary values are valid, not just interior ones


class TestFaultConfigValidation:
    def test_rejects_unknown_failure_model(self):
        with pytest.raises(ValueError, match="Unknown FaultConfig.failure_model"):
            FaultConfig(failure_model="bogus")

    def test_rejects_unknown_distribution(self):
        with pytest.raises(ValueError, match="Unknown FaultConfig.distribution"):
            FaultConfig(distribution="bogus")

    @pytest.mark.parametrize("density", [-0.1, 1.1])
    def test_rejects_out_of_range_density(self, density):
        with pytest.raises(ValueError, match="density must be in"):
            FaultConfig(density=density)

    def test_accepts_density_boundary_values(self):
        FaultConfig(density=0.0)
        FaultConfig(density=1.0)

    def test_rejects_non_positive_injection_interval(self):
        with pytest.raises(ValueError, match="injection_interval must be >= 1"):
            FaultConfig(injection_interval=0)

    def test_rejects_non_positive_weibull_shape(self):
        with pytest.raises(ValueError, match="weibull_shape must be positive"):
            FaultConfig(weibull_shape=0.0)

    def test_rejects_non_positive_weibull_scale_when_set(self):
        with pytest.raises(ValueError, match="weibull_scale must be positive"):
            FaultConfig(failure_model="weibull", weibull_scale=-1.0)

    def test_rejects_non_positive_exponential_rate_when_set(self):
        with pytest.raises(ValueError, match="exponential_rate must be positive"):
            FaultConfig(failure_model="exponential", exponential_rate=0.0)

    @pytest.mark.parametrize(
        "temperature_kelvin,reference_temperature_kelvin", [(0.0, 300.0), (300.0, -1.0)]
    )
    def test_rejects_non_positive_temperatures(
        self, temperature_kelvin, reference_temperature_kelvin
    ):
        with pytest.raises(ValueError, match="temperature_kelvin"):
            FaultConfig(
                temperature_kelvin=temperature_kelvin,
                reference_temperature_kelvin=reference_temperature_kelvin,
            )

    def test_still_requires_weibull_scale_and_exponential_rate_explicitly(self):
        """Pre-existing checks (added an earlier session) must still work
        alongside the new ones added here."""
        with pytest.raises(ValueError, match="requires weibull_scale"):
            FaultConfig(failure_model="weibull")
        with pytest.raises(ValueError, match="requires exponential_rate"):
            FaultConfig(failure_model="exponential")


class TestMitigationConfigValidation:
    @pytest.mark.parametrize("value", [-0.1, 100.1])
    def test_rejects_out_of_range_health_threshold(self, value):
        with pytest.raises(ValueError, match="health_threshold must be in"):
            MitigationConfig(health_threshold=value)

    def test_accepts_health_threshold_boundary_values(self):
        MitigationConfig(health_threshold=0.0)
        MitigationConfig(health_threshold=100.0)

    def test_rejects_non_positive_health_check_interval(self):
        with pytest.raises(ValueError, match="health_check_interval must be >= 1"):
            MitigationConfig(health_check_interval=0)

    def test_rejects_unknown_detection_method(self):
        with pytest.raises(ValueError, match="Unknown MitigationConfig.detection_method"):
            MitigationConfig(detection_method="bogus")

    def test_rejects_negative_checksum_z_threshold(self):
        with pytest.raises(ValueError, match="checksum_z_threshold must be >= 0"):
            MitigationConfig(checksum_z_threshold=-0.01)


class TestExperimentConfigValidation:
    def test_rejects_unknown_simulator(self):
        with pytest.raises(ValueError, match="Unknown ExperimentConfig.simulator"):
            ExperimentConfig(name="x", simulator="bogus")

    def test_rejects_unknown_dataset(self):
        with pytest.raises(ValueError, match="Unknown ExperimentConfig.dataset"):
            ExperimentConfig(name="x", dataset="bogus")

    @pytest.mark.parametrize(
        "field,value",
        [
            ("train_samples", 0),
            ("test_samples", 0),
            ("batch_size", 0),
            ("epochs", 0),
            ("num_batches", 0),
            ("eval_interval", 0),
        ],
    )
    def test_rejects_non_positive_count_fields(self, field, value):
        with pytest.raises(ValueError):
            ExperimentConfig(name="x", **{field: value})

    def test_rejects_non_positive_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            ExperimentConfig(name="x", learning_rate=0.0)

    def test_default_construction_does_not_raise(self):
        ExperimentConfig(name="x")  # every default value must itself be valid
