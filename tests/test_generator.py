from eemeter.generator import ConsumptionGenerator
from eemeter.generator import ProjectGenerator
from eemeter.generator import generate_periods

from eemeter.consumption import DatetimePeriod
from eemeter.models.temperature_sensitivity import DoubleBalancePointModel
from eemeter.models.temperature_sensitivity import PRISMModel

from fixtures.weather import gsod_722880_2012_2014_weather_source

import pytest
from datetime import datetime
from datetime import timedelta
from numpy.testing import assert_almost_equal
from scipy.stats import uniform

@pytest.fixture
def periods_one_year():
    return [DatetimePeriod(datetime(2012,1,1),datetime(2012,2,1)),
            DatetimePeriod(datetime(2012,2,1),datetime(2012,3,1)),
            DatetimePeriod(datetime(2012,3,1),datetime(2012,4,1)),
            DatetimePeriod(datetime(2012,4,1),datetime(2012,5,1)),
            DatetimePeriod(datetime(2012,5,1),datetime(2012,6,1)),
            DatetimePeriod(datetime(2012,6,1),datetime(2012,7,1)),
            DatetimePeriod(datetime(2012,7,1),datetime(2012,8,1)),
            DatetimePeriod(datetime(2012,8,1),datetime(2012,9,1)),
            DatetimePeriod(datetime(2012,9,1),datetime(2012,10,1)),
            DatetimePeriod(datetime(2012,10,1),datetime(2012,11,1)),
            DatetimePeriod(datetime(2012,11,1),datetime(2012,12,1)),
            DatetimePeriod(datetime(2012,12,1),datetime(2013,1,1))]

@pytest.mark.slow
def test_consumption_generator_no_base_load(periods_one_year,gsod_722880_2012_2014_weather_source):
    model = DoubleBalancePointModel()
    params = 1.0,1.0,0.0,65.0,10.0
    gen = ConsumptionGenerator("electricity", "J", "degF", model, params)
    consumptions = gen.generate(gsod_722880_2012_2014_weather_source, periods_one_year)
    consumption_joules = [c.to("J") for c in consumptions]
    assert_almost_equal(consumption_joules, [241.5, 279.2, 287.6, 139.2, 56.2, 2.1, 22.6, 154.3, 106.8, 53.4, 137.9, 351.1])

@pytest.mark.slow
def test_consumption_generator_with_base_load(periods_one_year,gsod_722880_2012_2014_weather_source):
    model = DoubleBalancePointModel()
    params = 1.0,1.0,1.0,65.0,10.0
    gen = ConsumptionGenerator("electricity", "J", "degF", model,params)
    consumptions = gen.generate(gsod_722880_2012_2014_weather_source, periods_one_year)
    consumption_joules = [c.to("J") for c in consumptions]
    assert_almost_equal(consumption_joules, [272.5, 308.2, 318.6, 169.2, 87.2, 32.1, 53.6, 185.3, 136.8, 84.4, 167.9, 382.1])

@pytest.mark.slow
def test_project_generator(gsod_722880_2012_2014_weather_source):
    electricity_model = DoubleBalancePointModel()
    gas_model = PRISMModel()
    electricity_param_distributions = (
            uniform(loc=1, scale=.5),
            uniform(loc=1, scale=.5),
            uniform(loc=5, scale=5),
            uniform(loc=60, scale=5),
            uniform(loc=2, scale=12))
    electricity_param_delta_distributions = (
            uniform(loc=-.2,scale=.3),
            uniform(loc=-.2, scale=.3),
            uniform(loc=-2, scale=3),
            uniform(loc=0, scale=0),
            uniform(loc=0, scale=0))
    gas_param_distributions = (
            uniform(loc=60, scale=5),
            uniform(loc=5, scale=5),
            uniform(loc=1, scale=.5))
    gas_param_delta_distributions = (
            uniform(loc=0, scale=0),
            uniform(loc=-2, scale=3),
            uniform(loc=-.2,scale=.3))

    generator = ProjectGenerator(electricity_model, gas_model,
                                electricity_param_distributions,
                                electricity_param_delta_distributions,
                                gas_param_distributions,
                                gas_param_delta_distributions)

    periods = generate_periods(datetime(2012,1,1),datetime(2013,1,1),period_jitter=timedelta(days=0))

    retrofit_start_date = datetime(2012,4,1)
    retrofit_completion_date = datetime(2012,5,1)

    elec_consumption, gas_consumption = generator.generate(gsod_722880_2012_2014_weather_source, periods, periods,
                                                           retrofit_start_date, retrofit_completion_date,
                                                           electricity_noise=None,gas_noise=None)

    elec_kwh = [c.to("kWh") for c in elec_consumption]
    gas_therms = [c.to("therms") for c in gas_consumption]

    assert len(elec_kwh) == 12
    assert len(gas_therms) == 12
    assert elec_kwh[0] > elec_kwh[5]
    assert gas_therms[0] > gas_therms[5]
    assert elec_kwh[0] < 600 # could probably lower this upper bound
    assert gas_therms[0] < 600 # could probably lower this upper bound

def test_generate_periods():
    uniform_periods = generate_periods(datetime(2012,1,1),datetime(2013,1,1),period_jitter=timedelta(days=0))
    assert uniform_periods[0].start == datetime(2012,1,1)
    assert uniform_periods[0].end == datetime(2012,1,31)
    assert uniform_periods[11].start == datetime(2012,11,26)
    assert uniform_periods[11].end == datetime(2012,12,26)

    short_uniform_periods = generate_periods(datetime(2012,1,1),datetime(2013,1,1),period_length_mean=timedelta(days=1),period_jitter=timedelta(days=0))
    assert short_uniform_periods[0].start == datetime(2012,1,1)
    assert short_uniform_periods[0].end == datetime(2012,1,2)
    assert short_uniform_periods[330].start == datetime(2012,11,26)
    assert short_uniform_periods[330].end == datetime(2012,11,27)

    jittery_periods = generate_periods(datetime(2012,1,1),datetime(2013,1,1))
    for p in jittery_periods:
        assert p.end - p.start < timedelta(days=34)
        assert p.end - p.start > timedelta(days=26)

    assert not all([p.end - p.start == timedelta(days=30) for p in jittery_periods])
