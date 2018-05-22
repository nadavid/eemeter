from eemeter.weather import  WeatherSource
import pandas as pd
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset
import numpy as np
import json

from eemeter.modeling.models import CaltrackMonthlyModel, CaltrackDailyModel


def generate_date_range(start_date, end_date, interval='D'):
    """
    Generate pandas series of daterange.
    :param start_date:
    :param end_date:
    :param interval:
    :return:
    """
    date_obj = datetime.strptime(start_date, "%m/%d/%Y")
    end_date_obj = datetime.strptime(end_date, "%m/%d/%Y")
    series = pd.date_range(date_obj,end_date_obj,
                  freq=interval, tz='utc')
    if interval != 'H':
        series  = series + DateOffset(hours=0, minutes=0)
    return series


def generate_energy_data(start_date,
                         days,
                         station_code,
                         interval='D',
                         use_cz=True):
    """
    Generate dataframe with temperature and energy data between start date and end date
    :param start_date:
    :param days:
    :param station_code:
    :param interval:
    :param use_cz:
    :return:
    """
    normalized = False
    ss = generate_data(start_date, days, interval)
    ww = WeatherSource(station_code, normalized, use_cz)
    temp = ww.indexed_temperatures(ss, 'degF')
    if temp.empty:
        raise ValueError("Invalid")
    df = pd.DataFrame({ 'tempF' : temp}, index=temp.index)
    df['energy'] = df['tempF'] * 0.1 + np.random.normal(0.0, 0.25, size=len(df))
    df['station_code'] = station_code
    return df


def build_model(df):
    cc = CaltrackDailyModel()
    result = cc.fit(df)
    return cc


def read_test_data(fname):
    df = pd.read_csv(fname, parse_dates=['date'])
    df['date'] = pd.to_datetime(df.date)
    df = df.sort_values(by='date')
    df_index = df.set_index(pd.DatetimeIndex(df['date']))
    newdf = df_index.asfreq('d', method='pad')
    newdf.index = pd.to_datetime(newdf.index.tz_localize('UTC'))
    return newdf


if __name__ == '__main__':
    station_code = "720406"
    usage_start = "01/01/2015"
    usage_end_date = "12/31/2017"
    dt = generate_date_range(usage_start,
                             usage_end_date)

    normalized = False
    use_cz = True
    ww = WeatherSource(station_code, normalized, use_cz)
    temp  = ww.indexed_temperatures(dt, 'degF')
    df = pd.DataFrame({ 'tempF' : temp}, index=temp.index)
    df['energy'] = df['tempF'] * 0.1 + np.random.normal(0.0, 0.25, size=len(df))
    df['station_code'] = station_code

    baseline_end_date = "2016-12-15"
    baseline_data = df[df.index <= baseline_end_date]
    print ("Number of rows in baseline data ",  len(baseline_data))
    reporting_period_start_date = "2016-12-16"
    reporting_data = df[df.index >= reporting_period_start_date]
    print ("Number of rows in reporting data ",  len(reporting_data))

    baseline_model = build_model(baseline_data)
    predicted_df, variance_df = baseline_model.predict(reporting_data, summed=False)
    assert type(predicted_df) == pd.core.series.Series

    predicted_df = predicted_df.dropna()
    assert len(predicted_df) == len(reporting_data)
    print ("Numer of days for which prediction was success ", len(predicted_df))