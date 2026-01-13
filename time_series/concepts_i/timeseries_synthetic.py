''' 
Time Series Process and Synthetic
''' 

##!pip install git+https://github.com/timesynth/timesynth.git

import timesynth as ts
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
np.random.seed()

def timeseries_visualize(time, values, label, legends=None):
  #if legends is not None: assert len(legends) == len(values)
  if isinstance(values, list):
    seriesdict = {'Time': time}
    for value, legend in zip(values, legends):
      seriesdict[legend] = value
    plot_df = pd.DataFrame(seriesdict)
    plot_df = pd.melt(plot_df, id_vars='Time', var_name='ts', value_name='Value')
  else:
    seriesdict = {'Time': time, 'Value': values, 'ts:': ""}
    plot_df = pd.DataFrame(seriesdict)

  if isinstance(values, list):
    fig = px.line(plot_df, x='Time', y='Value', line_dash='ts')
  else:
    fig = px.line(plot_df, x='Time', y='Value')

  fig.update_layout(
      autosize=False, width=600, height=400,
      title={'text': label, 'y':0.9, 'x':0.5,
      'xanchor': 'center', 'yanchor': 'top'}, titlefont={'size': 25},
      yaxis=dict(title_text='Value'), xaxis=dict(title_text='Time')
  )
  return fig.show()

def generate_time_series(signal, noise=None):
  ts_sample = ts.TimeSampler(stop_time=20)
  regular_timeseries_sample = ts_sample.sample_regular_time(num_points=100)
  timeseries = ts.TimeSeries(signal_generator=signal, noise_generator=noise)
  samples, signals, errors = timeseries.sample(regular_timeseries_sample)
  return samples, regular_timeseries_sample, signals, errors

time, values  = np.arange(100), np.random.randn(100)*100
timeseries_visualize(time, values, "", 'white noise')

r = 0.4
time = np.arange(100)
white_noise = np.random.rand(100)*100
values = np.zeros(100)
for i, v in enumerate(white_noise):
  if i ==0:
    values[i] = v
  else:
    values[i] = r*values[i-1] + np.sqrt(1 - np.power(r, 2))*v #r*white_noise[i]
timeseries_visualize(time, values, "", 'red noise')

''' sinusoidal '''
s1 = ts.signals.Sinusoidal(amplitude=1.5, frequency=0.5)
s2 = ts.signals.Sinusoidal(amplitude=1, frequency=0.5)
samples_1, regular_timesamples, signals_1, errors_1 = generate_time_series(s1)
samples_2, regular_timesamples, signals_2, errors_2 = generate_time_series(s2)
figure = timeseries_visualize(regular_timesamples, [samples_1, samples_2], 'sinusoidal',
           legends = ['amplitude 1.5, freq=0.25', 'amplitude 1, freq=0.25']
)
figure

''' pseudo-periodics '''
pp_s1 = ts.signals.Sinusoidal(amplitude=1.5, frequency=0.5)
samples, regular_time_samples, signals, errors = generate_time_series(pp_s1)
timeseries_visualize(regular_time_samples, samples, '')

''' auto-regressive '''
from autoregressive import AutoRegressive
ar_s1 = AutoRegressive(ar_param=[1.5, -0.75])
samples, regular_time_samples, signals, errors = generate_time_series(ar_s1)
timeseries_visualize(regular_time_samples, samples, 'auto-regressive ts')

ts_mix = ts.TimeSeries(signal_generator=ar_s1, noise_generator=s2)
ts_mix

ar_samples, regular_time_samples, _, _ = generate_time_series(ar_s1)
pp_samples, regular_time_samples, _, _ = generate_time_series(pp_s1)
ts_mix_ = pp_samples*2+ar_samples
timeseries_visualize(regular_time_samples, ts_mix_, 'combining two ts')

''' non-stationary sinusoidal with trend and white noise '''
s_ts = ts.signals.Sinusoidal(amplitude=1.5, frequency=0.5)
noise = ts.noise.GaussianNoise(std=.3)
s_samples, regular_time_samples, _, _ = generate_time_series(s_ts, noise)
timeseries_visualize(regular_time_samples, s_samples, 'non-stationary ts')

trend = regular_time_samples*.4
ts_ = s_samples+trend
timeseries_visualize(regular_time_samples, ts_, 'non-stationary ts with trend')

s_samples, regular_time_samples, _, _ = generate_time_series(s_ts)
noise = [np.random.randn()*np.sqrt(i) for i, value in enumerate(regular_time_samples)]
timeseries_visualize(regular_time_samples, ts_, 'non-stationary ts with trend')