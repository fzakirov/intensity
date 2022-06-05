import mne
import os
import numpy as np
from mne.preprocessing import ICA
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import peak_prominences

'''
rejects epochs instance by n*sd

epochs - epochs instance
sd - number of std for thresholding

operates in place, returns number of rejected epochs
'''
def drop_epochs_by_sigma(epochs, sd):
    maximas=[]
    for i in range(epochs.get_data().shape[0]):
        maximas.append(epochs.get_data()[i].max(axis=1))

    maximas=np.array(maximas)
    maximas_std=maximas.std(axis=0)
    maximas_mean=maximas.mean(axis=0)

    th_max=maximas_mean+sd*maximas_std
    th_min=maximas_mean-sd*maximas_std
    rej=[]
    for i in range(epochs.get_data().shape[0]):
        for j in range(epochs.get_data().shape[1]):
            if maximas[i,j]>th_max[j] or maximas[i,j]<th_min[j]:
                rej.append(i)
    print()
    reject_list=list(set(rej))
    epochs.drop(reject_list)

'''
reads and makes a list of preprocessed raw files (no bad chs + rereferencing + ica)

p - list of str IDs
path - path to raw instances

returns list of raws
'''
def read_clean_raws (p, path):
    raws=[]
    for i in p:
        raw=mne.io.read_raw(path+i+'-raw.fif', preload=True)
        raws.append(raw)
    return raws

'''
creates epochs from raws with no reject threshold (INTENSITY ONLY)

raws - list of raw instances
sd - whether +-3*sd rejection is needed (default=False)
baseline - baseline tuple (default=(-0.2, 0.0))
sr - sampling frequency to resample (default=250)
tmin - tmin to crop (default=-0.2)
tmax - tmax to crop (default=0.7)

returns 4 lists of epochs (50,60,70,80dB)
'''
def epo_from_raw (raws, sd=False, baseline=(-0.2, 0.0), sr=250, tmin=-0.2, tmax=0.7):

  list_of_epo_50=[]
  list_of_epo_60=[]
  list_of_epo_70=[]
  list_of_epo_80=[]

  for i,c in enumerate(raws):

    events=mne.find_events(c)
    epochs=mne.Epochs(c, events, event_id={'80dB':1, '70dB':2,'60dB':3, '50dB':4},
                  #reject=dict(eeg=150e-6),
                   baseline=None, tmin=-0.5, tmax=0.8, preload=True)

    epochs=epochs.resample(sr).crop(tmin, tmax).apply_baseline(baseline)

    if sd==True:
        drop_epochs_by_sigma(epochs, 3)

    epo_50=epochs['50dB']
    epo_60=epochs['60dB']
    epo_70=epochs['70dB']
    epo_80=epochs['80dB']

    list_of_epo_50.append(epo_50)
    list_of_epo_60.append(epo_60)
    list_of_epo_70.append(epo_70)
    list_of_epo_80.append(epo_80)

  return(list_of_epo_50,
         list_of_epo_60,
         list_of_epo_70,
         list_of_epo_80)

'''
creates evokeds from epochs (INTENSITY ONLY)

p - list of str IDs
epo_50, etc - list of epochs 50dB
baseline - baseline tuple (default=(-0.2, 0.0))

returns 4 lists of evokeds (50,60,70,80dB)
'''
def evo_from_epo (p, epo_50, epo_60, epo_70, epo_80, baseline=(-0.2, 0.0)):
  list_of_evo_50=[]
  list_of_evo_60=[]
  list_of_evo_70=[]
  list_of_evo_80=[]

  for i,c in enumerate(p):

    evo_50=epo_50[i].average().apply_baseline(baseline)
    evo_60=epo_60[i].average().apply_baseline(baseline)
    evo_70=epo_70[i].average().apply_baseline(baseline)
    evo_80=epo_80[i].average().apply_baseline(baseline)

    list_of_evo_50.append(evo_50)
    list_of_evo_60.append(evo_60)
    list_of_evo_70.append(evo_70)
    list_of_evo_80.append(evo_80)

  return(list_of_evo_50,
         list_of_evo_60,
         list_of_evo_70,
         list_of_evo_80)

'''
makes list of TD evokeds for given age range from existing epochs and TD_INFO df (used for Zakl)

low - int for min age
hight int for max age 
path_info - path to TD_INFO df
path_epo - path to TD epochs
baseline - baseline tuple (default=(-0.2, 0.0))

returns 4 lists of evokeds (50,60,70,80dB)
'''
def create_zakl_evokeds (low, high, path_info, path_epo, baseline=(-0.2, 0.0)):
  ages_td=pd.read_csv(path_info+'td_df.csv', dtype=str)
  ages_td['AGE']=ages_td['AGE'].astype(float)
  p=list(ages_td[(ages_td['AGE']>=low) & (ages_td['AGE']<=high)]['ID'])

  list_of_evo_50=[]
  list_of_evo_60=[]
  list_of_evo_70=[]
  list_of_evo_80=[]

  for i in p: #p - list of str IDs
    epo=mne.read_epochs(path_epo+i+'-epo.fif')
    epo=epo.apply_baseline(baseline).crop(-0.2,0.7)

    evo_50=epo['50dB'].average()
    evo_60=epo['60dB'].average()
    evo_70=epo['70dB'].average()
    evo_80=epo['80dB'].average()

    list_of_evo_50.append(evo_50)
    list_of_evo_60.append(evo_60)
    list_of_evo_70.append(evo_70)
    list_of_evo_80.append(evo_80)

  return(list_of_evo_50, list_of_evo_60,
           list_of_evo_70, list_of_evo_80)

'''
plot and compare TD and SUBJECT (used for Zakl)

ch_name - str channel to plot, ex. 'FCz' 
list_of_evoked_1 - list of TD evokeds
evoked_sub_1 - evoked instance of SUBJECT
condition_title - str condition, '50dB'
sub - str SUBJECT ID
std_num - number of std to plot for TD (default=2)

returns plot
'''
def plot_evoked (ch_name, list_of_evoked_1, evoked_sub_1,
                condition, sub, std_num=2):

  ga_evoked_1=mne.grand_average(list_of_evoked_1)
  ga_evoked_1=ga_evoked_1.resample(200)
  ch_num=ga_evoked_1.ch_names.index(ch_name)

  ga_evoked_array=ga_evoked_1.data[ch_num]
  for_std=np.array([ev.resample(200).data[ch_num] for ev in (list_of_evoked_1)])
  gstd_evoked_array=np.std(for_std,0)

  evoked_sub_1=evoked_sub_1.apply_baseline((-0.2, 0.0))
  evoked_sub_1=evoked_sub_1.resample(200)
  ch_num=evoked_sub_1.ch_names.index(ch_name)
  evoked_sub_1_array=evoked_sub_1.data[ch_num]

  #gstd_evoked_array=scipy.stats.sem(for_std)

  plt.plot(ga_evoked_1.times,ga_evoked_array*10**6,'b',label='TD (n='+str(len(p))+') '+condition)
  plt.plot(ga_evoked_1.times,evoked_sub_1_array*10**6,'r',label=sub+' '+condition)

  #plt.plot(ga_evoked_array+gstd_evoked_array)
  #plt.plot(ga_evoked_array-gstd_evoked_array)
  plt.fill_between(ga_evoked_1.times,ga_evoked_array*10**6 - std_num*gstd_evoked_array*10**6, ga_evoked_array*10**6 + std_num*gstd_evoked_array*10**6,'b',alpha=0.3)
  plt.axvline(x=0, linestyle="--", color="black")
  plt.axhline(y=0, linestyle="--", color="black")
  limits = [-0.5, 0.8, -10, 10]
  plt.axis(limits)
  plt.margins(x=0)
  plt.title(ch_name)
  plt.ylabel('Amplitude, μV')
  plt.xlabel('Time, s')

  plt.legend()
  return(plt)

'''
merges existing PDFs into 1 .pdf

p - str list of IDs
path - path to PDFs
title - str name for merged .pdf, (default='result')

saves .pdf file into the path
'''
def merge_pdf (p, path, title='result'):

  pdfs=map((lambda a: path+a+'.pdf'), p)
  pdfs=list(pdfs)

  from PyPDF2 import PdfFileMerger

  merger = PdfFileMerger()

  for pdf in pdfs:
      merger.append(pdf)

  merger.write(path+title+'pdf')
  merger.close()

'''
plot and compare 2 SINGLE subject evokeds

ch_name - str channel to plot, ex. 'FCz' 
evoked_1 - 1st subject evoked instance
evoked_2 - 2nd subject evoked instance
condition_title - str condition, '50dB'
sub1 - str 1st subject ID
sub2 - str 2nd subject ID
invert - whether inversion is needed (default=False)

returns plot
'''
def plot_evoked_single (ch_name, evoked_1, evoked_2,
                condition, sub_1, sub_2):

  times=evoked_1.times

  ch_num=evoked_1.ch_names.index(ch_name)
  evoked_1=evoked_1.apply_baseline((-0.2, 0.0))
  evoked_array_1=evoked_1.data[ch_num]

  ch_num=evoked_2.ch_names.index(ch_name)
  evoked_2=evoked_2.apply_baseline((-0.2, 0.0))
  evoked_array_2=evoked_2.data[ch_num]

  plt.plot(times,evoked_array_1*10**6,'b',label=sub_1+condition)
  plt.plot(times,evoked_array_2*10**6,'r',label=sub_2+condition)

  plt.axvline(x=0, linestyle='--', color='black')
  plt.axhline(y=0, linestyle='--', color='black')
  limits=[-0.5, 0.8, -10, 10]
  plt.axis(limits)
  plt.margins(x=0)
  plt.title(ch_name)
  plt.ylabel('Amplitude, μV')
  plt.xlabel('Time, s')

  plt.legend()
  return(plt)

'''
creates df from epochs

epochs - epochs instance 
cond - str condition, ex. '50dB'
chan - str channel, ex. 'FCz'
delay - whether delay is needed (default=False)
invert - whether inversion is needed (default=False)

returns df with epochs data
'''
def to_dataframe_new (epochs, cond, chan, delay=False, invert=False):
  if delay==True:
    evoked=epochs[cond].average().apply_baseline((0.0, 0.3)).resample(100)
  else:
    evoked=epochs[cond].average().apply_baseline((-0.2, 0.0)).resample(100)

  df=evoked.to_data_frame()

  times=[]
  for i in range(-500, 800, 10):
    times.append(i)

  df['time']=times
  df=df[[chan, 'time']]

  if invert==True:
    df[chan]=df[chan]*(-1)

  return(df)

'''
creates GA from epochs (INTENSITY ONLY)

sub - list of str IDs
path - path to epochs
baseline - baseline tuple (default=(-0.2, 0.0))

returns GA instances for each of 4 conditions
'''
def create_ga (sub, path, baseline=(-0.2,0)):
  list_of_evo_80=[]
  list_of_evo_70=[]
  list_of_evo_60=[]
  list_of_evo_50=[]
  for i,c in enumerate(sub):
    epochs=mne.read_epochs(path+c+'-epo.fif').resample(250).crop(-0.4, 0.7)
    epochs=epochs.apply_baseline(baseline)

    evo_80=epochs['80dB'].average()
    evo_70=epochs['70dB'].average()
    evo_60=epochs['60dB'].average()
    evo_50=epochs['50dB'].average()

    list_of_evo_80.append(evo_80)
    list_of_evo_70.append(evo_70)
    list_of_evo_60.append(evo_60)
    list_of_evo_50.append(evo_50)

  ga_80=mne.grand_average(list_of_evo_80)
  ga_70=mne.grand_average(list_of_evo_70)
  ga_60=mne.grand_average(list_of_evo_60)
  ga_50=mne.grand_average(list_of_evo_50)

  return(ga_50, ga_60, ga_70, ga_80)

'''
saves evokeds to .csv (INTENSITY ONLY)

epochs - epochs instance
baseline - baseline tuple (default=(-0.2, 0.0))
title - .csv title
path - path to save .csv
'''
def evoked_to_csv (epochs, baseline=(-0.2, 0.0), title="", path=""):
    epochs=epochs.apply_baseline(baseline)
    evoked_50=epochs['50dB'].average()
    evoked_60=epochs['60dB'].average()
    evoked_70=epochs['70dB'].average()
    evoked_80=epochs['80dB'].average()

    tb_pnd=evoked_50.to_data_frame()
    tb_pnd.to_csv(path+title+'_evo_50.csv',sep=';')

    tb_pnd=evoked_60.to_data_frame()
    tb_pnd.to_csv(path+title+'_evo_60.csv',sep=';')

    tb_pnd=evoked_70.to_data_frame()
    tb_pnd.to_csv(path+title+'_evo_70.csv',sep=';')

    tb_pnd=evoked_80.to_data_frame()
    tb_pnd.to_csv(path+title+'_evo_80.csv',sep=';')

'''
calculates ERP components amplitude and latencies 
find_comp and find_latency for positive components
find_comp_neg and find_latency_neg for negative components

data - df from
comp - str ERP component to calculate, ex. 'P2'
condition - str condition, ex. '50dB'
nsubj - number of subjects
df - 
chan - str channel to calculate, ex. 'FCz'

adds calculates values to data
'''
def find_comp(data, comp, condition, nsubj, df, chan):
    imax=data[data['time'].isin(df[df.ID==nsubj][comp+'_window'].iloc[0])][chan].mean()
    df.loc[(df['ID']==nsubj)&(df['COND']==condition),comp+'_AMP']=imax
    return imax

def find_comp_neg(data, comp, condition, nsubj, df, chan):
    imax=data[data['time'].isin(df[df.ID==nsubj][comp+'_window'].iloc[0])][chan].mean()
    df.loc[(df['ID']==nsubj)&(df['COND']==condition),comp+'_AMP']=imax
    return imax

def find_latency(data, comp, condition, nsubj, df, chan):
    imax=data[data['time'].isin(df[df.ID==nsubj][comp+'_window'].iloc[0])][chan].max()
    latency=data[data[chan]==imax]['time'].iloc[0]
    df.loc[(df['ID']==nsubj)&(df['COND']==condition),comp+'_LAT']=latency
    return latency

def find_latency_neg(data, comp, condition, nsubj, df, chan):
    imax=data[data['time'].isin(df[df.ID==nsubj][comp+'_window'].iloc[0])][chan].min()
    latency=data[data[chan]==imax]['time'].iloc[0]
    df.loc[(df['ID']==nsubj)&(df['COND']==condition),comp+'_LAT']=latency
    return latency

'''
makes heart_beat events ready to use (3 intervals)

events - initial event array

returns corrected events array
31:start_1 32:start_2 33:start_3
101:end_1 102:end_2 103:end_3
'''
def hb_events(events):

    a=pd.DataFrame({'time':[i[0] for i in events],
                    '0':[0 for i in list(range(len(events)))],
              'id':[i[2] for i in events]})

    list_3=[31,32,33]
    list_1000=[100,100,101,102,103]

    for i in range(list(a.id).count(3)):
        index_3=list(a.id).index(3)
        a.id[index_3]=list_3[i]

    for i in range(list(a.id).count(1002)):
        index_1000=list(a.id).index(1002)
        a.id[index_1000]=list_1000[i]

    return(a.to_numpy())

'''
makes Photo instancer from raw

p - list of str IDs
hb - whether input raw is heart_beat or not (default=True)
if True, raw is cut to 3 intervals, 
if False, raw is cut from the 1st to the last events

returns list of Photo instances
if hb=True, each element has 3 elements (1 per interval)
if hb=False, each element has 1 element

and list of sampling frequencies 
these lists are input for calc_hr
'''
def calc_photo (p, hb=True):

  photos=[]
  sr=[]
  for i in p:
    raw=mne.io.read_raw(path+i+'_heart_beat.vhdr', preload=True, verbose=50)
    sfreq=raw.info['sfreq']
    sr.append(sfreq)
    events=mne.events_from_annotations(raw, verbose=False)

    if hb==False:
      tmin=events[0][0][0]/sfreq
      tmax=events[0][-1][0]/sfreq
      raw.crop(tmin, tmax)
      photoS=raw['Photo']

    else:
      hbs=hb_events(events[0])
      s_1=[i[2] for i in hbs].index(31) #start 1
      s_2=[i[2] for i in hbs].index(32) #start 2
      s_3=[i[2] for i in hbs].index(33) #start 3
      e_1=[i[2] for i in hbs].index(101) #end_1
      e_2=[i[2] for i in hbs].index(102) #end_2
      e_3=[i[2] for i in hbs].index(103) #end_3

      tmin_1=events[0][s_1][0]/sfreq
      tmax_1=events[0][e_1][0]/sfreq
      tmin_2=events[0][s_2][0]/sfreq
      tmax_2=events[0][e_2][0]/sfreq
      tmin_3=events[0][s_3][0]/sfreq
      tmax_3=events[0][e_3][0]/sfreq

      raw_1=raw.copy()
      raw_2=raw.copy()
      raw_3=raw.copy()

      raw_1.crop(tmin=tmin_1, tmax=tmax_1)
      raw_2.crop(tmin=tmin_2, tmax=tmax_2)
      raw_3.crop(tmin=tmin_3, tmax=tmax_3)

      photo_1=raw_1['Photo']
      photo_2=raw_2['Photo']
      photo_3=raw_3['Photo']

      photoS=[photo_1, photo_2, photo_3]

    photos.append(photoS)

  return(photos, sr)

'''
calculates HR for Photo instance
makes plots returns with HR for each instance

p - str list of IDs
photos - list of Photo instances (raw['Photo']) from calc_photo
sr - list of sampling frequencies from calc_photo
plot - whether plot for every ID is needed (default False)

returns df with HR for each Photo instance
'''
def calc_hr(p, photos, sr, plot=False):

  amps=[i[0][0] for i in photos]
  times=[i[1] for i in photos]
  HRs=[]
  for i in range(len(p)):

    peaks, _ = find_peaks(amps[i])

    alpha = peak_prominences(amps[i], peaks, wlen=None)
    mean=np.mean(alpha[0])
    sd=np.std(alpha[0])

    peaks, _ = find_peaks(amps[i], prominence=(mean+sd, ))

    beats=len(peaks)
    time=len(times[i])/sr[i]
    hr=beats/time*60
    HRs.append(hr)

    if plot==True:

      plt.figure(figsize=(25,2))
      plt.plot(times[i], amps[i])
      plt.plot((peaks/sr[i]), amps[i][peaks], "x")
      plt.title(p[i]+' HR='+str(hr))
      plt.show()

  hr_df=pd.DataFrame({'ID':p, 'HR':HRs})
  return(hr_df)

'''
calculates HR for Photo instance
makes plots returns with HR for each instance

p - str list of IDs
photos - list of Photo instances (raw['Photo']) from calc_photo
sr - list of sampling frequencies from calc_photo
plot - whether plot for every ID is needed (default False)

returns df with HR for each Photo instance
'''
def event_to_power (epochs, event, baseline):
    epochs=epochs.resample(200)
    freqs = np.arange(0.2, 36, 0.2)
    n_cycles = freqs
    if event in list(list_of_epo[i].event_id.keys()):
        power = tfr_multitaper(epochs[event].average(), freqs=freqs,
                           n_cycles=n_cycles, use_fft=True, return_itc=False, decim=3, n_jobs=1)
        power.apply_baseline(baseline, mode='mean')
        return(power)
    else: print('No '+event)

#(-2, -1.5) for pre-choice
#(-0.3, 0.0)

def epoch_to_power (epoch, baseline, tmin, tmax):
    epoch=epoch.resample(200)
    freqs = np.arange(0.2, 36, 0.5)
    n_cycles = freqs
    power = tfr_multitaper(epoch.average(), freqs=freqs,
                           n_cycles=n_cycles, use_fft=True, return_itc=False, decim=3, n_jobs=1)
    power.apply_baseline(baseline, mode='mean')
    power.crop(tmin, tmax)
    return(power)

def ga_to_power (ga, baseline):
    freqs = np.arange(0.2, 36, 0.2)
    n_cycles = freqs
    power = tfr_multitaper(ga, freqs=freqs,
                           n_cycles=n_cycles, use_fft=True, return_itc=False, decim=3, n_jobs=1)
    power.apply_baseline(baseline, mode='mean')
    return(power)

'''
plots TFR time-course

tfr_1 - tfr instance for condition 1
tft_2 - tfr instance for condition 2
picks - list of str channels to plot, ex. ['FC1','FC2']
bands - str band, ex. 'theta'
tmin - tmin for plotting in secs
tmax - tmax for plotting in secs
ylim - ylim tuple (default=(-5,5))
cond_1 - str title for condition 1 (default='cond_1')
cond_2 - str title for condition 2 (default='cond_2')

returns tfr time-course plot
'''
def plot_time_course (tfr_1, tfr_2, picks, bands, tmin, tmax, ylim=(-5, 5), cond_1='cond_1', cond_2='cond_2'):

    df_1=tfr_1.to_data_frame(time_format=None, long_format=True, picks=picks)
    df_1['condition']=cond1

    df_2=tfr_2.to_data_frame(time_format=None, long_format=True, picks=picks)
    df_2['condition']=cond2

    df=pd.concat([df_1, df_2], ignore_index=True)
    df['value']=df['value']*10**10

    df=df[(df['time'] >= tmin) & (df['time'] <= tmax)]

    freq_bounds = {'_': 0,
               'delta': 3,
               'theta': 7,
               'alpha': 13,
               'beta': 35,
               'gamma': 140}

    df['band'] = pd.cut(df['freq'], list(freq_bounds.values()), labels=list(freq_bounds)[1:])

    df=df[df.band.isin(bands)]
    df['band'] = df['band'].cat.remove_unused_categories()

    df['channel'].cat.reorder_categories(picks, ordered=True, inplace=True)

    sns.set(font_scale=1.5, style='white')
    g=sns.FacetGrid(df, row='band', col='channel', aspect=2, height=6)
    g.map(sns.lineplot, 'time', 'value', 'condition', n_boot=1000, palette=['b', 'r'])
    plt.margins(x=0)
    #g.set_yticklabels(size = 15)
    axline_kw = dict(color='black', linestyle='dashed', linewidth=1, alpha=0.5)
    g.map(plt.axhline, y=0, **axline_kw)
    g.map(plt.axvline, x=0, **axline_kw)
    g.set(ylim=ylim)
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('ERSP (dB)', fontsize=18)
    g.set_titles(col_template='{col_name}'#+' (n='+str(len(p))+')'
    , row_template='{row_name}', size=18)
    #labels=[cond1, cond2]
    plt.legend(fontsize=18, ncol=1, loc='upper left')

    g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
    return(g)

'''
plots TFR

tfr_1 - tfr instance for condition 1
tft_2 - tfr instance for condition 2
picks - list of str channels to plot, ex. ['FC1','FC2']
bands - str band, ex. 'theta'
tmin - tmin for plotting in secs
tmax - tmax for plotting in secs
vmin - min colormap to plot (default=-1e-9)
vmax - max colormap to plot (default=1e-9)
cond_1 - str title for condition 1 (default='cond_1')
cond_2 - str title for condition 2 (default='cond_2')
fmin - min frequency to plot (default=1.0)
fmax - max frequency to plot (default=35.0)

returns tfr plot
'''
def plot_tfr (tfr1, tfr2, picks, bands, tmin=-0.3, tmax=1.0, vmin=-1e-9, vmax=1e-9, cond_1='cond_1', cond_2='cond_2', fmin=1.0, fmax=35.0):

  fig, axs = plt.subplots(1, 2, figsize=(18, 8))
  fig.tight_layout(pad=5)

  if bands=='delta':
    fmin=1.0
    fmax=4.0
  elif bands=='theta':
    fmin=4.0
    fmax=7.0
  elif bands=='alpha':
    fmin=7.0
    fmax=13.0
  elif bands=='beta':
    fmin=13.0
    fmax=35.0

  tfr1.plot(picks=picks, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax, baseline=None, show=False, axes=axs[0])
  axs[0].set_title(cond_1)

  tfr2.plot(picks=picks, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax, baseline=None, show=False, axes=axs[1])
  axs[1].set_title(cond_2)

  fig.suptitle(bands+' | '+picks+' (n='+str(len(p))+')', fontsize=40)

  def significant_sensors (tfr1, tfr2, bands, cond1, cond2, tmin=0, tmax=1, threshold=0.05, ms=200):

  significant_chs=[]
  chs=tfr1.ch_names
  times=tfr1.times

  if bands=='delta':
    fmin=1.0
    fmax=4.0
  elif bands=='theta':
    fmin=4.0
    fmax=7.0
  elif bands=='alpha':
    fmin=7.0
    fmax=13.0
  elif bands=='beta':
    fmin=13.0
    fmax=35.0

  tfr3=tfr1.copy()
  tfr3.crop(tmin, tmax, fmin, fmax)
  tfr4=tfr2.copy()
  tfr4.crop(tmin, tmax, fmin, fmax)

  for ch in chs:
    x1=tfr3.copy()
    x1=x1.pick(ch).data[0,:,:] #make 1D array
    x2=tfr4.copy()
    x2=x2.pick(ch).data[0,:,:]
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([x1, x2], seed=97,
                                                                     out_type='mask', n_permutations=1000, tail=0)
    sign_ch=None
    for i,c in enumerate(clusters):
      c=c[0]
      if ((len(list(range(c.start, c.stop)))>=ms/15) and (any([p<=threshold for p in cluster_p_values]))):#3 time samples=approx. 50ms
        sign_ch=ch
    significant_chs.append(sign_ch)

  significant_chs=list(filter(None, significant_chs))

  tfr3.pick_channels(significant_chs)
  tfr4.pick_channels(significant_chs)

  fig, axs = plt.subplots(1, 2, figsize=(12, 10)) #plotting
  fig.tight_layout(pad=3)

  tfr3.plot_topomap(sensors='bo',  show=False, show_names=True, axes=axs[0]);
  axs[0].set_title(cond1)

  tfr4.plot_topomap(sensors='bo',  show=False, show_names=True, axes=axs[1]);
  axs[1].set_title(cond2)

  fig.suptitle(bands+' | '+str(len(significant_chs))+' significant channels', fontsize=30, y=.85)
  return(fig, significant_chs)

#LATEST VERSION
def plot_sign_channel (tfr1, tfr2, ch, cond1, cond2, bands, tmin=-0.3, tmax=1, threshold=0.05):

  if bands=='delta':
    fmin=1.0
    fmax=4.0
  elif bands=='theta':
    fmin=4.0
    fmax=7.0
  elif bands=='alpha':
    fmin=7.0
    fmax=13.0
  elif bands=='beta':
    fmin=13.0
    fmax=35.0

  x1=tfr1.copy()
  x1.crop(tmin, tmax, fmin, fmax)
  times=list(x1.times)
  x2=tfr2.copy()
  x2.crop(tmin, tmax, fmin, fmax)

  x1=x1.pick(ch).data[0,:,:] #make 1D array
  x2=x2.pick(ch).data[0,:,:]

  T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([x1, x2], seed=97,
                                                                     out_type='mask', n_permutations=1000, tail=0)

  fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)

  #ERSP
  ax0.set_title('ERSP | '+bands+' '+ch)
  ax0.axvline(x=0, linestyle="--", color="black")
  ax0.axhline(y=0, linestyle="--", color="black")
  ax0.plot(times, np.mean(x1, axis=0), label=cond1, color='red')
  ax0.plot(times, np.mean(x2, axis=0), label=cond1, color='blue')
  ax0.legend(loc=2)
  ax0.set_ylabel("dB")
  ax0.margins(x=0)

  #t-values
  ax1.axvline(x=0, linestyle="--", color="black")
  hf = ax1.plot(times, T_obs, 'g')
  ax1.set_ylabel("T-values")
  h = None
  if len(clusters)>0:
    for i, c in enumerate(clusters):
      c = c[0]
      if cluster_p_values[i] <= threshold:
        h = ax1.axvspan(times[c.start], times[c.stop - 1], color='red', alpha=0.5)
        plt.legend((h, ), ('p<0.05', ), loc=2)
      #else:
          #ht = ax1.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)
          #plt.legend((ht, h), ('non-significant clusters', 'p<0.05'), loc=2)
          #hf = ax2.plot(times, T_obs, 'g')
        #plt.xlabel("time (s)")
        #plt.ylabel("T-values")
        #if ht is not None and h is not None:
          #plt.legend((ht, h), ('non-significant clusters', 'cluster p-value < 0.05'), loc=2)
        #elif ht is not None:
            #plt.legend((ht, ), ('non-significant clusters',), loc=2)
        #elif h is not None:
              #plt.legend((h, ), ('cluster p-value < 0.05', ), loc=2)
  plt.xlabel("time (s)")
  plt.ylabel("T-values")
  ax1.margins(x=0)