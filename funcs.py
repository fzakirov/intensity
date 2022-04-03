pip install mne

pip install PyPDF2

import mne
import os
import numpy as np
from mne.preprocessing import ICA
import pandas as pd

def drop_epochs_by_sigma(epochs_1, sd):
    maximas=[]
    for i in range(epochs_1.get_data().shape[0]):
        maximas.append(epochs_1.get_data()[i].max(axis=1))
    maximas=np.array(maximas)
    #стандартное отколенние максимумов по каждому каналу
    maximas_std=maximas.std(axis=0)
    #среднее максимумов по каждому каналу
    maximas_mean=maximas.mean(axis=0)
    th_max=maximas_mean+sd*maximas_std
    th_min=maximas_mean-sd*maximas_std
    rej=[]
    for i in range(epochs_1.get_data().shape[0]):
        for j in range(epochs_1.get_data().shape[1]):
            if maximas[i,j]>th_max[j] or maximas[i,j]<th_min[j]:
                rej.append(i)
    print()            
    reject_list=list(set(rej))
    epochs_1.drop(reject_list)

def read_clean_raws(p, path):
    raws=[]
    for i in p:
        raw=mne.io.read_raw(path+i+'-raw.fif', preload=True)
        raws.append(raw)
    return raws

def epo_from_raw(raws, baseline):

  list_of_epo_50=[]
  list_of_epo_60=[]
  list_of_epo_70=[]
  list_of_epo_80=[]

  for i,c in enumerate(raws):

    events=mne.find_events(c)
    
    epochs=mne.Epochs(c, events, event_id={'80dB':1, '70dB':2,'60dB':3, '50dB':4},
                  #reject=dict(eeg=150e-6),
                   baseline=None, tmin=-0.5, tmax=0.8, preload=True)
    
    epochs=epochs.resample(250).crop(-0.2, 0.7).apply_baseline(baseline)

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
  

    # evo_50=epochs['50dB'].average()#.apply_baseline((-0.2, 0))
    # evo_60=epochs['60dB'].average()#.apply_baseline((-0.2, 0))
    # evo_70=epochs['70dB'].average()#.apply_baseline((-0.2, 0))
    # evo_80=epochs['80dB'].average()#.apply_baseline((-0.2, 0))

    # list_of_evo_50.append(evo_50)
    # list_of_evo_60.append(evo_60)
    # list_of_evo_70.append(evo_70)
    # list_of_evo_80.append(evo_80)

def evo_from_epo(p, epo_50, epo_60, epo_70, epo_80):
  list_of_evo_50=[]
  list_of_evo_60=[]
  list_of_evo_70=[]
  list_of_evo_80=[]
  
  for i,c in enumerate(p):
    
    evo_50=epo_50[i].average()#.apply_baseline((-0.2, 0))
    evo_60=epo_60[i].average()#.apply_baseline((-0.2, 0))
    evo_70=epo_70[i].average()#.apply_baseline((-0.2, 0))
    evo_80=epo_80[i].average()#.apply_baseline((-0.2, 0))

    list_of_evo_50.append(evo_50)
    list_of_evo_60.append(evo_60)
    list_of_evo_70.append(evo_70)
    list_of_evo_80.append(evo_80)
  
  return(list_of_evo_50,
         list_of_evo_60,
         list_of_evo_70,
         list_of_evo_80)

def create_zakl_evokeds(low, high): 
  ages_td=pd.read_csv('Felix/intensity/children/TD/td_df.csv', dtype=str) #df с возрастом и айдишниками
  ages_td['AGE']=ages_td['AGE'].astype(float)
  p=list(ages_td[(ages_td['AGE']>=low) & (ages_td['AGE']<=high)]['ID'])

  list_of_evo_50=[]
  list_of_evo_60=[]
  list_of_evo_70=[]
  list_of_evo_80=[]

  for i in p:
    epo=mne.read_epochs('Felix/intensity/children/TD/epo_3sd_ica/'+i+'-epo.fif')
    epo=epo.apply_baseline((-0.2, 0.0))..crop(-0.2,0.7)

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

def plot_evoked(ch_name, list_of_evoked_1, evoked_sub_1, 
                condition_title, sub, std_num=2):

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

  plt.plot(ga_evoked_1.times,ga_evoked_array*10**6,'b',label='TD (n='+str(len(p))+') '+condition_title)
  plt.plot(ga_evoked_1.times,evoked_sub_1_array*10**6,'r',label=sub+' '+condition_title)

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

def merge_pdf(p, path, title='result'):

  pdfs=map((lambda a: path+a+'.pdf'), p)
  pdfs=list(pdfs)
#pdfs.insert(0, 'intensity/rtt_plots/2-20/ga.pdf')
#pdfs

  from PyPDF2 import PdfFileMerger

  merger = PdfFileMerger()

  for pdf in pdfs:
      merger.append(pdf)

  merger.write(path+title+'pdf')
  merger.close()

def plot_erps(p, path, evo50, evo60, evo70, evo80):

  ga_50=mne.grand_average(evo50)
  ga_60=mne.grand_average(evo60)
  ga_70=mne.grand_average(evo70)
  ga_80=mne.grand_average(evo80)

  for i,c in enumerate(p):

    a=mne.viz.plot_compare_evokeds({'50dB':evo50[i],
                               '60dB':evo60[i],
                               '70dB':evo70[i],
                               '80dB':evo80[i]}, axes='topo')

    a[0].suptitle(c)
    a[0].savefig(path+c+'.pdf')

  a=mne.viz.plot_compare_evokeds({'50dB':ga_50,
                               '60dB':ga_60,
                               '70dB':ga_70,
                               '80dB':ga_80}, axes='topo')

  a[0].suptitle('GA (n='+str(len(p))+')')
  a[0].savefig(path+'GA.pdf')

def plot_ga_df(ch_name, df_con, list_of_df, cond, color='b', std_num=2, std=False, limits=[-0.5, 0.8, -15, 15]):

    for_std=np.array([ev[ch_name] for ev in (list_of_df)])
    std_array=np.std(for_std,0) 
    #gstd_evoked_array=scipy.stats.sem(for_std)

    plt.plot(df_con.time/1000, df_con[ch_name], color, label='TD '+cond)
    

    if std==True:
        plt.fill_between(df_con.time/1000, df_con.FCz - std_num*std_array, 
                   df_con.FCz + std_num*std_array, color, alpha=0.3)


    plt.axvline(x=0, linestyle="--", color="black")
    plt.axhline(y=0, linestyle="--", color="black")
    plt.title(ch_name)
    plt.ylabel('Amplitude, μV')
    plt.xlabel('Time, s')
    plt.axis(limits)
    plt.legend()
    return(plt)

def plot_evoked_single(ch_name, evoked_1, evoked_2, 
                condition_title, sub1, sub2, std_num=2):
  
  times=evoked_1.times

  #ga_evoked_1.resample(200)
  ch_num=evoked_1.ch_names.index(ch_name)
  evoked_1=evoked_1.apply_baseline((-0.2, 0.0))
  evoked_array_1=evoked_1.data[ch_num]
  
  #evoked_sub_1.resample(200)
  ch_num=evoked_2.ch_names.index(ch_name)
  evoked_2=evoked_2.apply_baseline((-0.2, 0.0))
  evoked_array_2=evoked_2.data[ch_num]

  plt.plot(times,evoked_array_1*10**6,'b',label=sub1+condition_title)
  plt.plot(times,evoked_array_2*10**6,'r',label=sub2+condition_title)

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

def to_dataframe_new(epochs, cond, chan, delay=False, invert=False):
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

def create_evo_csv (sub, path_epo, path_csv, baseline=(-0.2,0)):
  list_of_evo_80=[]
  list_of_evo_70=[]
  list_of_evo_60=[]
  list_of_evo_50=[]
  for i,c in enumerate(sub):
    epochs=mne.read_epochs(path_epo+c+'-epo.fif')
    epochs=epochs.apply_baseline(baseline)

    evo_80=epochs['80dB'].average()
    df_80=evo_80.to_data_frame()
    df_80.to_csv(path_csv+c+'_evo_80.csv',sep=';')

    evo_70=epochs['70dB'].average()
    df_70=evo_70.to_data_frame()
    df_70.to_csv(path_csv+c+'_evo_70.csv',sep=';')

    evo_60=epochs['60dB'].average()
    df_60=evo_60.to_data_frame()
    df_60.to_csv(path_csv+c+'_evo_60.csv',sep=';')

    evo_50=epochs['50dB'].average()
    df_50=evo_50.to_data_frame()
    df_50.to_csv(path_csv+c+'_evo_50.csv',sep=';')

def create_evo (sub, path_epo, baseline=(-0.2,0)):
  list_of_evo_80=[]
  list_of_evo_70=[]
  list_of_evo_60=[]
  list_of_evo_50=[]
  for i,c in enumerate(sub):
    epochs=mne.read_epochs(path_epo+c+'-epo.fif').resample(250).crop(-0.4, 0.7)
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