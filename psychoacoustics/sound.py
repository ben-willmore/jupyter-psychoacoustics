'''
Basic sound functions
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import spectrogram as ss_spec

def tdt50k():
    '''
    TDT "50k" sample rate
    '''
    return 48828.125

def amp2level(amp):
    '''
    Convert an amplitude multiplier to level in dB.
    E.g. a multiplier of 10 is +20dB
    '''
    return 20*np.log10(amp)

def level2amp(dB):
    '''
    Convert a dB difference to amplitude.
    E.g. a difference of +10dB is a multiplier of 3.16
    '''
    return 10**(dB/20)

def rms2dBSPL(rms_Pa):
    '''
    Convert a sound RMS in Pascals to dB SPL.
    E.g. 1 Pa RMS = 94dB
    '''
    return amp2level(rms_Pa)+94

def dBSPL2rms(dBSPL):
    '''
    Convert dBSPL to RMS in Pascals
    E.g. 94dB = 1 Pa RMS
    '''
    return level2amp(dBSPL-94)

def puretone(fs, n_samples, freq=440, level_dB='max', phase=0):
    '''
    Generate a pure tone
    '''
    t = np.arange(n_samples) * 1/fs
    if level_dB == 'max':
        mult = 1
    else:
        mult = np.sqrt(2) * dBSPL2rms(level_dB)
    return np.sin(2*np.pi*freq*t + phase) * mult

def freq_sweep(fs, n_samples, f_min=200, f_max=1000, method='log', level_dB='max', phase=0):
    '''
    Generate a frequency sweep
    '''
    t = np.arange(n_samples)/fs
    if level_dB == 'max':
        mult = 1
    else:
        mult = np.sqrt(2) * dBSPL2rms(level_dB)
    if method.lower().startswith('li'):
        c = (f_max-f_min)/(n_samples/fs)
        return np.sin(2*np.pi*(f_min*t + c/2*(t**2)) + phase) * mult
    # else: # method == 'log'
    k = (f_max/f_min)**(fs/n_samples)
    return np.sin(2*np.pi*f_min*(k**t-1)/np.log(k) + phase) * mult

def whitenoise(n_samples, method='uniform', level_dB='max'):
    '''
    Generate white noise -- NB method='normal', level_dB='max' will
    scale the range depending on the random numbers produced, so won't
    give a consistent sound level
    '''
    if method.lower()[0] == 'u': # 'uniform'
        if level_dB == 'max':
            mult = 1
        else:
            mult = np.sqrt(3) * dBSPL2rms(level_dB)
        return ((2*np.random.random((n_samples)))-1) * mult
    # elif method.lower()[0] == 'n': # 'normal'
    rand = np.random.randn((n_samples))
    if level_dB == 'max':
        return rand / np.max(np.abs(rand))
    else:
        return np.random.randn((n_samples)) * dBSPL2rms(level_dB)

def cosramp_on(n_samples, ramp_samples=None):
    '''
    Ramp on - total length n_samples, ramp length ramp_samples
    '''
    if ramp_samples is None:
        ramp_samples = n_samples
    t = np.minimum(np.arange(n_samples), ramp_samples)
    return np.sin(np.pi/2/ramp_samples*t)

def cosramp_off(n_samples, ramp_samples=None):
    '''
    Ramp off - total length n_samples, ramp length ramp_samples
    '''
    if ramp_samples is None:
        ramp_samples = n_samples
    return cosramp_on(n_samples, ramp_samples)[::-1]

def cosramp_onoff(n_samples, ramp_samples):
    '''
    Ramp on and off - total length n_samples, ramp lengths ramp_samples
    '''
    r = cosramp_on(n_samples, ramp_samples)
    return r * r[::-1]

def make_diotic(snd, n_channels=2):
    '''
    Duplicate sound to 2 or more channels
    '''
    return np.stack(snd.ravel(), axis=0)

def apply_ild(fs, snd, ild_dB=10):
    left = snd
    right = snd * level2amp(ild_dB)
    return np.stack((left, right), axis=0)

def apply_itd(fs, snd, itd_us=100):
    shift_samples = np.int(np.abs(itd_us*1000/fs))
    leading = np.concatenate((snd, np.zeros(shift_samples)))
    lagging = np.concatenate((np.zeros(shift_samples), snd))
    if itd_us < 0:
        return np.stack((leading, lagging), axis=0)
    # else: itd_us >= 0
    return np.stack((lagging, leading), axis=0)

def ild_stimulus(fs, len_s, f0, ild_dB, level_dB='max'):
    n_samples = np.int(len_s*fs)
    snd_mono = puretone(fs, n_samples, f0, level_dB=level_dB)
    ramplen_ms = 5
    snd_mono = snd_mono * cosramp_onoff(n_samples, ramp_samples=np.round(ramplen_ms/1000*fs))
    return apply_ild(fs, snd_mono, ild_dB=ild_dB)

def itd_stimulus(fs, len_s, f0, itd_us, level_dB='max'):
    n_samples = np.int(len_s*fs)
    snd_mono = puretone(fs, n_samples, f0, level_dB=level_dB)
    ramplen_ms = 5
    snd_mono = snd_mono * cosramp_onoff(n_samples, ramp_samples=np.round(ramplen_ms/1000*fs))
    return apply_itd(fs, snd_mono, itd_us=itd_us)

def binaural_beats(f_s, n_samples, f_l=520, f_r=530, level_dB='max'):
    '''
    Binaural beat stimulus
    '''
    return np.stack((puretone(f_s, n_samples, f_l, level_dB='max'),
                     puretone(f_s, n_samples, f_r)), axis=0, level_dB='max')

def spectrogram(*args, **kwargs):
    '''
    This is just to help find the scipy spectrogram function
    '''
    return ss_spec(*args, **kwargs)

def show_spectrogram(*args, **kwargs):
    '''
    Show spectrogram using pyplot
    '''
    f, t, s = ss_spec(*args, **kwargs)
    _, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(s, origin='lower', extent=[np.min(t), np.max(t), np.min(f), np.max(f)])
    ax.set_aspect((np.max(t)-np.min(t))/(np.max(f)-np.min(f))/2)
