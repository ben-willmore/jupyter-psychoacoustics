'''
Sound localization practical using jupyter / google colab
'''

# pylint: disable=C0103, R0912, R0914

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.signal import stft, istft, spectrogram
from scipy.io.wavfile import read as wavread
import ipywidgets as widgets
from IPython.display import display, clear_output
from psychoacoustics.jupyterpsych import JupyterPsych, AudioPlayer
from psychoacoustics.sound import puretone, make_diotic, cosramp_onoff, level2amp

class AudiogramExpt():
    '''
    Audiogram Expt
    '''
    def __init__(self, jupyterpsych=None):
        if jupyterpsych is None:
            jupyterpsych = JupyterPsych()
        self.jupyterpsych = jupyterpsych

        self.f_s = self.jupyterpsych.f_s

        self.freqs = [10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000,
                      10000, 11000, 12000, 13000, 14000, 15000,
                      16000, 17000, 18000, 19000, 20000]
        self.thresh = np.ones((len(self.freqs)))*np.nan
        self.widgets = self.setup_ui()
        self.plot()

    def setup_ui(self):
        '''
        Set up widgets and display them
        '''
        w = {}
        w['graphoutput'] = widgets.Output()

        # freq slider
        w['freqslider'] = widgets.IntSlider(
            min=0, max=len(self.freqs)-1, step=1, value=6, readout=False)
        w['freqvalue'] = widgets.Label('23 Hz')
        def f(_):
            w['freqvalue'].value = '%d Hz' % self.freqs[w['freqslider'].value]
        w['freqvalue'].value = '%d Hz' % self.freqs[w['freqslider'].value]
        w['freqslider'].observe(f)

        # level slider
        w['levelslider'] = widgets.IntSlider(
            min=-20, max=60, step=5, value=40, readout=False)
        w['levelvalue'] = widgets.Label('34 dB')
        def l(_):
            w['levelvalue'].value = '%d dB SPL' % w['levelslider'].value
        w['levelvalue'].value = '%d dB SPL' % w['levelslider'].value
        w['levelslider'].observe(l)

        # arrange widgets in vertical boxes for alignment
        labelbox = widgets.VBox((widgets.Label('Frequency'), widgets.Label('Level')))
        sliderbox = widgets.VBox((w['freqslider'], w['levelslider']))
        valuebox = widgets.VBox((w['freqvalue'], w['levelvalue']))

        # arrange boxes into a horizontal box -- this needs to be display()ed
        w['sliderbox'] = widgets.HBox((labelbox, sliderbox, valuebox))

        # buttons
        w['soundButton'] = widgets.Button(description="Play sound")
        w['soundButton'].on_click(self.on_soundButton_clicked)

        w['recordButton'] = widgets.Button(description="Record point")
        w['recordButton'].on_click(self.on_recordButton_clicked)

        w['deleteButton'] = widgets.Button(description="Delete point")
        w['deleteButton'].on_click(self.on_deleteButton_clicked)

        w['clearButton'] = widgets.Button(description="Clear graph")
        w['clearButton'].on_click(self.on_clearButton_clicked)


        w['buttonbox'] = widgets.HBox(
            (w['soundButton'], w['recordButton'], w['deleteButton'], w['clearButton']))

        w['audiooutput'] = widgets.Output()

        w['audio'] = AudioPlayer(np.ones(100), rate=self.f_s,
                                 autoplay=True, hide_on_click=True)

        display(w['graphoutput'])
        display(w['sliderbox'])
        display(w['buttonbox'])
        display(w['audiooutput'])

        return w

    def plot(self):
        '''
        Plot graph
        '''
        with self.widgets['graphoutput']:
            clear_output(wait=True)
            if self.jupyterpsych.is_colab():
                figsize = (12, 6)
                plt.rc('font', size=15)
            else:
                figsize = (9, 4)
            _, ax = plt.subplots(figsize=figsize)
            if np.all(np.isnan(self.thresh)):
                plt.scatter(np.array(self.freqs), 1000*np.ones(len(self.freqs)))
            else:
                plt.scatter(np.array(self.freqs), self.thresh)
            ax.set_ylim((-30, 60))
            ax.set_xscale('log')
            ax.set_xlim((10, 20000))
            ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Level (dB)')
            ax.grid(True, which='both')
            plt.show()

    @classmethod
    def standard_thresholds(cls):
        freqs = [20, 70, 100, 200, 300, 400, 600, 1000, 1800, 2000, 3800, 9000,
                 10000, 12000, 15000, 20000, 25000]
        levels = [75, 33, 26, 13, 9, 7, 6, 5, 5, 4, -5, 17, 17, 20, 20, 75, 75]
        return freqs, levels

    def plot_standard_thresholds(self, use_fake_data=True):
        if use_fake_data:
            self.thresh = [75, 60, 40, 26, 13, 7, 5, 5, 0, 10, 17,
                           17, 18, 19, 20, 15, 65,
                           60, 60, 75, 75, 75]
            self.thresh = [s + 20 for s in self.thresh]

        freqs, levels = self.standard_thresholds()

        if self.jupyterpsych.is_colab():
            figsize = (12, 6)
            plt.rc('font', size=15)
        else:
            figsize = (9, 4)
        _, ax = plt.subplots(figsize=figsize)

        plt.plot(self.freqs, self.thresh)
        plt.plot(freqs, levels)
        ax.set_ylim((-30, 80))
        ax.set_xscale('log')
        ax.set_xlim((10, 20000))
        ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Level (dB)')
        ax.grid(True, which='both')
        plt.legend(['Your thresholds', 'Standard thresholds'])
        plt.show()

    def plot_clinical_audiogram(self, use_fake_data=True):
        '''
        Plot difference of measured and standard values, on confusing monotonically
        decreasing scale, as used for clinical audiograms
        '''
        if use_fake_data:
            self.thresh = [75, 60, 40, 26, 13, 7, 5, 5, 0, 10, 17,
                           17, 18, 19, 20, 15, 65,
                           60, 60, 75, 75, 75]
            self.thresh = [s + 20 for s in self.thresh]
        freqs, levels = self.standard_thresholds()
        levels_interp = np.interp(self.freqs, freqs, levels)
        if self.jupyterpsych.is_colab():
            figsize = (12, 6)
            plt.rc('font', size=15)
        else:
            figsize = (9, 4)
        _, ax = plt.subplots(figsize=figsize)
        plt.plot(self.freqs, self.thresh - levels_interp)
        plt.plot([10, 20000], [0, 0], 'k')
        ax.invert_yaxis()
        ax.set_ylim((60, -20))
        ax.set_xscale('log')
        ax.set_xlim((10, 20000))
        ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Level (dB)')
        ax.grid(True, which='both')
        # plt.legend(['Your thresholds', 'Standard thresholds'])
        plt.show()

    def on_recordButton_clicked(self, _):
        '''
        Update graph data and redisplay whole UI
        '''
        freq_idx = self.widgets['freqslider'].value
        level = self.widgets['levelslider'].value
        if np.isnan(self.thresh[freq_idx]) or level < self.thresh[freq_idx]:
            self.thresh[freq_idx] = level
        self.plot()

    def on_soundButton_clicked(self, _):
        '''
        Play sound
        '''
        freq = self.freqs[self.widgets['freqslider'].value]
        level = self.widgets['levelslider'].value
        snd = puretone(self.f_s, self.f_s, freq=freq, level_dB=level) * \
            cosramp_onoff(self.f_s, 5/1000*self.f_s)
        snd = make_diotic(snd)
        snd_calib = snd * self.jupyterpsych.calib_multiplier
        self.widgets['audio'].update_data(self.f_s, snd_calib, scale_to_max=False)
        with self.widgets['audiooutput']:
            clear_output(wait=True)
            display(self.widgets['audio'])

    def on_deleteButton_clicked(self, _):
        '''
        Delete point
        '''
        freq_idx = self.widgets['freqslider'].value
        self.thresh[freq_idx] = np.nan
        self.plot()

    def on_clearButton_clicked(self, _):
        '''
        Clear graph
        '''
        self.thresh = np.ones((len(self.freqs)))*np.nan
        self.plot()

def print_setup_message():
    '''
    Print setup message
    '''
    print('\n=== Setup complete ===\n')
    print('Now, move on to the next cell to set up your headphones\n')


def plot_disorders(jupyterpsych=None):
    '''
    Plot hearing disorder audiograms
    '''
    if jupyterpsych is None:
        jupyterpsych = JupyterPsych()
    if jupyterpsych.is_colab():
        figsize = (12, 6)
        plt.rc('font', size=15)
    else:
        figsize = (9, 4)

    _, ax = plt.subplots(figsize=figsize)

    # presbyacusis
    freqs = [50, 125, 250, 500, 1000, 2000, 3000, 4000, 5000, 8000]
    levels = [15, 10, 15, 25, 20, 30, 45, 67, 70, 70]
    plt.plot(freqs, levels)

    # conductive hearing loss
    freqs = [50, 125, 250, 500, 1000, 2000, 3000, 4000, 5000, 8000]
    levels = [35, 30, 35, 30, 24, 30, 35, 40, 30, 30]
    plt.plot(freqs, levels)

    # noise induced hearing loss
    freqs = [50, 125, 250, 500, 1000, 2000, 4000, 8000]
    levels = [7, 5, 5, 5, 10, 20, 50, 20]
    plt.plot(freqs, levels)

    ax.invert_yaxis()
    ax.set_ylim((80, -10))
    ax.set_xscale('log')
    ax.set_xlim((10, 20000))
    ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Level (dB)')
    ax.grid(True, which='both')
    plt.legend(['A', 'B', 'C'])
    plt.show()  # generate a presbyacusis audiogram

class SquareWave():
    '''
    Plot square wave composed of sinusoids
    '''

    def __init__(self, jupyterpsych=None):
        if jupyterpsych is None:
            jupyterpsych = JupyterPsych()

        self.jupyterpsych = jupyterpsych
        self.widgets = {}
        self.t = np.linspace(0, 1, 400)
        self.n = 10
        self.widgets['graphoutput'] = widgets.Output()

        # control number of components
        freqlabel = widgets.Label('Frequency of highest component:')
        down = widgets.Button(description='Decrease')
        down.on_click(lambda x: self.update_n(self.n-1))
        self.widgets['freqvalue'] = widgets.Label('%d Hz' % (((self.n*2)-1)*500))
        up = widgets.Button(description='Increase')
        up.on_click(lambda x: self.update_n(self.n+1))
        self.widgets['audiooutput'] = widgets.Output()
        self.widgets['audioplayer'] = AudioPlayer(np.ones(500), rate=self.jupyterpsych.f_s,
                                                  autoplay=False, hide_on_click=False)
        freqbox = widgets.HBox(
            (freqlabel, down, self.widgets['freqvalue'], up, self.widgets['audiooutput']))
        self.widgets['box'] = widgets.VBox((self.widgets['graphoutput'], freqbox))
        display(self.widgets['box'])
        self.plot()

    def update_n(self, n):
        '''
        Update display and sound to match current number of components
        '''
        self.n = max(min(n, 20), 1)
        self.widgets['freqvalue'].value = '%d Hz' % (((self.n*2)-1)*500)
        self.plot()

    def plot(self):
        '''
        Plot square wave composed of n sinusoidal components
        '''
        if self.jupyterpsych is not None and self.jupyterpsych.is_colab():
            figsize = (12, 6)
            plt.rc('font', size=15)
        else:
            figsize = (9, 4)
        with self.widgets['graphoutput']:
            clear_output(wait=True)
            plt.subplots(figsize=figsize)
            y = np.zeros(self.jupyterpsych.f_s*2)
            for a in range(1, self.n+1):
                c = a*2-1
                tone = 1/c*puretone(self.jupyterpsych.f_s*2, self.jupyterpsych.f_s*2, 500*c)
                plt.plot(self.t, tone[:400], 'b')
                y = y + tone
            plt.plot(self.t, y[:400], 'k', linewidth=4)
            plt.show()

        with self.widgets['audiooutput']:
            clear_output(wait=True)
            self.widgets['audioplayer'].update_data(self.jupyterpsych.f_s, y)
            display(self.widgets['audioplayer'])

class NaturalSound():
    '''
    Demonstration of natural sound without high frequency components
    '''

    def __init__(self, jupyterpsych=None):
        if jupyterpsych is None:
            jupyterpsych = JupyterPsych()
        self.jupyterpsych = jupyterpsych
        self.n = [4, 8, 16, 64, 128, 'Presbyacusis']
        self.f_s, self.full_sound = wavread(Path(Path(__file__).parent, 'wav', 'shipping.wav'))
        self.f, self.t, self.stft = stft(self.full_sound, self.f_s)

        # spectrogram
        self.widgets = {}
        self.widgets['graphoutput'] = widgets.Output()

        # buttons
        butlist = []
        def f(but):
            self.update(but.description)
        for n in self.n:
            butlist.append(widgets.Button(description=str(n)))
            butlist[-1].on_click(f)
        self.widgets['buttonbox'] = widgets.HBox(butlist)

        self.widgets['audiooutput'] = widgets.Output()
        self.widgets['audioplayer'] = AudioPlayer(np.ones(100), rate=self.f_s,
                                                  autoplay=False, hide_on_click=False)

        display(self.widgets['graphoutput'])
        display(self.widgets['buttonbox'])
        display(self.widgets['audiooutput'])
        self.update(128)

    def update(self, n):
        '''
        Update plot and sound
        '''
        try:
            # Get n as an integer if possible
            n = int(n)

            # reconstruct sound from STFT using only n channels (and DC)
            snd_stft = self.stft.copy()
            snd_stft[n+1:, :] = 0
            _, snd = istft(snd_stft, self.f_s)
        except ValueError:
            # presbyacusis
            freqs = [50, 125, 250, 500, 1000, 2000, 3000, 4000, 5000, 8000]
            levels = [15, 10, 15, 25, 20, 30, 45, 67, 70, 70]

            levels = [level2amp(-l) for l in levels]
            coeff = np.interp(self.f, freqs, levels)
            coeff[0] = 0 # set DC to zero
            snd_stft = self.stft.copy()
            _, snd = istft(snd_stft * coeff[:, np.newaxis], self.f_s)
        with self.widgets['audiooutput']:
            clear_output(wait=True)
            self.widgets['audioplayer'].update_data(self.f_s, snd)
            display(self.widgets['audioplayer'])
        with self.widgets['graphoutput']:
            clear_output(wait=True)
            self.show_shipping(snd)

    def show_shipping(self, snd):
        '''
        Show spectrogram
        '''
        f, t, spec = spectrogram(snd, self.f_s, nperseg=2000)
        log_spec = np.log(spec+0.0001)
        log_spec = np.maximum(log_spec, -10)

        if self.jupyterpsych.is_colab():
            figsize = (12, 6)
            plt.rc('font', size=15)
        else:
            figsize = (9, 4)

        _, _ = plt.subplots(figsize=figsize)
        plt.imshow(log_spec, origin='lower', aspect='auto', clim=(-10, 14),
                   cmap='hot', extent=[np.min(t), np.max(t), np.min(f), np.max(f)])
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.show()

class TestGraph():
    '''
    Demo of updating a graph inside widgets.Output()
    '''
    def __init__(self):
        self.freqs = np.logspace(np.log10(50), np.log10(20000))
        self.thresh = np.ones(self.freqs.shape)

        self.widgets = {}
        self.widgets['graphoutput'] = widgets.Output()
        self.widgets['button'] = widgets.Button(description="Update")
        self.widgets['button'].on_click(self.on_button_clicked)

        display(self.widgets['graphoutput'])
        display(self.widgets['button'])
        self.plot()

    def plot(self):
        '''
        Plot graph in place of previous one
        '''
        with self.widgets['graphoutput']:
            clear_output()
            _, ax = plt.subplots()
            ax.set_ylim((0, 10))
            plt.plot(self.freqs, self.thresh)
            # THE FOLLOWING IS ESSENTIAL:
            plt.show()

    def on_button_clicked(self, _):
        '''
        Respond to button click
        '''
        self.thresh = self.thresh + 1
        self.plot()
