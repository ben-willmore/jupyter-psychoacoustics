'''
Sound localization practical using jupyter / google colab
'''

# pylint: disable=C0103, R0912, R0914

import asyncio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import ipywidgets as widgets
from IPython.display import display, clear_output
from psychoacoustics.jupyterpsych import JupyterPsych, AudioPlayer
from psychoacoustics.sound import puretone, make_diotic, cosramp_onoff

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
            clear_output()
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
            clear_output()
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
    if jupyterpsych is not None and jupyterpsych.is_colab():
        figsize = (12, 6)
        plt.rc('font', size=15)
    else:
        figsize = (9, 4)

    _, ax = plt.subplots(figsize=figsize)

    freqs = [50, 125, 250, 500, 1000, 2000, 3000, 4000, 5000, 8000]
    levels = [15, 10, 15, 25, 20, 30, 45, 67, 70, 70]
    plt.plot(freqs, levels)

    freqs = [50, 125, 250, 500, 1000, 2000, 3000, 4000, 5000, 8000]
    levels = [35, 30, 35, 30, 24, 30, 35, 0, 30, 30]
    plt.plot(freqs, levels)

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

# see https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Events.html

class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback
        self._task = asyncio.ensure_future(self._job())

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def cancel(self):
        self._task.cancel()

def debounce(wait):
    """ Decorator that will postpone a function's
        execution until after `wait` seconds
        have elapsed since the last time it was invoked. """
    def decorator(fn):
        timer = None
        def debounced(*args, **kwargs):
            nonlocal timer
            def call_it():
                fn(*args, **kwargs)
            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)
        return debounced
    return decorator

class SquareWave():
    '''
    Plot square wave composed of sinusoids
    '''

    def __init__(self, jupyterpsych=None):
        self.jupyterpsych = jupyterpsych
        self.widgets = {}
        self.t = np.linspace(0, 1, 400)
        self.widgets['graphoutput'] = widgets.Output()

        # freq slider
        self.widgets['freqlabel'] = widgets.Label('Frequency of highest component:')
        self.widgets['slider'] = widgets.IntSlider(min=1, max=20, step=1, value=10, readout=False)
        self.widgets['freqvalue'] = widgets.Label('23 Hz')
        @debounce(1)
        def f(_):
            self.widgets['freqvalue'].value = '%d Hz' % (((self.widgets['slider'].value*2)-1)*500)
            self.plot()
        self.widgets['slider'].observe(f)
        self.widgets['sliderbox'] = widgets.HBox(
            (self.widgets['freqlabel'], self.widgets['slider'], self.widgets['freqvalue']))
        self.widgets['box'] = widgets.VBox((self.widgets['graphoutput'], self.widgets['sliderbox']))
        self.plot()
        display(self.widgets['box'])

    def plot(self):
        '''
        Plot square wave composed of n sinusoidal components
        '''
        # with self.widgets['graphoutput']:
        #     clear_output()
        #     _, ax = plt.subplots()
        #     ax.set_ylim((0, 10))
        #     plt.plot(self.freqs, self.thresh)
        #     plt.show()


        if self.jupyterpsych is not None and self.jupyterpsych.is_colab():
            figsize = (12, 6)
            plt.rc('font', size=15)
        else:
            figsize = (9, 4)
        with self.widgets['graphoutput']:
            clear_output(wait=True)
            _, ax = plt.subplots(figsize=figsize)
            y = np.zeros(400)
            for a in range(1, self.widgets['slider'].value+1):
                c = a*2-1
                tone = 1/c*puretone(1, 400, c/200)
                plt.plot(self.t, tone, 'r')
                y = y + tone
            plt.plot(self.t, y, 'k', linewidth=4)
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
            plt.show()

    def on_button_clicked(self, _):
        '''
        Respond to button click
        '''
        self.thresh = self.thresh + 1
        self.plot()
