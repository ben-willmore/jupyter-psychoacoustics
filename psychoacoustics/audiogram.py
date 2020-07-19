'''
Sound localization practical using jupyter / google colab
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import ipywidgets as widgets
from IPython.display import display, clear_output
from psychoacoustics.sound import ild_stimulus, itd_stimulus
from psychoacoustics.stats import logistic, probit_fit
# need headphone_check so it is included in 'from psychoacoustics.localization import *' in notebook
from psychoacoustics.jupyterpsych import is_colab, JupyterPsych, AudioPlayer, collate_responses

F_S = 44100

class TestGraph():
    def __init__(self):
        self.freqs = np.logspace(np.log10(50), np.log10(20000))
        self.thresh = np.ones(self.freqs.shape)

        self.widgets = {}
        self.widgets['graphoutput'] = widgets.Output()
        self.widgets['button'] = widgets.Button(description="Update")
        self.widgets['button'].on_click(self.on_button_clicked)
        self.display()

    def display(self):
        '''
        Clear and redraw the entire cell output, using a new Output to hold the new graph
        because you don't seem to be able to clear Outputs on colab.
        '''
        clear_output()
        self.widgets['graphoutput'] = widgets.Output()
        with self.widgets['graphoutput']:
            fig, ax = plt.subplots()
            ax.set_ylim((0, 10))
            line = plt.plot(self.freqs, self.thresh)
            plt.show()
        display(self.widgets['graphoutput'])
        display(self.widgets['button'])

    def on_button_clicked(self, _):
        self.thresh = self.thresh + 1
        self.display()

class AudiogramExpt():
    '''
    Example of updating a graph when mouse is clicked, which works on google colab, unlike
    other methods. E.G. you can't do self.widgets['graphoutput'].clear_output() on colab.
    '''
    def __init__(self, jupyterpsych=None):
        if jupyterpsych is None:
            jupyterpsych = JupyterPsych()
        self.jupyterpsych = jupyterpsych

        self.f_s = self.jupyterpsych.f_s

        self.freqs = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000]
        self.thresh = np.ones((len(self.freqs)))*np.nan
        self.widgets = self.setup_ui()
        self.display()

    def setup_ui(self):
        '''
        Set up widgets (but don't display them)
        '''
        w = {}
        w['graphoutput'] = widgets.Output()

        # freq slider
        w['freqslider'] = widgets.IntSlider(
            min=0, max=len(self.freqs)-1, step=1, value=int(len(self.freqs)/2), readout=False)
        w['freqvalue'] = widgets.Label('23 Hz')
        def f(_):
            w['freqvalue'].value = '%d Hz' % self.freqs[w['freqslider'].value]
        w['freqvalue'].value = '%d Hz' % self.freqs[w['freqslider'].value]
        w['freqslider'].observe(f)

        # level slider
        w['levelslider'] = widgets.IntSlider(
            min=0, max=90, step=1, value=70, readout=False)
        w['levelvalue'] = widgets.Label('34 dB')
        def l(_):
            w['levelvalue'].value = '%d dB SPL' % w['levelslider'].value
        w['levelvalue'].value = '%d dB SPL' % w['levelslider'].value
        w['levelslider'].observe(l)

        # arrange widgets in vertical boxes for alignment
        labelbox = widgets.VBox((widgets.Label('Frequency'), widgets.Label('Level')))
        sliderbox = widgets.VBox(
            (w['freqslider'], w['levelslider']))
        valuebox = widgets.VBox((w['freqvalue'], w['levelvalue']))

        # arrange boxes into a horizontal box -- this needs to be display()ed
        w['sliderbox'] = widgets.HBox((labelbox, sliderbox, valuebox))

        # buttons
        w['soundButton'] = widgets.Button(description="Play sound")
        w['soundButton'].on_click(self.on_soundButton_clicked)

        w['button'] = widgets.Button(description="Record this point")
        w['button'].on_click(self.on_recordButton_clicked)
        w['buttonbox'] = widgets.HBox(
            (w['soundButton'], w['button']))

        return w

    def plot_graph(self):
        _, ax = plt.subplots(figsize=(8, 3))
        if np.all(np.isnan(self.thresh)):
            plt.scatter(np.array(self.freqs), 1000*np.ones(len(self.freqs)))
        else:
            plt.scatter(np.array(self.freqs), self.thresh)
        ax.set_ylim((0, 90))
        ax.set_xscale('log')
        ax.set_xlim((10, 20000))
        ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Level (dB)')
        ax.grid(True, which='both')
        plt.show()

    def display(self):
        '''
        Clear and redraw the entire cell output, using a new Output to hold the new graph
        because you don't seem to be able to clear Outputs on colab.
        '''
        clear_output()
        self.widgets['graphoutput'] = widgets.Output()
        with self.widgets['graphoutput']:
            self.plot_graph()

        display(self.widgets['graphoutput'])
        display(self.widgets['sliderbox'])
        display(self.widgets['buttonbox'])

    def on_recordButton_clicked(self, _):
        '''
        Update graph data and redisplay whole UI
        '''
        freq_idx = self.widgets['freqslider'].value
        level = self.widgets['levelslider'].value
        if np.isnan(self.thresh[freq_idx]) or level < self.thresh[freq_idx]:
            self.thresh[freq_idx] = level
        self.display()

    def on_soundButton_clicked(self, _):
        '''
        Update graph data and redisplay whole UI
        '''
        # play sound

def print_setup_message():
    '''
    Print setup message
    '''
    print('\n=== Setup complete ===\n')
    print('Now, move on to the next cell to set up your headphones\n')
