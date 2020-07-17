#!/usr/bin/env python
# coding: utf-8

from io import BytesIO
import struct, math
import wave
import numpy as np
from matplotlib import pyplot as plt
import ipywidgets as widgets
from IPython.display import display, Audio
from benlib.sound import *

# remove glitchy padding around audioplayer widget
# https://github.com/jupyter-widgets/ipywidgets/issues/1845
from IPython.core.display import HTML, display as cdisplay
def rm_out_padding(): cdisplay(HTML("<style>div.output_subarea { padding:unset;}</style>"))
rm_out_padding()


# In[4]:


class AudioPlayer(Audio):

    def ndarray2wavbytes(self, fs, snd):
        nchannels = snd.shape[0]
        mx = np.max(np.abs(snd))
        snd = snd.transpose().ravel()
        if mx > 0:
            snd = snd/mx*32767.0
        snd_b = snd.astype(np.int16).tostring()
        output = BytesIO()
        with wave.open(output, 'w') as s:
            s.setnchannels(nchannels)
            s.setsampwidth(2)
            s.setframerate(fs)
            s.writeframesraw(snd_b)
        return output.getvalue()

    def update_data(self, fs, ndarray):
        self.data = self.ndarray2wavbytes(fs, ndarray)

    def _repr_html_(self):
        audio = super()._repr_html_()
        audio = audio.replace('<audio', f'<audio onended="this.parentNode.removeChild(this)"')
        return f'<div style="display:none">{audio}</div>'


# In[1]:


class LocalizationExpt():

    def __init__(self, freqs=[500, 3000], n_reps=6, type='ILD'):
        self.fs = 44100
        self.len_s = 0.5
        self.type = type
        if self.type=='ILD':
            self.indep = np.linspace(-5, 5, 8)
        else:
            self.type = 'ITD'
            self.indep = np.linspace(-500, 500, 8)

        self.freqs = np.array([500, 3000])
        self.n_reps = n_reps
        self.buttons = {}

        all_trial_params = []
        for indep in self.indep:
            for freq in self.freqs:
                all_trial_params.append([freq, indep])

        all_trial_params = np.tile(np.array(all_trial_params), (n_reps, 1))
        self.all_trial_params = np.random.permutation(all_trial_params)

        self.n_trials = all_trial_params.shape[0]

        self.trial_idx = 0

        self.responses = []

        self.widgets = {}
        self.widgets['audio'] = AudioPlayer(np.ones(20), rate=self.fs, autoplay=True)
        self.widgets['leftButton'] = widgets.Button(description='Left')
        self.widgets['leftButton'].on_click(lambda x: self.responseButton_clicked('left', x))
        self.widgets['leftButton'].disabled = True

        self.widgets['rightButton'] = widgets.Button(description='Right')
        self.widgets['rightButton'].on_click(lambda x: self.responseButton_clicked('right', x))
        self.widgets['rightButton'].disabled = True

        self.widgets['soundButton'] = widgets.Button(description="Play sound")
        self.widgets['soundButton'].on_click(self.on_soundButton_clicked)

        self.widgets['statusBox'] = widgets.Text('Trial %d of %d: Click "Play sound"' %
                                                 (self.trial_idx+1, self.n_trials))
        self.widgets['output'] = widgets.Output()

        self.widgets['responseButtons'] = widgets.HBox((self.widgets['leftButton'], self.widgets['rightButton']))
        display(self.widgets['statusBox'])
        display(self.widgets['soundButton'])
        display(self.widgets['responseButtons'])
        display(self.widgets['output'])
        display(self.widgets['audio'])

    def set_response_buttons_enabled(self, state):
        self.widgets['leftButton'].disabled = not state
        self.widgets['rightButton'].disabled = not state

    def set_sound_button_enabled(self, state):
        self.widgets['soundButton'].disabled = not state

    def set_status_text(self, string):
        self.widgets['statusBox'].value = string

    def on_soundButton_clicked(self, b):
        self.set_response_buttons_enabled(True)
        self.set_sound_button_enabled(False)
        (freq, indep) = self.all_trial_params[self.trial_idx, :]
        self.widgets['audio'].update_data(self.fs, self.stim_gen(freq, indep))
        self.widgets['audio'].autoplay = True
        display(self.widgets['audio'])
        self.set_status_text('Trial %d of %d: Click "Left" or "Right"' % (self.trial_idx+1, self.n_trials))

    def responseButton_clicked(self, side, b):
        self.set_response_buttons_enabled(False)
        self.responses.append(side)
        with self.widgets['output']:
            if self.trial_idx == self.n_trials-1:
                with self.widgets['output']:
                    self.set_status_text('Trial %d of %d: Experiment complete' % (self.trial_idx+1, self.n_trials))
                    self.set_response_buttons_enabled(False)
            else:
                self.trial_idx = self.trial_idx + 1
                self.set_status_text('Trial %d of %d: Click "Play sound"' % (self.trial_idx+1, self.n_trials))
                self.set_sound_button_enabled(True)

    def stim_gen(self, freq, indep):
        if self.type == 'ILD':
            return ild_stimulus(self.fs, self.len_s, freq, ild_dB=indep)
        return itd_stimulus(self.fs, self.len_s, freq, itd_us=indep)

    def analyse_results(self):
        freqs = np.unique(self.freqs)
        indep = np.unique(np.array(self.indep))
        if self.responses == []:
            with self.widgets['output']:
                print('Faking data')
            self.responses = list(self.all_trial_params[:, 1])
            self.responses = ['left' if r<=0 else 'right' for r in self.responses]

        n_right = np.zeros((len(freqs), len(indep)))
        n_total = np.zeros((len(freqs), len(indep)))
        for f, ff in enumerate(freqs):
            for i, ii in enumerate(indep):
                trial_idxes = np.where((self.all_trial_params[:, 0]==ff) & (self.all_trial_params[:, 1]==ii))
                for trial_idx in trial_idxes[0]:
                    n_total[f, i] = n_total[f, i] + 1
                    if self.responses[trial_idx] == 'right':
                        n_right[f, i] = n_right[f, i] + 1
        print(n_right)
        print(n_total)

        for f, ff in enumerate(freqs):
            plt.plot(indep, n_right[f, :]/n_total[f, :])

