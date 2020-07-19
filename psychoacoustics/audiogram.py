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

        self.widgets = {}
        self.widgets['graphoutput'] = widgets.Output()

        self.widgets['freqslider'] = widgets.IntSlider(
            min=0, max=len(self.freqs)-1, step=1, value=int(len(self.freqs)/2), readout=True)

        self.widgets['levelslider'] = widgets.IntSlider(
            min=0, max=90, step=1, value=70, readout=True)

        sliderbox = widgets.VBox(
            (self.widgets['freqslider'], self.widgets['levelslider']))

        labelbox = widgets.VBox((widgets.Label('Frequency (Hz)'), widgets.Label('Level (dB)')))

        self.widgets['sliderbox'] = widgets.HBox((labelbox, sliderbox))

        self.widgets['soundButton'] = widgets.Button(description="Play sound")
        self.widgets['soundButton'].on_click(self.on_soundButton_clicked)

        self.widgets['button'] = widgets.Button(description="Update")
        self.widgets['button'].on_click(self.on_button_clicked)
        self.widgets['buttonbox'] = widgets.HBox(
            (self.widgets['soundButton'], self.widgets['button']))

        self.display()

    def display(self):
        '''
        Clear and redraw the entire cell output, using a new Output to hold the new graph
        because you don't seem to be able to clear Outputs on colab.
        '''
        clear_output()
        self.widgets['graphoutput'] = widgets.Output()
        with self.widgets['graphoutput']:
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
        display(self.widgets['graphoutput'])
        display(self.widgets['sliderbox'])
        display(self.widgets['buttonbox'])

    def on_button_clicked(self, _):
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
        # print(self.widgets['freqslider'].value)
        # self.thresh = self.thresh + 1
        self.display()


class AudiogramExptOld():
    '''
    Audiogram expt
    '''

    def __init__(self, jupyterpsych=None):
        if jupyterpsych is None:
            jupyterpsych = JupyterPsych()
        self.jupyterpsych = jupyterpsych

        self.f_s = self.jupyterpsych.f_s
        self.len_s = 0.5

        self.responses = []
        self.results = {}

        self.freqs = np.logspace(np.log10(50), np.log10(20000))
        self.thresh = np.ones(self.freqs.shape)

        # graph
        self.widgets = {}

        self.widgets['graphoutput'] = widgets.Output()

        with self.widgets['graphoutput']:
            fig, ax = plt.subplots()
            ax.set_ylim((0, 10))
            line = plt.plot(self.freqs, self.thresh)
            plt.show()

        display(self.widgets['graphoutput'])


        # self.widgets['graphoutput'].clear_output()

        self.widgets['audio'] = AudioPlayer(np.ones((self.f_s)), rate=self.f_s,
                                            autoplay=True, hide_on_click=True)
        self.widgets['yesButton'] = widgets.Button(description='Yes')
        self.widgets['yesButton'].on_click(lambda x: self.responseButton_clicked('yes', x))
        self.widgets['yesButton'].disabled = True

        self.widgets['noButton'] = widgets.Button(description='No')
        self.widgets['noButton'].on_click(lambda x: self.responseButton_clicked('no', x))
        self.widgets['noButton'].disabled = True

        self.widgets['soundButton'] = widgets.Button(description="Play sound")
        self.widgets['soundButton'].on_click(self.on_soundButton_clicked)

        self.widgets['statusBox'] = widgets.Text('Click "Play sound"')
        self.widgets['output'] = widgets.Output()

        self.widgets['responseButtons'] = widgets.HBox(
            (self.widgets['yesButton'], self.widgets['noButton']))
        display(self.widgets['statusBox'])
        display(self.widgets['soundButton'])
        display(self.widgets['responseButtons'])
        display(self.widgets['output'])

    def set_response_buttons_enabled(self, state):
        '''
        Enable / disable Left/Right buttons
        '''
        self.widgets['yesButton'].disabled = not state
        self.widgets['noButton'].disabled = not state

    def set_sound_button_enabled(self, state):
        '''
        Enable / disable "Play Sound" button
        '''
        self.widgets['soundButton'].disabled = not state

    def set_status_text(self, string):
        '''
        Update status widget
        '''
        self.widgets['statusBox'].value = string

    def on_soundButton_clicked(self, _):
        '''
        When soundButton is clicked, update audio,
        enable left/right buttons, and disable sound button
        '''
        self.set_response_buttons_enabled(True)
        self.set_sound_button_enabled(False)
        self.thresh[0] = self.thresh[0] + 1

        self.widgets['graphoutput'].clear_output()
        with self.widgets['graphoutput']:
            fig, ax = plt.subplots()
            ax.set_ylim((0,10))
            self.line = plt.plot(self.freqs, self.thresh)
            plt.show()
            # self.fig.canvas.draw()
        # freq, indep = self.all_trial_params[self.trial_idx, :]
        # self.widgets['audio'].update_data(self.f_s, self.stim_gen(freq, indep))
        display(self.widgets['audio'])
        self.set_status_text('Click "Yes" or "No"')

    def responseButton_clicked(self, side, _):
        '''
        When left/right are clicked, enable the sound button
        and update status text. Finish expt if all trials have been run
        '''
        self.set_response_buttons_enabled(False)
        self.responses.append(side)
        if False: #self.trial_idx == self.n_trials-1:
            # with self.widgets['output']:
            #     self.set_status_text('Trial %d of %d: Experiment complete' %
            #                          (self.trial_idx+1, self.n_trials))
            # self.set_response_buttons_enabled(False)
            pass
        else:
            # self.trial_idx = self.trial_idx + 1
            # self.set_status_text('Trial %d of %d: Click "Play sound"' %
            #                      (self.trial_idx+1, self.n_trials))
            self.set_sound_button_enabled(True)

    def stim_gen(self, freq, indep):
        '''
        Generate ILD/ITD stimulus
        '''
        if self.type == 'ILD':
            return ild_stimulus(self.f_s, self.len_s, freq, ild_dB=0)
        return itd_stimulus(self.f_s, self.len_s, freq, itd_us=0)

    def generate_data(self):
        '''
        Generate fake response data for testing
        '''
        rng = (np.max(self.indep) - np.min(self.indep))/8
        self.responses = []
        for idx in range(self.all_trial_params.shape[0]):
            _, indep = self.all_trial_params[idx]
            prob = logistic(indep, 0, rng)
            if np.random.random() < prob:
                self.responses.append('right')
            else:
                self.responses.append('left')

    def x_label(self):
        '''
        Label for independent variable
        '''
        if self.type == 'ILD':
            return 'ILD (dB)'
        return 'ITD (us)'

    def results_table(self):
        '''
        Print results table
        '''
        titles = self.x_label() + ' : '
        for indep in self.indep:
            titles = titles + '%7.1f  ' % indep

        for (freq, res) in self.results.items():
            print('Frequency %d Hz' % freq)

            values = '% right  : '
            for val in res['prop_r']:
                values = values + '%7.1f  ' % (val * 100)
            print(titles)
            print(values)
            print()

    def plot_results(self):
        '''
        Plot results
        '''
        if is_colab():
            figsize = (14, 8)
            plt.rc('font', size=15)
        else:
            figsize = (9, 6)

        plt.figure(figsize=figsize)

        for (_, res) in self.results.items():
            ax = plt.gca()
            col = ax._get_lines.get_next_color()
            plt.scatter(self.indep, res['prop_r'])
            plt.plot(res['indep_spc'], res['r_hat'], color=col)
            plt.fill_between(res['indep_spc'], res['ci_5'], res['ci_95'],
                             color=col, alpha=0.1)

        plt.legend(['%d Hz tone' % f for f in self.results])
        plt.xlabel(self.x_label())
        plt.ylabel('Percentage right responses')

    def analyse_results(self, use_test_data=False):
        '''
        Collate data, and show table and graph of results
        '''
        freqs = np.unique(self.freqs)
        indeps = np.unique(np.array(self.indep))

        if use_test_data:
            self.generate_data()

        # generate ndarray of responses (0=left, 1=right)
        responses = np.array([1 if r == 'right' else 0 for r in self.responses])

        # collate results as n_r, n_t, prop_r
        self.results = {}

        for freq in freqs:
            res = {}
            w = np.where(self.all_trial_params[:, 0] == freq)[0]
            res['all_indep'] = self.all_trial_params[w, 1]
            res['all_resp'] = responses[w]

            res['n_t'], res['n_r'], res['prop_r'] = \
                collate_responses(indeps, res['all_indep'], res['all_resp'])

            res['fit_params'], res['r_hat'], (res['indep_spc'], res['ci_5'], res['ci_95']) = \
                probit_fit(res['all_indep'], res['all_resp'])

            self.results[freq] = res

        self.results_table()
        self.plot_results()

def print_setup_message():
    '''
    Print setup message
    '''
    print('\n=== Setup complete ===\n')
    print('Now, move on to the next cell to set up your headphones\n')
