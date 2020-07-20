'''
Sound localization practical using jupyter / google colab
'''

# pylint: disable=C0103, R0912, R0914

from pathlib import Path
import json
import numpy as np
from matplotlib import pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from psychoacoustics.sound import ild_stimulus, itd_stimulus
from psychoacoustics.stats import logistic, probit_fit
from psychoacoustics.jupyterpsych import JupyterPsych, AudioPlayer, collate_responses

F_S = 44100

class LocalizationExpt():
    '''
    Sound localisation expt
    '''

    def __init__(self, freqs=None, n_reps=10, cue_type='ILD', jupyterpsych=None):
        if jupyterpsych is None:
            jupyterpsych = JupyterPsych()
        self.jupyterpsych = jupyterpsych

        # self.jupyterpsych.remove_widget_padding()

        self.f_s = self.jupyterpsych.f_s
        self.len_s = 0.5
        self.cue_type = cue_type.upper()
        if self.cue_type == 'ILD':
            self.indep = np.linspace(-5, 5, 8)
        else:
            self.cue_type = 'ITD'
            self.indep = np.linspace(-300, 300, 8)

        if freqs is None:
            freqs = [500, 2000]
        self.freqs = np.array(freqs)

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
        self.results = {}

        self.widgets = {}
        freq, indep = self.all_trial_params[0, :]
        self.widgets['audio'] = AudioPlayer(self.stim_gen(freq, indep), rate=self.f_s,
                                            autoplay=True, hide_on_click=True)
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

        self.widgets['responseButtons'] = widgets.HBox(
            (self.widgets['leftButton'], self.widgets['rightButton']))
        display(self.widgets['statusBox'])
        display(self.widgets['soundButton'])
        display(self.widgets['responseButtons'])
        display(self.widgets['output'])

    def set_response_buttons_enabled(self, state):
        '''
        Enable / disable Left/Right buttons
        '''
        self.widgets['leftButton'].disabled = not state
        self.widgets['rightButton'].disabled = not state

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
        freq, indep = self.all_trial_params[self.trial_idx, :]
        self.widgets['audio'].update_data(self.f_s, self.stim_gen(freq, indep))
        with self.widgets['output']:
            display(self.widgets['audio'])
        self.set_status_text('Trial %d of %d: Click "Left" or "Right"' %
                             (self.trial_idx+1, self.n_trials))

    def responseButton_clicked(self, side, _):
        '''
        When left/right are clicked, enable the sound button
        and update status text. Finish expt if all trials have been run
        '''
        self.set_response_buttons_enabled(False)
        with self.widgets['output']:
            clear_output()
        self.responses.append(side)

        if self.trial_idx == self.n_trials-1:
            self.set_status_text('Trial %d of %d: Experiment complete' %
                                 (self.trial_idx+1, self.n_trials))
            self.set_response_buttons_enabled(False)
        else:
            self.trial_idx = self.trial_idx + 1
            self.set_status_text('Trial %d of %d: Click "Play sound"' %
                                 (self.trial_idx+1, self.n_trials))
            self.set_sound_button_enabled(True)

    def stim_gen(self, freq, indep):
        '''
        Generate ILD/ITD stimulus
        '''
        if self.cue_type == 'ILD':
            return ild_stimulus(self.f_s, self.len_s, freq, ild_dB=indep)
        return itd_stimulus(self.f_s, self.len_s, freq, itd_us=indep)

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
        if self.cue_type == 'ILD':
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
        if self.jupyterpsych.is_colab():
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

    def dump_data(self, filename):
        '''
        Dump data to JSON file
        '''
        results = {'cue_type': self.cue_type,
                   'all_trial_params': self.all_trial_params.tolist(),
                   'responses': self.responses}

        with open(filename, 'w') as outfile:
            json.dump(results, outfile)

    def load_data(self, filename=None):
        '''
        Load saved data from JSON file
        '''
        if not filename:
            filename = Path(Path(__file__).parent, 'data', self.cue_type.upper()+ '_results.json')

        with open(filename, 'r') as infile:
            results = json.load(infile)

        self.cue_type = results['cue_type']
        self.all_trial_params = np.array(results['all_trial_params'])
        self.responses = results['responses']
        self.freqs = np.unique(np.array(self.all_trial_params)[:, 0])
        self.indep = np.unique(np.array(self.all_trial_params)[:, 1])

def print_setup_message():
    '''
    Print setup message
    '''
    print('\n=== Setup complete ===\n')
    print('Now, move on to the next cell to set up your headphones\n')
