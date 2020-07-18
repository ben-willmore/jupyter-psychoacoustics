'''
Sound localization practical using jupyter / google colab
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np
from matplotlib import pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from statsmodels.tools.tools import add_constant
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from .sound import ild_stimulus, itd_stimulus
from .stats import logistic
from .jupyterpsych import is_colab, AudioPlayer

F_S = 44100

class LocalizationExpt():
    '''
    Sound localisation expt
    '''

    def __init__(self, freqs=None, n_reps=8, type='ILD'):
        self.fs = F_S
        self.len_s = 0.5
        self.type = type.upper()
        if self.type == 'ILD':
            self.indep = np.linspace(-5, 5, 8)
        else:
            self.type = 'ITD'
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
        self.widgets['audio'] = AudioPlayer(self.stim_gen(freq, indep), rate=self.fs,
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
        self.widgets['audio'].update_data(self.fs, self.stim_gen(freq, indep))
        display(self.widgets['audio'])
        self.set_status_text('Trial %d of %d: Click "Left" or "Right"' %
                             (self.trial_idx+1, self.n_trials))

    def responseButton_clicked(self, side, _):
        '''
        When left/right are clicked, enable the sound button
        and update status text. Finish expt if all trials have been run
        '''
        self.set_response_buttons_enabled(False)
        self.responses.append(side)
        with self.widgets['output']:
            if self.trial_idx == self.n_trials-1:
                with self.widgets['output']:
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
        if self.type == 'ILD':
            return ild_stimulus(self.fs, self.len_s, freq, ild_dB=indep)
        return itd_stimulus(self.fs, self.len_s, freq, itd_us=indep)

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

    def analyse_results(self, use_test_data=False):
        '''
        Make table and graph of results
        '''
        freqs = np.unique(self.freqs)
        indeps = np.unique(np.array(self.indep))

        if use_test_data:
            self.generate_data()

        # generate ndarray of responses (0=left, 1=right)
        responses = np.array([1 if r == 'right' else 0 for r in self.responses])

        # collate results as n_r, n_t, pct_correct
        self.results = {}
        n_indep = indeps.shape[0]

        for freq in freqs:
            n_t = np.zeros((n_indep))
            n_r = np.zeros((n_indep))
            pct_correct = np.zeros((n_indep))
            w = np.where(self.all_trial_params[:, 0] == freq)[0]
            all_indep = self.all_trial_params[w, 1]
            all_resp = responses[w]

            for i_idx, indep in enumerate(indeps):
                trial_idxes = np.where((self.all_trial_params[:, 0] == freq) &
                                       (self.all_trial_params[:, 1] == indep))[0]

                n_t[i_idx] = trial_idxes.shape[0]
                n_r[i_idx] = np.sum(responses[trial_idxes])
                pct_correct[i_idx] = n_r[i_idx]/n_t[i_idx]

            # binomial GLM with probit link
            model = GLM(all_resp, add_constant(all_indep),
                        family=families.Binomial(),
                        link=families.links.probit())
            mod_result = model.fit(disp=0)
            xt = np.linspace(np.min(all_indep), np.max(all_indep), 100)
            r_hat = mod_result.predict(add_constant(xt))
            pred_summ = mod_result.get_prediction(add_constant(xt)).summary_frame(alpha=0.05)
            ci_5, ci_95 = pred_summ['mean_ci_lower'], pred_summ['mean_ci_upper']

            res = {'all_indep': all_indep,
                   'all_resp': all_resp,
                   'n_t': n_t,
                   'n_r': n_r,
                   'pct_correct': pct_correct,
                   'indep_spc': xt,
                   'r_hat': r_hat,
                   'ci_5': ci_5,
                   'ci_95': ci_95,
                   'mod_result': mod_result}
            self.results[freq] = res

        # display table and figure of results
        if is_colab():
            figsize = (12, 8)
            plt.rc('font', size=14)
        else:
            figsize = (9, 6)

        colors = ['b', 'r']

        # table and figure
        plt.figure(figsize=figsize)

        if type == 'ILD':
            title = 'ILD (dB)'
        else:
            title = 'ITD (us)'

        titles = title + ' : '
        for indep in indeps:
            titles = titles + '%7.1f  ' % indep

        for f_idx, (freq, res) in enumerate(self.results.items()):
            print('Frequency %d Hz' % freq)

            values = '% right  : '
            for val in res['pct_correct']:
                values = values + '%7.1f  ' % val
            print(titles)
            print(values)

            plt.scatter(indeps, res['pct_correct'], c=colors[f_idx])
            plt.plot(res['indep_spc'], res['r_hat'], colors[f_idx])
            plt.fill_between(res['indep_spc'], res['ci_5'], res['ci_95'],
                             color=colors[f_idx], alpha=0.1)

        plt.legend(['%d Hz tone' % f for f in freqs])
        plt.xlabel(title)
        plt.ylabel('Percentage right responses')

def print_setup_message():
    '''
    Print setup message
    '''
    print('\n=== Setup complete ===\n')
    print('Now, move on to the next cell to set up your headphones\n')

def headphone_check():
    '''
    Show widgets which play left/right sounds
    '''
    left_stim = ild_stimulus(F_S, 2, 500, ild_dB=-100)
    left_widget = AudioPlayer(left_stim, rate=F_S, autoplay=False)
    right_stim = ild_stimulus(F_S, 2, 500, ild_dB=100)
    right_widget = AudioPlayer(right_stim, rate=F_S, autoplay=False)
    display(widgets.Label('This sound should play in the left headphone only:'))
    display(left_widget)
    display(widgets.Label('This sound should play in the right headphone only:'))
    display(right_widget)
