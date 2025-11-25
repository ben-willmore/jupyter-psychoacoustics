'''
Jupyter psychoacoustics
'''

# pylint: disable=C0103, R0912, R0914

import warnings
from io import BytesIO
import wave
import numpy as np
from IPython.display import Audio, display
from IPython.core.display import HTML, display as coredisplay
import ipywidgets as widgets
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from psychoacoustics.sound import ild_stimulus, make_diotic, whitenoise, level2amp, dBSPL2rms

# suppress statsmodels warnings
warnings.simplefilter('ignore', ConvergenceWarning)

class JupyterPsych():
    '''
    Basic setup of common experimental parameters
    '''

    def __init__(self):
        self.f_s = 44100
        self.calib_multiplier = None

    def headphone_check(self):
        '''
        Show widgets which play left/right sounds
        '''
        left_stim = ild_stimulus(self.f_s, 2, 500, ild_dB=-100)
        left_widget = AudioPlayer(left_stim, rate=self.f_s, autoplay=False)
        right_stim = ild_stimulus(self.f_s, 2, 500, ild_dB=100)
        right_widget = AudioPlayer(right_stim, rate=self.f_s, autoplay=False)
        display(widgets.Label('This sound should play in the left headphone only:'))
        display(left_widget)
        display(widgets.Label('This sound should play in the right headphone only:'))
        display(right_widget)

    def calibrate_sound_level(self):
        '''
        Calibrate sound level to match level of rubbing hands together (roughly 60dB SPL)
        '''
        n_samples = self.f_s

        snd = make_diotic(whitenoise(n_samples, method='uniform'))
        wn = snd/np.max(np.abs(snd))
        max_rms = np.sqrt(np.mean(wn**2))

        def f(Volume):
            '''
            Called when slider is created and adjusted
            '''

            # slider value of 1 == -20dB, slider value of 10 == 0dB
            dB_atten = -20+(Volume-1)*(20/9)
            mult = level2amp(dB_atten)
            audio = AudioPlayer(wn*mult, rate=self.f_s, autoplay=False,
                                scale_to_max=False, hide_on_click=False)
            audio.update_data(self.f_s, wn*mult, scale_to_max=False)
            display(audio)

            # If we generate a sound with RMS_out=1, we want to get a sound with RMS_Pa=1,
            # i.e. expected_rms_Pa=1
            #
            # If we calibrate with a sound with expected_rms_Pa=X and actual_rms_Pa=Y, then:
            # multiplier = expected_rms_Pa / actual_rms_Pa
            expected_rms_Pa = mult * max_rms
            actual_rms_Pa = dBSPL2rms(60)
            self.calib_multiplier = expected_rms_Pa / actual_rms_Pa

        txt = ('Adjust the slider and your computer\'s '
               'volume so that the level of the sound is 60 dB SPL.\n'
               'If you have a decibel meter (e.g. on your phone), use that.\n'
               'Otherwise, do this roughly by adjusting the sound so its level matches the '
               'sound made by rubbing your hands together:')

        display(widgets.Textarea(txt, layout={'width': '100%'}, rows=3))

        widgets.interact(
            f, Volume=widgets.IntSlider(min=1, max=10, step=1, value=10, readout=False))

    @classmethod
    def is_colab(cls):
        '''
        True if running in google colab
        '''
        try:
            import google.colab
            return True
        except ModuleNotFoundError:
            return False

    @classmethod
    def remove_widget_padding(cls):
        '''
        remove glitchy padding around audioplayer widget
        # https://github.com/jupyter-widgets/ipywidgets/issues/1845
        '''
        coredisplay(HTML("<style>div.output_subarea { padding:unset;}</style>"))




class AudioPlayer(Audio):
    '''
    Customised AudioPlayer class
    '''

    def __init__(self, snd, *args, rate=None, hide_on_click=False, scale_to_max=True, **kwargs):
        '''
        Initialise with empty sound, then update using update_data, so that we can set the
        level of the sound if we want to by setting scale_to_max=False.
         If multichannel, sound should be (n_channels, n_samples)
        '''
        self.data = None
        super().__init__(np.ones(snd.shape), rate=rate, *args, **kwargs)
        self.update_data(rate, snd, scale_to_max=scale_to_max)
        self.hide_on_click = hide_on_click

    @classmethod
    def ndarray2wavbytes(cls, fs, snd, scale_to_max=True):
        '''
        Convert ndarray to wav format bytes. If multichannel, sound should be
        (n_channels, n_samples)
        '''
        if len(snd.shape) == 1:
            nchannels = 1
        else:
            nchannels = snd.shape[0]
            snd = snd.transpose().ravel()

        mx = np.max(np.abs(snd))

        if scale_to_max and mx > 0:
            # if sound is empty, don't attempt to scale it
            snd = snd/mx*32767.0
        else:
            if mx > 1:
                print('Clipping sound on wav conversion')
            snd = snd * 32767.0

        snd_b = snd.astype(np.int16_).tostring()
        output = BytesIO()
        with wave.open(output, 'wb') as s:
            # (nchannels, sampwidth, framerate, nframes, comptype, compname)
            s.setparams((nchannels, 2, fs, 1, 'NONE', 'NONE'))
            s.writeframesraw(snd_b)
        return output.getvalue()

    def update_data(self, fs, ndarray, scale_to_max=True):
        '''
        Update sound data
        '''
        self.data = self.ndarray2wavbytes(fs, ndarray, scale_to_max=scale_to_max)

    def _repr_html_(self):
        '''
        Modified HTML div
        '''
        audio = super()._repr_html_()
        if self.hide_on_click:
            return audio.replace('<audio', f'<audio onended="this.parentNode.removeChild(this)"')

        return audio
        # return f'<div style="height:1px">{audio}</div>'

def collate_responses(x_unique, x, resp):
    '''
    Collate responses where x is independent variable, resp is 1 or 0
    and x_unique are unique values of x, into:
    n_t: number of trials (for each value of x_unique)
    n_r: number of 1 (e.g. 'right' responses
    prop_r: n_r / n_t
    '''
    n_x = x_unique.shape[0]
    n_t = np.zeros((n_x))
    n_r = np.zeros((n_x))
    prop_r = np.zeros((n_x))

    for i_idx, indep in enumerate(x_unique):
        trial_idxes = np.where((x == indep))[0]

        n_t[i_idx] = trial_idxes.shape[0]
        n_r[i_idx] = np.sum(resp[trial_idxes])
        prop_r[i_idx] = n_r[i_idx]/n_t[i_idx]

    return n_t, n_r, prop_r
