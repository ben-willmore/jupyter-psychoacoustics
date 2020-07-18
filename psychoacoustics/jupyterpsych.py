'''
Jupyter psychoacoustics
'''

# pylint: disable=C0103, R0912, R0914

import warnings
from io import BytesIO
import wave
import numpy as np
from IPython.display import Audio, display
from IPython.core.display import HTML, display as cdisplay
import ipywidgets as widgets
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from psychoacoustics.sound import ild_stimulus

F_S = 44100

def rm_out_padding():
    '''
    remove glitchy padding around audioplayer widget
    # https://github.com/jupyter-widgets/ipywidgets/issues/1845
    '''
    cdisplay(HTML("<style>div.output_subarea { padding:unset;}</style>"))

rm_out_padding()

# suppress statsmodels warnings
warnings.simplefilter('ignore', ConvergenceWarning)

def is_colab():
    '''
    True if running in google colab
    '''
    try:
        import google.colab
        return True
    except ModuleNotFoundError:
        return False

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

def ndarray2wavbytes(fs, snd):
    '''
    Convert ndarray to wav format bytes
    '''
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

class AudioPlayer(Audio):
    '''
    Customised AudioPlayer class
    '''

    def __init__(self, snd, *args, hide_on_click=False, **kwargs):
        self.data = None
        super().__init__(snd, *args, **kwargs)
        self.hide_on_click = hide_on_click

    def update_data(self, fs, ndarray):
        '''
        Update sound data
        '''
        self.data = ndarray2wavbytes(fs, ndarray)

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
