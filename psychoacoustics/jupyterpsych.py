'''
Jupyter psychoacoustics
'''

# pylint: disable=C0103, R0912, R0914

import warnings
from io import BytesIO
import wave
import numpy as np
from IPython.display import Audio
from IPython.core.display import HTML, display as cdisplay
from statsmodels.tools.sm_exceptions import ConvergenceWarning

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
