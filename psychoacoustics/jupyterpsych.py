'''
Jupyter psychoacoustics
'''

# pylint: disable=C0103, R0912, R0914

from io import BytesIO
import wave
import numpy as np
from IPython.display import Audio

def is_colab():
    '''
    True if running in google colab
    '''
    try:
        import google.colab
        return True
    except ModuleNotFoundError:
        return False

class AudioPlayer(Audio):
    '''
    Customised AudioPlayer class
    '''

    def __init__(self, snd, hide_on_click=False, *args, **kwargs):
        self.data = None
        super().__init__(snd, *args, **kwargs)
        self.hide_on_click = hide_on_click

    def ndarray2wavbytes(self, fs, snd):
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

    def update_data(self, fs, ndarray):
        '''
        Update sound data
        '''
        self.data = self.ndarray2wavbytes(fs, ndarray)

    def _repr_html_(self):
        '''
        Modified HTML div
        '''
        audio = super()._repr_html_()
        if self.hide_on_click:
            return audio.replace('<audio', f'<audio onended="this.parentNode.removeChild(this)"')

        return audio
        # return f'<div style="height:1px">{audio}</div>'
