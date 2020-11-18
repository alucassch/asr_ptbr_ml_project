import os
import numpy as np
import subprocess
from typing import Union
from .helper import tmpdir

from kaldiio import ReadHelper

class KaldiParam(object):
    def __init__(self, num_mel_bins: int = 80) -> None:
        self.num_mel_bins = num_mel_bins

    @staticmethod
    def run_cmd(cmd: str) -> Union[bool, int]:
        devnull = open('/dev/null','w')
        proc = subprocess.Popen(cmd, shell=True, stderr=devnull, stdout=devnull)
        ret = proc.wait()
        devnull.close()
        return ret

    def fbank_features(self, audiofile: str) -> np.ndarray:
        fbank = None
        with tmpdir() as tmp:
            with open(tmp+'/wav.scp', 'w') as f:
                f.write(f'x {os.path.abspath(audiofile)}')
            cmd = f"compute-fbank-feats --num-mel-bins={self.num_mel_bins} scp,p:{tmp+'/wav.scp'} ark:{tmp+'/fbank.ark'}"
            if self.run_cmd(cmd) == 0:
                with ReadHelper(f"ark:{tmp+'/fbank.ark'}") as reader:
                    for _, numpy_array in reader:
                        fbank = numpy_array
            else:
                raise ValueError(f"Could not extract fbank params from file {audiofile}")
            
        return fbank
    
    def pitch_features(self, audiofile: str) -> np.ndarray:
        pitch = None
        with tmpdir() as tmp:
            with open(tmp+'/wav.scp', 'w') as f:
                f.write(f'x {os.path.abspath(audiofile)}')
            cmd = f"compute-kaldi-pitch-feats scp,p:{tmp+'/wav.scp'} ark:- | process-kaldi-pitch-feats ark:- ark:{tmp+'/pitch.ark'}"
            if self.run_cmd(cmd) == 0:
                with ReadHelper(f"ark:{tmp+'/pitch.ark'}") as reader:
                    for _, numpy_array in reader:
                        pitch = numpy_array
            else:
                raise ValueError(f"Could not extract pitch params from file {audiofile}")
            
        return pitch
    
    def extract_fbank_pitch(self, audiofile: str) -> np.ndarray:
        fbank = self.fbank_features(audiofile)
        pitch = self.pitch_features(audiofile)
        return np.concatenate((fbank,pitch), axis=1)