# coding: utf-8
"""
Preprocess dataset

usage: preprocess.py [options] <wavefiles_path> <feats_path>

options:
    --num_workers=<n>        Num workers.
    --hparams=<parmas>       Hyper parameters [default: ].
    --preset=<json>          Path of preset parameters (json).
    -h, --help               Show help message.
"""

from docopt import docopt
import time
import sys
import glob
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import librosa
import librosa.display
import matplotlib
import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from hparams import hparams, hparams_debug_string
import utils

def deltas(X_in):
    assert (len(X_in.shape)==3)
    # X_in [ channel : num_feats : time_freams ]
    X_out = (X_in[:,:,2:]-X_in[:,:,:-2])/10.0
    X_out = X_out[:,:,1:-1]+(X_in[:,:,4:]-X_in[:,:,:-4])/5.0
    return X_out

def logmeldeltasx3(logmel):
    
    # compute deltas
    logmel_deltas = deltas(logmel)
    logmel_deltas_deltas = deltas(logmel_deltas)

    # concatenate
    return np.concatenate((logmel[:,:,4:-4],logmel_deltas[:,:,2:-2],logmel_deltas_deltas),axis=0)

def unit_do_deltas():
    logmel = np.load('./datasets/multi_channel/pedestrian-barcelona-141-4299-a-D.npy')
    logmel = np.expand_dims(logmel, axis=0)
    result = logmeldeltasx3(logmel)
    print(result.shape)

class PreProcessBase(object):

    def __init__(self, wavefiles_path, feats_path = "datasets/features"):
        self.wavefiles_path = wavefiles_path
        self.feats_path = feats_path

        if not os.path.exists(feats_path):
            os.makedirs(feats_path,exist_ok=True)
            print("Creat a folder: %s"%feats_path)

    def extract_feature(self, wavefile_path):
        raise Exception("Please implement this function")

    def process(self, num_workers):

        executor = ProcessPoolExecutor(max_workers=num_workers)
        futures = []

        all_wavefiles = glob.glob('%s/*.wav'%self.wavefiles_path)
        print('Find %d wave files'%len(all_wavefiles))

        for wavefile_path in all_wavefiles:
            futures.append(executor.submit(partial(self.extract_feature, wavefile_path)))

        return [future.result() for future in tqdm(futures)]

def LogMelOneChannelSave(x, feats_path, utt_id, channel_type):
    F = librosa.feature.melspectrogram(x,
                        sr=hparams.sample_rate,
                        n_fft=hparams.n_fft,
                        hop_length=hparams.hop_length,
                        win_length=hparams.win_length,
                        n_mels=hparams.num_mels)

    F = np.log10(np.maximum(F, 1e-10))

    # print(F.shape)
    if (F.shape[-1] - 500) < 3 and (F.shape[-1] - 500) > 0: F=F[:,:500]

    assert(F.shape[-1] == 500)
    F = np.expand_dims(np.flip(F), axis=0) # (1, 40, 500)

    if hparams.two_channel:
        the_feats_path = os.path.join(feats_path, "%s-%s.npy"%(utt_id, channel_type))
    else:
        the_feats_path = os.path.join(feats_path, "%s.npy"%utt_id)

    if hparams.deltas:
        F = logmeldeltasx3(F)
        
    np.save(the_feats_path, F.astype(np.float32), allow_pickle=False)
        

class LogMelPreProcess(PreProcessBase):

    def __init__(self, wavefiles_path, feats_path = "datasets/features"):
        super().__init__(wavefiles_path, feats_path)

    def extract_feature(self, wavefile_path):
    
        if hparams.two_channel:
            utt_id = utils.GetUttID(wavefile_path)
            x, sr  = librosa.load(wavefile_path, hparams.sample_rate, mono=False)
            r_x = x[0].copy()        # channel 1
            l_x = x[1].copy()        # channel 2
            a_x = (r_x + l_x) / 2.0  # average
            d_x = r_x - l_x          # difference

            LogMelOneChannelSave(r_x, self.feats_path, utt_id, channel_type="R")
            LogMelOneChannelSave(l_x, self.feats_path, utt_id, channel_type="L")
            LogMelOneChannelSave(a_x, self.feats_path, utt_id, channel_type="A")
            LogMelOneChannelSave(d_x, self.feats_path, utt_id, channel_type="D")

        else:
            utt_id = utils.GetUttID(wavefile_path)
            x , sr = librosa.load(wavefile_path, hparams.sample_rate)
            LogMelOneChannelSave(x, self.feats_path, utt_id, channel_type="A")

        return

def filterMel(spectrogram):
    mel_filter = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels)
    S = np.dot(mel_filter, spectrogram)
    S = np.abs(S)
    S = np.log10(np.maximum(S, 1e-10))
    if (S.shape[-1] - 500) < 3 and (S.shape[-1] - 500) > 0: S=S[:,:500]
    assert(S.shape[-1] == 500)
    S = np.expand_dims(np.flip(S), axis=0) # (1, 40, 500)
    return S

class HPSSPreProcess(PreProcessBase):
    
    def __init__(self, wavefiles_path, feats_path = "datasets/features"):
        super().__init__(wavefiles_path, feats_path)

    def extract_feature(self, wavefile_path):
    
        assert (hparams.do_hpss)
        utt_id = utils.GetUttID(wavefile_path)
        x, sr  = librosa.load(wavefile_path, hparams.sample_rate, mono=False)
        assert (sr==hparams.sample_rate)

        S = librosa.stft(x, n_fft=hparams.n_fft,
                        hop_length=hparams.hop_length,
                        win_length=hparams.win_length)

        H, P = librosa.decompose.hpss(S)

        # S = filterMel(S)
        H = filterMel(H)
        P = filterMel(P)

        F = np.concatenate((H,P), axis=0)
        # print(F.shape)

        the_feats_path = os.path.join(self.feats_path, "%s.npy"%utt_id)
        np.save(the_feats_path, F.astype(np.float32), allow_pickle=False)
        # self.HpssPlot(utt_id, F)


    def HpssPlot(self, utt_id, F):
        plt.figure()
        fig, ax = plt.subplots(1,3, figsize=(16,8))

        fig.suptitle(utt_id)

        ax[1].matshow(F[0])
        ax[1].set_title("Harmonic")
        ax[2].matshow(F[1])
        ax[2].set_title("Percussive")

        plots_path = os.path.join(self.feats_path, 'plot2check')
        os.makedirs(plots_path, exist_ok=True)
        # plt.tight_layout()
        plt_path=os.path.join(plots_path, "%s.png"%utt_id)
        
        fig.savefig(plt_path, format="png")

        return

def PlotAFigure(S, file_path):
    plt.matshow(S)
    plt.savefig(file_path, format="png")
    plt.close()


def PlotToCheck(feats_path, num):
    from pathlib import Path
    import random

    plots_path = os.path.join(Path(feats_path).parent, 'plot2check')
    os.system('rm -rf %s/*.png'%plots_path)
    os.makedirs(plots_path, exist_ok=True)

    all_file = os.listdir(feats_path)

    all_utt_ids = list(set([ utils.GetUttID(x) for x in all_file ]))
    num_all = len(all_utt_ids)

    utt_ids = None
    if num < num_all:
        random.shuffle(all_utt_ids)
        utt_ids = all_utt_ids[:num]
    else:
        utt_ids = all_utt_ids

    for i in utt_ids:
        R=np.load(os.path.join(feats_path, "%s-R.npy"%i))
        L=np.load(os.path.join(feats_path, "%s-L.npy"%i))
        A=np.load(os.path.join(feats_path, "%s-A.npy"%i))
        D=np.load(os.path.join(feats_path, "%s-D.npy"%i))
        
        print("PlotToCheck:", "R", R.shape, "L",L.shape, "A", A.shape, "D", D.shape, i)

        # PlotAFigure(R, os.path.join(feats_path, "%s-R.png"%i))
        # PlotAFigure(L, os.path.join(feats_path, "%s-L.png"%i))
        # PlotAFigure(A, os.path.join(feats_path, "%s-A.png"%i))
        # PlotAFigure(D, os.path.join(feats_path, "%s-D.png"%i))

        # fig, ax = plt.subplots(4, 1, figsize=(16,8))
        fig, ax = plt.subplots(4, 1, figsize=(16,int(hparams.num_mels/40*6.5)))

        fig.suptitle(i)

        # plt.figure(figsize=(128, 128))
        # fig.set_size_inches(12,12)

        ax[0].matshow(R)
        ax[0].set_title("Right")
        ax[1].matshow(L)
        ax[1].set_title("Left")
        ax[2].matshow(A)
        ax[2].set_title("Average")
        ax[3].matshow(D)
        ax[3].set_title("Difference")

        # plt.tight_layout()
        plt_path=os.path.join(plots_path, "%s.png"%i)
        
        fig.savefig(plt_path, format="png")

if __name__ == "__main__":
    args = docopt(__doc__)
    wavefiles_path = args["<wavefiles_path>"]
    feats_path = args["<feats_path>"]
    num_workers = args["--num_workers"]
    if num_workers is None: num_workers = 1
    preset = args["--preset"]

    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    
    print(hparams_debug_string())

    print("The feature file will be located in {}".format(feats_path))
    time_tag = time.time()

    if not hparams.do_hpss:
        preProcess = LogMelPreProcess(wavefiles_path, feats_path)
    else:
        print("HPSS")
        preProcess = HPSSPreProcess(wavefiles_path, feats_path)

    preProcess.process(int(num_workers))
    #PlotToCheck(feats_path, 20)
    print("Done. {}".format(time.time() - time_tag))

