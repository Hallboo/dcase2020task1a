from asc.tfcompat.hparam import HParams
import numpy as np

# Default hyperparameters:
hparams = HParams(
    name="asc",
    # Extract Feature
    # feature_type="logmel",
    do_hpss=False,
    sample_rate=48000,
    two_channel=True,
    num_mels=40,
    n_fft=int(0.04*48000),
    # n_fft=2048,
    hop_length=int(0.02*48000),
    win_length=int(0.04*48000),
    deltas=True,

    # training testing evaluating
    model_type='Cnn_13layers_AvgPooling',
    # Cnn_5layers_AvgPooling,
    # Cnn_9layers_MaxPooling,
    # Cnn_9layers_AvgPooling,
    # Cnn_13layers_AvgPooling
    use_cuda=True,
    max_epoch=100,
    batch_size=128,
    
    labels=[
        'airport',
        'shopping_mall',
        'metro_station',
        'street_pedestrian',
        'public_square',
        'street_traffic',
        'tram',
        'bus',
        'metro',
        'park'
    ],

    devices = ['a','b','c','d','s1','s2','s3','s4','s5','s6']
)

def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
