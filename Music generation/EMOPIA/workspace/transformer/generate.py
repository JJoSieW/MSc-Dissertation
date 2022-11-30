from midiSynth.synth import MidiSynth
midi_synth = MidiSynth()


import os
import pickle
import torch


from utils import write_midi
from models import TransformerModel, network_paras

path_dictionary = '../../dataset/co-representation/dictionary.pkl'
assert os.path.exists(path_dictionary)


dictionary = pickle.load(open(path_dictionary, 'rb'))
event2word, word2event = dictionary


# config
n_class = []   # num of classes for each token
for key in event2word.keys():
    n_class.append(len(dictionary[0][key]))
n_token = len(n_class)


path_saved_ckpt = 'exp/pretrained_transformer/loss_25_params.pt'
assert os.path.exists(path_saved_ckpt)

# init model
net = TransformerModel(n_class, is_training=False)
net.cuda()
net.eval()

net.load_state_dict(torch.load(path_saved_ckpt))



emotion_tag = 3  # the target emotion class you want. It should belongs to [1,2,3,4].
path_outfile = 'q301' # output midi file name

res, _ = net.inference_from_scratch(dictionary, emotion_tag, n_token=8, display=False)
write_midi(res, path_outfile + '.mid', word2event)

#midi_synth.play_midi(path_outfile + '.mid')
#midi_synth.midi2audio(path_outfile + '.mid', path_outfile + '.mp3')