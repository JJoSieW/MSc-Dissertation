
import numpy as np
import os, sys
from os.path import join
import argparse
import pickle, gzip, h5py
import tensorflow as tf
#from .vocab import IDX_EOS, IDX_BOS, IDX_PAD


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def list_all_files(root_dir, file_ext):
    '''Lists all files in a folder and its subfolders that match file_ext
    Args:
        root_dir: top directory
        file_ext: file extensions, e.g. ('.mid','.wav') or '.mid'
    Returns:
        A list of files in root_dir
    '''
    return [join(root, name) for root, dirs, files in os.walk(root_dir) 
        for name in files if name.endswith(file_ext)]


def top_k_sampling(logits, top_k=25, temperature=0.8):
    'k must be greater than 0'

    values, _ = tf.nn.top_k(logits, k=top_k)
    min_value = tf.reduce_min(values)
    logits = tf.where(
        logits < min_value,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits)

    logits = logits / temperature

    sample = tf.random.categorical(tf.expand_dims(logits, 0), num_samples=1, dtype=tf.int32)
    return tf.reduce_sum(sample)



def top_k_top_p_sampling(logits, top_k=0, top_p=0.0, temperature=0.8):
    ''' Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k sampling).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus sampling).
    '''
    filter_value = -float('Inf')
    logits = logits / temperature
    top_k = min(top_k, logits.shape[-1])  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        logits = tf.where(
            logits < tf.nn.top_k(logits, top_k)[0][-1],
            tf.ones_like(logits, dtype=logits.dtype) * filter_value,
            logits)

    if top_p > 0.0:
        sorted_logits = tf.sort(logits, direction='DESCENDING')
        sorted_indices = tf.argsort(logits, direction='DESCENDING')
        cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits))

        # Remove tokens with cumulative probability above the threshold
        t_sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        indices = tf.range(1, tf.shape(logits)[0], 1)
        sorted_indices_to_remove = tf.scatter_nd(tf.expand_dims(indices, 1), t_sorted_indices_to_remove[:-1], logits.shape)
        indices_to_remove = tf.boolean_mask(sorted_indices, sorted_indices_to_remove)
        t = tf.ones(tf.shape(indices_to_remove)[0], dtype=tf.bool)
        to_remove = tf.scatter_nd(tf.expand_dims(indices_to_remove, 1), t, logits.shape)
        logits = tf.where(
            to_remove,
            tf.ones_like(logits, dtype=logits.dtype) * filter_value,
            logits
        )
    
    #sample = tf.random.categorical(tf.expand_dims(logits, 0), num_samples=1, dtype=tf.int32)
    sample = tf.random.categorical(tf.expand_dims(logits, 0), num_samples=1)[0,0].numpy()
    return sample #tf.reduce_sum(sample)



def nucleus_sampling(logits, p=0.9):
    sorted_logits = tf.sort(logits, direction='DESCENDING')
    sorted_indices = tf.argsort(logits, direction='DESCENDING')
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits))
    t_sorted_indices_to_remove = cumulative_probs > p
    ''' Shift the indices to the right to keep also the first token above the threshold '''
    indices = tf.range(1, tf.shape(logits)[0], 1)
    sorted_indices_to_remove = tf.scatter_nd(tf.expand_dims(indices, 1), t_sorted_indices_to_remove[:-1], logits.shape)
    indices_to_remove = tf.boolean_mask(sorted_indices, sorted_indices_to_remove)
    t = tf.ones(tf.shape(indices_to_remove)[0], dtype=tf.bool)
    to_remove = tf.scatter_nd(tf.expand_dims(indices_to_remove, 1), t, logits.shape)
    logits = tf.where(
        to_remove,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits
    )

    sample = tf.random.categorical(tf.expand_dims(logits, 0), num_samples=1, dtype=tf.int32)
    return tf.reduce_sum(sample)


def argmax(logits):
    return tf.argmax(logits)


def sampling(logits, temperature=0.8):
    logits = logits / temperature
    sample = tf.random.categorical(tf.expand_dims(logits, 0), num_samples=1, dtype=tf.int32)
    return tf.reduce_sum(sample)


def dict2params(d, f=','):
    return f.join(f'{k}={v}' for k, v in d.items())


def params2dict(p, f=',', e='='):
    d = {}
    for item in p.split(f):
        item = item.split(e)
        if len(item) < 2:
            continue
        k, *v = item
        d[k] = eval('='.join(v))
    return d

