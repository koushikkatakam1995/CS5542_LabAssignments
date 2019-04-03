# Disclaimer: This program is inspired directly by the codes that were given in
# CS5542 (UMKC Class) By Ms. Mayanka Shekar, and on it we're trying to improve and develop!


#required libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
import nltk.translate.gleu_score as gleu
import numpy
import os
try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')
import nltk

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu

import logging
import math

import tensorflow as tf

from medium_show_and_tell_caption_generator.caption_generator import CaptionGenerator
from medium_show_and_tell_caption_generator.model import ShowAndTellModel
from medium_show_and_tell_caption_generator.vocabulary import Vocabulary

FLAGS = tf.flags.FLAGS
# setting input files and model location
tf.flags.DEFINE_string("model_path", "C:/Users/kiree/Desktop/UMKC/2nd Semester/Big Data Analytics and Applications/ICP/ICP6/Tutorial 6 Source Code/medium-show-and-tell-caption-generator-master/model/show-and-tell.pb", "Model graph def path")
tf.flags.DEFINE_string("vocab_file", "C:/Users/kiree/Desktop/UMKC/2nd Semester/Big Data Analytics and Applications/ICP/ICP6/Tutorial 6 Source Code/medium-show-and-tell-caption-generator-master/etc/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "C:/Users/kiree/Desktop/UMKC/2nd Semester/Big Data Analytics and Applications/ICP/ICP6/Tutorial 6 Source Code/medium-show-and-tell-caption-generator-master/imgs/457774327_d3c798c162.jpg",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def main(_):
    model = ShowAndTellModel(FLAGS.model_path)
    vocab = Vocabulary(FLAGS.vocab_file)
    filenames = _load_filenames()
    can1 = "a table with different kinds of food"
    candidate=can1.split()
    generator = CaptionGenerator(model, vocab)
    for filename in filenames:
        with tf.gfile.GFile(filename, "rb") as f:
            image = f.read()
        captions = generator.beam_search(image)
        print("Captions: ")
        for i, caption in enumerate(captions):
            sentence = [vocab.id_to_token(w) for w in caption.sentence[1:-1]]
            sentence = " ".join(sentence)
            temp = "  %d) %s (p=%f)" % (i+1, sentence, math.exp(caption.logprob))
            print(temp)
            comp = [sentence.split()]
            # Calculating The Blue Score
            print('Blue cumulative 1-gram: %f' % sentence_bleu(comp, candidate, weights=(1, 0, 0, 0)))
            print('Blue cumulative 2-gram: %f' % sentence_bleu(comp, candidate, weights=(0.5, 0.5, 0, 0)))
            # Glue Score
            G = gleu.sentence_gleu(comp, candidate, min_len=1, max_len=2)
            print("Glue score for this sentence: {}".format(G))



def _load_filenames():
    filenames = []
    for file_pattern in FLAGS.input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    logger.info("Running caption generation on %d files matching %s",
                len(filenames), FLAGS.input_files)
    return filenames


if __name__ == "__main__":
    tf.app.run()