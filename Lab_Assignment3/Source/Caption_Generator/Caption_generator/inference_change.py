from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os

import tensorflow as tf

from medium_show_and_tell_caption_generator.caption_generator import CaptionGenerator
from medium_show_and_tell_caption_generator.model import ShowAndTellModel
from medium_show_and_tell_caption_generator.vocabulary import Vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model_path", "C:/Users/kiree/Desktop/UMKC/2nd Semester/Big Data Analytics and Applications/ICP/ICP6/Tutorial 6 Source Code/medium-show-and-tell-caption-generator-master/model/show-and-tell.pb", "Model graph def path")
tf.flags.DEFINE_string("vocab_file", "C:/Users/kiree/Desktop/UMKC/2nd Semester/Big Data Analytics and Applications/ICP/ICP6/Tutorial 6 Source Code/medium-show-and-tell-caption-generator-master/etc/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "C:/Users/kiree/Desktop/UMKC/2nd Semester/Big Data Analytics and Applications/ICP/ICP6/Tutorial 6 Source Code/medium-show-and-tell-caption-generator-master/imgs",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def main(_):
    model = ShowAndTellModel(FLAGS.model_path)
    vocab = Vocabulary(FLAGS.vocab_file)
    filenames = _load_filenames()

    generator = CaptionGenerator(model, vocab)
    with open('C:/Users/kiree/Desktop/UMKC/2nd Semester/Big Data Analytics and Applications/ICP/ICP6/Tutorial 6 Source Code/medium-show-and-tell-caption-generator-master/etc/predict.txt', 'a') as f1:
        for filename in filenames:
            with tf.gfile.GFile(filename, "rb") as f:
                image = f.read()
            captions = generator.beam_search(image)
            print("Captions for image %s:" % os.path.basename(filename))
            for i, caption in enumerate(captions):
                # Ignore begin and end tokens <S> and </S>.
                sentence = [vocab.id_to_token(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                if i == 1:
                    f1.write("%s \n" % sentence)
                    # print("this is---", sentence)
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))


def _load_filenames():
    filenames = []
    fn = []
    directory = FLAGS.input_files
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            pathname = os.path.join(directory, filename)
            filenames.append(pathname)
            # print(filenames)
            continue
        else:
            continue
    # for file_pattern in FLAGS.input_files.split(","):
    #     print(file_pattern)
    #     filenames.extend(tf.gfile.Glob(file_pattern))
    # logger.info("Running caption generation on %d files matching %s",
    #             len(filenames), FLAGS.input_files)
    # print(filenames)
    #filenames = sorted(filenames, key=lambda x: int(x.partition('imgs/')[2].partition('.')[0]))
    return filenames


if __name__ == "__main__":
    tf.app.run()
