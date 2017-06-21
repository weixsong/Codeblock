from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.contrib.rnn import *
from tensorflow.contrib.seq2seq.python.ops \
        import attention_wrapper as wrapper, helper, basic_decoder, decoder
import models.ops as ops
import sys
import audio

class Config(object):
    max_decode_iter = 350 // audio.r
    attention_units = 256
    decoder_units = 256
    mel_features = 80
    embed_dim = 256
    fft_size = 1025
    dropout_prob = 0.5

    scheduled_sample = 0

    cap_grads = 5

    init_lr = 0.0005
    annealing_rate = 1

    batch_size = 32

class Tacotron(object):
    # transformation applied to input character sequence
    # and each input frame to the decoder
    def pre_net(self, inputs, units=[256,128], train=True):
        with tf.variable_scope('pre_net'):
            layer_1 = tf.layers.dense(inputs, units[0], activation=tf.nn.relu)
            layer_1 = tf.layers.dropout(layer_1, rate=self.config.dropout_prob, training=train)
            layer_2 = tf.layers.dense(layer_1, units[1], activation=tf.nn.relu)
            layer_2 = tf.layers.dropout(layer_2, rate=self.config.dropout_prob, training=train)
        return layer_2

    def create_decoder(self, encoded, inputs, train=True):
        config = self.config
        attention_mech = wrapper.BahdanauAttention(
                config.attention_units,
                encoded,
                memory_sequence_length=inputs['text_length']
        )

        inner_cell = [GRUCell(config.decoder_units) for _ in range(3)]

        decoder_cell = OutputProjectionWrapper(
                InputProjectionWrapper(
                    ResidualWrapper(
                        MultiRNNCell(inner_cell)
                ), config.decoder_units)
        , config.mel_features * config.r)

        # feed in rth frame at each time step
        decoder_frame_input = \
            lambda inputs, attention: tf.concat(
                    [self.pre_net(tf.slice(inputs,
                        [0, (config.r - 1)*config.mel_features], [-1, -1]), train=train),
                    attention]
                , -1)

        cell = wrapper.AttentionWrapper(
                decoder_cell,
                attention_mech,
                attention_layer_size=config.attention_units,
                cell_input_fn=decoder_frame_input,
                alignment_history=True,
                output_attention=False
        )

        if train:
            if config.scheduled_sample:
                decoder_helper = helper.ScheduledOutputTrainingHelper(
                        inputs['mel'], inputs['speech_length'], config.scheduled_sample)
            else:
                decoder_helper = helper.TrainingHelper(inputs['mel'], inputs['speech_length'])
        else:
            decoder_helper = ops.InferenceHelper(
                    tf.shape(inputs['text'])[0],
                    config.mel_features * config.r
            )

        dec = basic_decoder.BasicDecoder(
                cell,
                decoder_helper,
                cell.zero_state(dtype=tf.float32, batch_size=tf.shape(inputs['text'])[0])
        )

        return dec

    def inference(self, inputs, train=True):
        config = self.config

        # extract character representations from embedding
        with tf.variable_scope('embedding', initializer=tf.contrib.layers.xavier_initializer()):
            embedding = tf.get_variable('embedding',
                    shape=(config.vocab_size, config.embed_dim), dtype=tf.float32)
            embedded_inputs = tf.nn.embedding_lookup(embedding, inputs['text'])

        # process text input with CBHG module 
        with tf.variable_scope('encoder'):
            pre_out = self.pre_net(embedded_inputs, train=train)
            tf.summary.histogram('pre_net_out', pre_out)

            encoded = ops.CBHG(pre_out, K=16, c=[128,128,128], gru_units=128)

        # pass through attention based decoder
        with tf.variable_scope('decoder'):
            dec = self.create_decoder(encoded, inputs, train)
            (seq2seq_output, _), attention_state, _ = \
                    decoder.dynamic_decode(dec, maximum_iterations=config.max_decode_iter)
            self.alignments = tf.transpose(attention_state.alignment_history.stack(), [1,0,2])
            tf.summary.histogram('seq2seq_output', seq2seq_output)

        # use second CBHG module to process mel features into linear spectogram
        with tf.variable_scope('post-process'):
            # reshape to account for r value
            post_input = tf.reshape(seq2seq_output, 
                    (tf.shape(seq2seq_output)[0], -1, config.mel_features))

            output = ops.CBHG(post_input, K=8, c=[128,256,80], gru_units=128)
            output = tf.layers.dense(output, units=config.fft_size)

            # reshape back to r frame representation
            output = tf.reshape(output, (tf.shape(output)[0], -1, config.fft_size*config.r))
            tf.summary.histogram('output', output)

        return seq2seq_output, output

    def add_loss_op(self, seq2seq_output, output, mel, linear):
        # total loss is equal weighting of seq2seq and output losses
        seq2seq_loss = tf.reduce_sum(tf.abs(seq2seq_output - mel))
        output_loss = tf.reduce_sum(tf.abs(output - linear))
        loss = seq2seq_loss + output_loss

        tf.summary.scalar('seq2seq loss', seq2seq_loss)
        tf.summary.scalar('output loss', output_loss)
        tf.summary.scalar('loss', loss)
        return loss

    def add_train_op(self, loss):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        opt = tf.train.AdamOptimizer(learning_rate=self.lr)

        gradients, variables = zip(*opt.compute_gradients(loss))
        # save selected gradient summaries
        #for grad in gradients:
            #if 'BasicDecoder' in grad.name or 'gru_cell' in grad.name or 'highway_3' in grad.name:
                #tf.summary.scalar(grad.name, tf.reduce_sum(grad))

        # optionally cap and noise gradients to regularize
        if self.config.cap_grads > 0:
            with tf.variable_scope('cap_grads'):
                tf.summary.scalar('global_gradient_norm', tf.global_norm(gradients))
                gradients, _ = tf.clip_by_global_norm(gradients, self.config.cap_grads)

        train_op = opt.apply_gradients(zip(gradients, variables), global_step=self.global_step)
        return train_op

    def __init__(self, config, inputs, train=True):
        self.config = config
        self.lr = tf.placeholder(tf.float32)
        self.seq2seq_output, self.output = self.inference(inputs, train)
        if train:
            self.loss = self.add_loss_op(self.seq2seq_output, self.output, inputs['mel'], inputs['stft'])
            self.train_op = self.add_train_op(self.loss)
        self.merged = tf.summary.merge_all()

