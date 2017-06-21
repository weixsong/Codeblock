"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""

from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.client import timeline

from wavenet import WaveNetModel, AudioReader, optimizer_factory

BATCH_SIZE = 1
GPU_NUMS = 1
DATA_DIRECTORY = './VCTK-Corpus'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 50
NUM_STEPS = int(10)
LEARNING_RATE = 1e-3
WAVENET_PARAMS = './wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.3
EPSILON = 0.001
MOMENTUM = 0.9


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]


    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once.')
    parser.add_argument('--gpu_nums', type=int, default=GPU_NUMS,
                        help='How many GPU to use to train the model.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--store_metadata', type=bool, default=False,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut audio samples to this many '
                        'samples.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                        'Disabled by default')
    parser.add_argument('--silence_threshold', type=float,
                        default=SILENCE_THRESHOLD,
                        help='Volume threshold below which to trim the start '
                        'and the end from the training set samples.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                        'used by sgd or rmsprop optimizer. Ignored by the '
                        'adam optimizer.')
    parser.add_argument('--histograms', type=_str_to_bool, default=False,
                         help='Whether to store histogram summaries.')
    parser.add_argument('--gc_channels', type=int, default=None,
                        help='Number of global condition channels.')
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }


def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), (grad0_gpu1, var0_gpu1)... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is None:
                continue

            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        if len(grads) == 0:
            average_grads.append((None, grad_and_vars[0][1]))
            continue

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(0, grad)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def main():
    args = get_arguments()

    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    logdir_root = directories['logdir_root']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    with tf.device("/cpu:0"):
        # Create coordinator.
        coord = tf.train.Coordinator()

        # Load raw waveform from VCTK corpus.
        with tf.name_scope('create_inputs'):
            # Allow silence trimming to be skipped by specifying a threshold near
            # zero.
            silence_threshold = args.silence_threshold if args.silence_threshold > \
                                                          EPSILON else None
            gc_enabled = args.gc_channels is not None
            reader = AudioReader(
                args.data_dir,
                coord,
                sample_rate=wavenet_params['sample_rate'],
                gc_enabled=gc_enabled,
                sample_size=args.sample_size,
                silence_threshold=silence_threshold)

        # Create network.
        net = WaveNetModel(
            batch_size=args.batch_size,
            dilations=wavenet_params["dilations"],
            filter_width=wavenet_params["filter_width"],
            residual_channels=wavenet_params["residual_channels"],
            dilation_channels=wavenet_params["dilation_channels"],
            skip_channels=wavenet_params["skip_channels"],
            quantization_channels=wavenet_params["quantization_channels"],
            use_biases=wavenet_params["use_biases"],
            scalar_input=wavenet_params["scalar_input"],
            initial_filter_width=wavenet_params["initial_filter_width"],
            histograms=args.histograms,
            global_condition_channels=args.gc_channels,
            global_condition_cardinality=reader.gc_category_cardinality)

        if args.l2_regularization_strength == 0:
            args.l2_regularization_strength = None

        global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), trainable=False)

        optimizer = optimizer_factory[args.optimizer](
            learning_rate=args.learning_rate,
            momentum=args.momentum)

        tower_grads = []
        tower_losses = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.gpu_nums):
                with tf.device("/gpu:%d" % i), tf.name_scope("tower_%d" % i) as scope:
                    audio_batch = reader.dequeue(args.batch_size)
                    if gc_enabled:
                        gc_id_batch = reader.dequeue_gc(args.batch_size)
                    else:
                        gc_id_batch = None

                    loss = net.loss(input_batch=audio_batch,
                                    global_condition_batch=gc_id_batch,
                                    l2_regularization_strength=args.l2_regularization_strength)
                    tower_losses.append(loss)

                    trainable = tf.trainable_variables()
                    grads = optimizer.compute_gradients(loss, var_list=trainable)
                    tower_grads.append(grads)

                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    tf.get_variable_scope().reuse_variables()

        # calculate the mean of each gradient. Synchronization point across all towers
        grads = average_gradients(tower_grads)
        train_ops = optimizer.apply_gradients(grads, global_step=global_step)

        # calculate the mean loss
        loss = tf.reduce_mean(tower_losses)

        # Set up logging for TensorBoard.
        writer = tf.summary.FileWriter(logdir)
        writer.add_graph(tf.get_default_graph())
        run_metadata = tf.RunMetadata()
        summaries_ops = tf.summary.merge(summaries)

        # Set up session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
        init = tf.global_variables_initializer()
        sess.run(init)

        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=tf.trainable_variables())

        try:
            saved_global_step = load(saver, sess, restore_from)
            if is_overwritten_training or saved_global_step is None:
                # The first training step will be saved_global_step + 1,
                # therefore we put -1 here for new or overwritten trainings.
                saved_global_step = -1

        except:
            print("Something went wrong while restoring checkpoint. "
                  "We will terminate training to avoid accidentally overwriting "
                  "the previous model.")
            raise

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        reader.start_threads(sess)

        step = None
        try:
            last_saved_step = saved_global_step
            for step in range(saved_global_step + 1, args.num_steps):
                start_time = time.time()
                if args.store_metadata and step % 50 == 0:
                    # Slow run that stores extra information for debugging.
                    print('Storing metadata')
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    summary, loss_value, _ = sess.run(
                        [summaries_ops, loss, train_ops],
                        options=run_options,
                        run_metadata=run_metadata)
                    writer.add_summary(summary, step)
                    writer.add_run_metadata(run_metadata,
                                            'step_{:04d}'.format(step))
                    tl = timeline.Timeline(run_metadata.step_stats)
                    timeline_path = os.path.join(logdir, 'timeline.trace')
                    with open(timeline_path, 'w') as f:
                        f.write(tl.generate_chrome_trace_format(show_memory=True))
                else:
                    summary, loss_value, _ = sess.run([summaries_ops, loss, train_ops])
                    writer.add_summary(summary, step)

                duration = time.time() - start_time
                print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'
                      .format(step, loss_value, duration))

                if step % args.checkpoint_every == 0:
                    save(saver, sess, logdir, step)
                    last_saved_step = step

        except KeyboardInterrupt:
            # Introduce a line break after ^C is displayed so save message
            # is on its own line.
            print()
        finally:
            if step > last_saved_step:
                save(saver, sess, logdir, step)
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    main()
