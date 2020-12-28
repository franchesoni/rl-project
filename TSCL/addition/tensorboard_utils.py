import tensorflow as tf


def create_summary_writer(logdir):
    return tf.summary.create_file_writer(logdir)


def create_summary(tag, value):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])


def add_summary(writer, tag, value, step):
    tf.summary.write(tag, value, step=step)


def add_summaries(writer, tags, values, step, prefix=''):
    for (t, v) in zip(tags, values):
        s = create_summary(prefix + t, v)
        tf.summary.write(prefix + t, v, step=step)
