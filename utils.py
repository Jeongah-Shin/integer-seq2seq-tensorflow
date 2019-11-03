import tensorflow as tf
import os, argparse
from common import get_checkpoint

CHECKPOINT_DIR = './pretrained_weights/t1_savedModel'
OUTPUT_NODE_NAME = 'dense'
OUTPUT_NAME = './frozen_models/frozen.pb'
OUTPUT_PATH = './'

def dynamic_to_fixed_point(model_path) :
    model = tf.keras.models.load_model(model_path)
    input_1 = tf.keras.layers.InputLayer(input_shape=(1, 301, 11), name="input_1")
    input_2 = tf.keras.layers.InputLayer(input_shape=(1, 197, 13), name="input_2")
    model.layers["input_1"] = input_1
    model.layers["input_2"] = input_2
    model.save("fixed_point.h5")
    return model

def ckpt_to_frozen(checkpoint_dir, output_node_name, output_name, output_path):
    if not os.path.exists(checkpoint_dir):
        print("%s does not exist.  Exiting." % checkpoint_dir)
        exit()

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    # absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    # output_graph = absolute_model_dir + "/frozen_model.pb"
    output_graph = os.path.join(output_path, output_name)

    # to allow tensorflow to control on which device it will load operation
    clear_devices = True
    tf.reset_default_graph()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        saver = tf.train.Saver()

        resumed_checkpoint, last_step = get_checkpoint(checkpoint_dir)
        saver.restore(sess, resumed_checkpoint)

        for elem in tf.get_default_graph().as_graph_def().node:
            print (elem.name)
        print ("==================================")
        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
            output_node_name.split(",")
        )

        output_graph_def = tf.graph_util.remove_training_nodes(output_graph_def, protected_nodes='output')

        for elem in output_graph_def.node:
            print (elem.name)
        print ("==================================")

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

if __name__ == '__main__':
    ckpt_to_frozen(CHECKPOINT_DIR, OUTPUT_NODE_NAME, OUTPUT_NAME, OUTPUT_PATH)