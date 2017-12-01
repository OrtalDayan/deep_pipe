import tensorflow as tf
import numpy as np
from os.path import join as jp


def get_tensor_vals(model, names):
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(jp(model, "model.meta"))
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model))
        output = []
        for name in names:
            tensor = tf.get_default_graph().get_tensor_by_name(name)
            output.append(sess.run(tensor))
    return output


def describe_nds(nds):
    for nd in nds:
        print("-----")
        print(type(nd))
        print(nd.shape)


def broadcast_nd(nd):
    return np.concatenate([nd] * 4, axis=3)


def broadcast_nds(tensors):
    return [broadcast_nd(tensor) for tensor in tensors]


def get_variables_to_restore(parent_scope, scopes_to_restore, vars_to_omit, debug_print):
    variables = []
    for scope in scopes_to_restore:
        variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="{}/{}".format(parent_scope, scope))
    if debug_print:
        print("# of variables to restore before filtering = {}".format(len(variables)))
    variables = [variable for variable in variables if variable.name not in vars_to_omit]
    if debug_print:
        print("# of variables to restore before filtering = {}".format(len(variables)))
    return variables


def get_name_to_variable_mapping():
    # [gleb] this function is needed because of the incorrect way Saver.restore and get_variable work with each other
    # https://github.com/tensorflow/tensorflow/issues/1325
    mapping = {}
    for variable in tf.trainable_variables():
        mapping[variable.name] = variable
    return mapping


def surgery(shaping_model, restoring_model, debug_print=True):
    tf.reset_default_graph()

    var_names = ['deep_medic/detailed/cba_0_a/cb/conv3d/kernel:0', 'deep_medic/context/cba_0_a/cb/conv3d/kernel:0']
    tensor_vals = get_tensor_vals(restoring_model, var_names)
    if debug_print:
        describe_nds(tensor_vals)
    tensor_vals = broadcast_nds(tensor_vals)
    if debug_print:
        describe_nds(tensor_vals)

    tf.reset_default_graph()
    tf.train.import_meta_graph(jp(shaping_model, "model.meta"))

    vars_to_restore = get_variables_to_restore('deep_medic', ['detailed', 'context', 'upsample', 'comm_1', 'comm_2'],
                                               var_names, debug_print)
    transfer_saver = tf.train.Saver(vars_to_restore)
    with tf.Session() as sess:
        transfer_saver.restore(sess, jp(restoring_model, "model"))
        name_to_variable = get_name_to_variable_mapping()

        for name, new_val in zip(var_names, tensor_vals):
            if debug_print:
                print("restoring {}".format(name))
            variable = name_to_variable[name]
            assign_op = variable.assign(new_val)
            sess.run(assign_op)
