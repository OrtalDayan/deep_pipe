import tensorflow as tf
import numpy as np
from os.path import join as jp


def get_tensor_vals(tensor_names, graph, sess):
    # http://bretahajek.com/2017/04/importing-multiple-tensorflow-models-graphs/ - nice article explaining how to
    # manage multiple graphs in tensorflow.
    output = []

    for name in tensor_names:
        tensor = graph.get_tensor_by_name(name)
        output.append(sess.run(tensor))

    return output


def get_tensor_vals_to_transfer(tensor_names, transfer_model):
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with graph.as_default():
        saver = tf.train.import_meta_graph(jp(transfer_model, "model.meta"))
        saver.restore(sess, tf.train.latest_checkpoint(transfer_model))
        return get_tensor_vals(tensor_names, graph, sess)


def import_graph_shape(model):
    tf.train.import_meta_graph(jp(model, "model.meta"))


def broadcast_nd(nd):
    return np.concatenate([nd] * 4, axis=3)


def broadcast_nds(tensors):
    return [broadcast_nd(tensor) for tensor in tensors]


def get_variables_to_transfer(vars_to_omit):
    parent_scope = 'deep_medic'
    scopes_to_transfer = ['detailed', 'context', 'upsample', 'comm_1', 'comm_2']
    variables = []

    for scope in scopes_to_transfer:
        variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="{}/{}".format(parent_scope, scope))

    filtered_variables = [variable for variable in variables if variable.name not in vars_to_omit]
    assert (len(filtered_variables) + len(vars_to_omit) == len(variables))

    return filtered_variables


def main_transfer(sess, vars_to_omit, transfer_model):
    vars_to_transfer = get_variables_to_transfer(vars_to_omit)
    transfer_saver = tf.train.Saver(vars_to_transfer)
    transfer_saver.restore(sess, jp(transfer_model, "model"))


def additional_transfer(sess, var_names_to_transfer, transfer_tensor_vals):
    assert (len(var_names_to_transfer) == len(transfer_tensor_vals))

    var_name_to_tf_var = get_name_to_variable_mapping()

    for name, new_val in zip(var_names_to_transfer, transfer_tensor_vals):
        variable = var_name_to_tf_var[name]
        assign_op = variable.assign(new_val)
        sess.run(assign_op)


def get_name_to_variable_mapping():
    # [gleb] this function is needed because of the incorrect way Saver.restore and get_variable work with each other
    # https://github.com/tensorflow/tensorflow/issues/1325
    mapping = {}
    for variable in tf.trainable_variables():
        mapping[variable.name] = variable
    return mapping


def surgery(sess, transfer_model):
    var_names = ['deep_medic/detailed/cba_0_a/cb/conv3d/kernel:0', 'deep_medic/context/cba_0_a/cb/conv3d/kernel:0']
    tensor_vals = get_tensor_vals_to_transfer(tensor_names=var_names, transfer_model=transfer_model)
    tensor_vals = broadcast_nds(tensor_vals)
    main_transfer(sess, var_names, transfer_model)
    additional_transfer(sess, var_names, tensor_vals)
