import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from transform_nets import input_transform_net1


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)
    end_points = {}
    k = 20

    adj_matrix = tf_util.pairwise_distance(point_cloud[:, :, 6:])
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net1(edge_feature, is_training, bn_decay, K=3)

    point_cloud_transformed = tf.matmul(point_cloud, transform)

    adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)
    edge_out1 = tf_util.conv2d(edge_feature, 128, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=True, is_training=is_training,
                               scope='dgcnn1', bn_decay=bn_decay)
    edge_out22 = edge_out1 - tf.nn.softmax(edge_out1, axis=-2) * edge_out1
    edge_out2 = tf_util.conv2d(edge_out22, 128, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=True, is_training=is_training,
                                scope='dgcnn2', bn_decay=bn_decay)

    attention_feature_2 = edge_out1 + edge_out2

    net_1 = tf.reduce_sum(attention_feature_2, axis=-2, keep_dims=True)

    net1 = net_1

    adj_matrix = tf_util.pairwise_distance(net_1)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)
    edge_out3 = tf_util.conv2d(edge_feature, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn3', bn_decay=bn_decay)

    edge_out33 = edge_out3 - tf.nn.softmax(edge_out3, axis=-2) * edge_out3
    edge_out4 = tf_util.conv2d(edge_out33, 128, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=True, is_training=is_training,
                                scope='dgcnn4', bn_decay=bn_decay)

    attention_feature_4 = edge_out3 + edge_out4

    net_2 = tf.reduce_sum(attention_feature_4, axis=-2, keep_dims=True)
    net2 = net_2

    adj_matrix = tf_util.pairwise_distance(net2)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net2, nn_idx=nn_idx, k=k)

    edge_out5 = tf_util.conv2d(edge_feature, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn5', bn_decay=bn_decay)
    edge_out55 = edge_out5 - tf.nn.softmax(edge_out5, axis=-2) * edge_out5

    edge_out6 = tf_util.conv2d(edge_out55, 128, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=True, is_training=is_training,
                                scope='dgcnn6', bn_decay=bn_decay)

    attention_feature_6 = edge_out5 + edge_out6

    net_3 = tf.reduce_sum(attention_feature_6, axis=-2, keep_dims=True)
    net3 = net_3

    net = tf_util.conv2d(tf.concat([net1, net2, net3], axis=-1), 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='agg', bn_decay=bn_decay)

    net = tf.reduce_max(net, axis=1, keep_dims=True)

    # MLP on global point cloud vector
    out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

    out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')

    expand = tf.tile(out_max, [1, num_point, 1, 1])

    concat = tf.concat(axis=3, values=[expand,
                                       net_1,
                                       net_2,
                                       net_3])

    # CONV
    net = tf.reshape(concat, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training,
                                  scope='fc3', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp3')
    net = tf_util.fully_connected(net, 13, activation_fn=None, scope='fc4')
    return net, end_points


def get_loss(pred, label):
  """ pred: B*NUM_CLASSES,
      label: B, """
  labels = tf.one_hot(indices=label, depth=13)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
  classify_loss = tf.reduce_mean(loss)
  return classify_loss

def count_flops(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};   Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

if __name__=='__main__':
  batch_size = 2
  num_pt = 124
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed>=0.5] = 1
  label_feed[label_feed<0.5] = 0
  label_feed = label_feed.astype(np.int32)

  # # np.save('./debug/input_feed.npy', input_feed)
  # input_feed = np.load('./debug/input_feed.npy')
  # print input_feed

  with tf.Graph().as_default() as graph:
    input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
    pos, ftr = get_model(input_pl, tf.constant(True))
    # loss = get_loss(logits, label_pl, None)
    count_flops(graph)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {input_pl: input_feed, label_pl: label_feed}
      res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
      print(res1.shape)
      print(res1)

      print(res2.shape)
      print(res2)













