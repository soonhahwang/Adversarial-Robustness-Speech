import tensorflow as tf
import math

def create_ds_cnn_model(model_settings, model_size_info):
    label_count = model_settings['label_count']
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']

    t_dim = input_time_size
    f_dim = input_frequency_size

    # Extract model dimensions from model_size_info.
    num_layers = model_size_info[0]
    conv_feat = [None]*num_layers
    conv_kt = [None]*num_layers
    conv_kf = [None]*num_layers
    conv_st = [None]*num_layers
    conv_sf = [None]*num_layers

    i = 1
    for layer_no in range(0, num_layers):
        conv_feat[layer_no] = model_size_info[i]
        i += 1
        conv_kt[layer_no] = model_size_info[i]
        i += 1
        conv_kf[layer_no] = model_size_info[i]
        i += 1
        conv_st[layer_no] = model_size_info[i]
        i += 1
        conv_sf[layer_no] = model_size_info[i]
        i += 1

    inputs = tf.keras.Input(shape=(model_settings['fingerprint_size']), name='input')

    # Reshape the flattened input.
    x = tf.reshape(inputs, shape=(-1, input_time_size, input_frequency_size, 1))

    # Depthwise separable convolutions.
    for layer_no in range(0, num_layers):
        if layer_no == 0:
            # First convolution.
            x = tf.keras.layers.Conv2D(filters=conv_feat[0],
                                       kernel_size=(conv_kt[0], conv_kf[0]),
                                       strides=(conv_st[0], conv_sf[0]),
                                       padding='SAME')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        else:
            # Depthwise convolution.
            x = tf.keras.layers.DepthwiseConv2D(kernel_size=(conv_kt[layer_no], conv_kf[layer_no]),
                                                strides=(conv_sf[layer_no], conv_st[layer_no]),
                                                padding='SAME')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

            # Pointwise convolution.
            x = tf.keras.layers.Conv2D(filters=conv_feat[layer_no], kernel_size=(1, 1))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        t_dim = math.ceil(t_dim/float(conv_st[layer_no]))
        f_dim = math.ceil(f_dim/float(conv_sf[layer_no]))

    # Global average pool.
    x = tf.keras.layers.AveragePooling2D(pool_size=(t_dim, f_dim), strides=1)(x)

    # Squeeze before passing to output fully connected layer.
    x = tf.reshape(x, shape=(-1, conv_feat[layer_no]))

    # Output connected layer.
    output = tf.keras.layers.Dense(units=label_count, activation='softmax')(x)

    return tf.keras.Model(inputs, output)