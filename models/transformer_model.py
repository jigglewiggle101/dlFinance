# Transformer model implementation

import tensorflow as tf
from tensorflow.keras import layers

def create_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=64)(inputs, inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1)(x)
    
    model = tf.keras.models.Model(inputs, x)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model
