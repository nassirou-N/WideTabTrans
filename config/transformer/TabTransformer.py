import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention

class TabTransformer(Layer):
    def __init__(self, 
                 num_heads=8, 
                 key_dim=64, 
                 ff_dim=256, 
                 num_layers=4, 
                 dropout_rate=0.1, 
                 **kwargs):
        super(TabTransformer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        self.feature_dim = input_shape[-1]
        
        # Couche de projection initiale
        self.projection = Dense(
            self.key_dim * self.num_heads, 
            activation='gelu',
            kernel_initializer='he_normal'
        )
        
        # Blocs transformer
        self.transformer_blocks = []
        for i in range(self.num_layers):
            block = {
                'mha': MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.key_dim,
                    dropout=self.dropout_rate
                ),
                'ffn1': Dense(self.ff_dim, activation='gelu'),
                'ffn2': Dense(self.key_dim * self.num_heads),
                'layernorm1': LayerNormalization(epsilon=1e-6),
                'layernorm2': LayerNormalization(epsilon=1e-6),
                'dropout1': Dropout(self.dropout_rate),
                'dropout2': Dropout(self.dropout_rate)
            }
            self.transformer_blocks.append(block)
        
        # Projection finale
        self.final_projection = Dense(
            self.key_dim, 
            activation='gelu',
            kernel_initializer='he_normal'
        )
        
        super(TabTransformer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Projection initiale
        x = self.projection(inputs)
        
        # Application des blocs transformer
        for block in self.transformer_blocks:
            # Multi-head attention
            attn_output = block['mha'](x, x, training=training)
            attn_output = block['dropout1'](attn_output, training=training)
            x1 = block['layernorm1'](x + attn_output)
            
            # Feed-forward network
            ffn_output = block['ffn1'](x1)
            ffn_output = block['dropout2'](ffn_output, training=training)
            ffn_output = block['ffn2'](ffn_output)
            
            x = block['layernorm2'](x1 + ffn_output)
        
        # Projection finale
        return self.final_projection(x)
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.key_dim,)
    
    def get_config(self):
        config = super(TabTransformer, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ff_dim": self.ff_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
        })
        return config