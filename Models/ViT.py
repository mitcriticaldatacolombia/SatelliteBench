from tensorflow.keras.layers import Input, Permute
from tensorflow.keras.models import Model
import tensorflow as tf

from transformers import TFViTModel

class ViTLayer(tf.keras.layers.Layer):
    def __init__(self, backbone, **kwargs):
        super(ViTLayer, self).__init__(**kwargs)
        self.backbone = backbone
        
    def build(self, input_shape):
        self.vit = TFViTModel.from_pretrained(self.backbone)
        
    def call(self, inputs):
        out = self.vit(inputs)['pooler_output']
        return out
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.vit.config.hidden_size)
    

# Get ViT Model:
def get_vit_backbone(target_size, freeze=True):
        
    # Inputs:
    inputs = Input(target_size)
    
    # ViT uses chanel first, but our images has chanel last
    # So we have to chenge (224, 224, 3) -> (3, 224, 224)
    chanel_fist_inputs = Permute((3,1,2))(inputs)
    
    # ViT pretrained
    vit = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    
    # Freeze base model
    if freeze:
        vit.trainable = False
    
    embeddings = vit.vit(chanel_fist_inputs)[0][:,0,:]
    
    #embeddings = keras.layers.GlobalAveragePooling1D()(vit)
    
    vit_model = Model(inputs=inputs, outputs=embeddings)
    
    # Compile Model
    vit_model.compile()

    return vit_model

