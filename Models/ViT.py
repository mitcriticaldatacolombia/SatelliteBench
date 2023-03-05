from tensorflow.keras.layers import Input, Permute
from tensorflow.keras.models import Model

from transformers import TFViTModel

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