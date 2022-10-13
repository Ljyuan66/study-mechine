from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np
from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
from keras.applications import VGG16


# 热力图最终没跑起来，电脑问题很大，平均一秒100000，要训练900000000次，需要花费时间过长，热力图本次先不进行求解


model = VGG16(weights='imagenet')
img_path = '/home/liang/Downloads/dogs-vs-cats/small_dataset/train/cats/cat_54'
img = image.load_img(img_path,target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
prads = model.predict(x)
print('predicted:', decode_predictions(prads,top=3)[0])
print(np.argmax(prads[0]))
cat_output = model.output[:, 386]
last_conv_layers = model.get_layer('block5_conv3')
grads = K.gradients(cat_output, last_conv_layers.output)[0]
pooled_grads = K.mean(grads,axis=(0,1,2))
iterate = K.function([model.input], [pooled_grads,last_conv_layers.output[0]])
pooled_grads_value, last_conv_layers_output_value = iterate(x)

for i in range(512):
    last_conv_layers_output_value[:,:,i] *= pooled_grads_value[i]

heatmap = np.mean(last_conv_layers_output_value,axis=-1)

heatmap = np.maximum(heatmap,0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)