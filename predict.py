'''
read image and Keras model to predict class with propability
top_k
load JSON mapping class value to categorical names
'''

import argparse
import numpy as np
import json
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

def p_predict(image_path, model, top_k, category_map):

    image = Image.open(image_path) #get the image
    image = np.asarray(image) #convert to np array
    image = process_image(image)
    image = np.expand_dims(image, axis = 0) #add axis

    probpredictions = model.predict(image)
    values, indices= tf.math.top_k(probpredictions, k=top_k, sorted=True, name=None) #give value and indices of top_k larges
    probs=values[0].numpy().tolist() #convert to list
    classes=indices[0].numpy().tolist()
    
    class_print = []
    for i in classes:
        class_print.append(class_names[str(i+1)])
        
    return probs, class_print

def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)
    return model

def process_image(image):
    image = tf.cast(image, tf.float32)
    #image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224,224))
    image /= 255
    image = image.numpy()
    return image
    
def class_names(category_map):
    with open(category_map, 'r') as f:
        class_names = json.load(f) #import class names as dict
    return class_names

parser = argparse.ArgumentParser(description = 'Gives the top categories of flowers given a flower image')
#parser = argparse(definition = 'Gives the top categories of flowers given a flower image')
parser.add_argument('-i','--image_path', help='Path to image for prediction', default='./test_images/wild_pansy.jpg')
parser.add_argument('-m','--model_path', help='Path to a Model in *.H5', default='./savedmodel.h5')
parser.add_argument('-k', '--top_k', type=int, help='Number of categories (1-5) to take into account', default=5)
parser.add_argument('-c', '--category_map', help='Path to a JSON file mapping number to class names', default='./label_map.json')

args = parser.parse_args()

'''
#preventing bad input
if args.top_k <= 0:
    top_k = 1
    print('You select less than 1 category. The most likely will be displayed')
elif args.top_k > 5:
    top_k = 5
    print('More than 5 categories is not allowed.')
else:
    top_k = args.top_k'''

if __name__ == "__main__":
    probs, class_print=p_predict(args.image_path, args.model_path, args.top_k, args.category_map)

#print('The Image :', args.image_path)
#print('The propability: ',probs)
#print('The classes :', class_print)