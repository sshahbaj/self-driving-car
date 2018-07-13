from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from constants import IMAGE_HEIGHT, IMAGE_WIDTH
from PIL import Image


CLASSIFICATION_LABELS = ['forward', 'idle', 'left', 'right']

class Predictor:
    def __init__(self):
        '''Load the architecture'''
        self.json_file = open('model.json', 'r')
        self.loaded_model_json = self.json_file.read()
        self.json_file.close()
        self.model = model_from_json(self.loaded_model_json)

        '''Load the trained weights'''
        self.model.load_weights("first_try.pkl")
    
    def predict(self, stream):
        img = self.convert_stream_to_array(stream)
        classes = self.model.predict_classes(img)
        return CLASSIFICATION_LABELS[classes[0]]


    def convert_stream_to_array(self, stream):
        #stream.seek(0)
        img = Image.open(stream).convert('L')
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img = img_to_array(img)
        img /= 255
        img = img.reshape((1,) + img.shape)
        return img
