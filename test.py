from constants import *
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import model_from_json

test_datagen = ImageDataGenerator(
    rescale = 1./255,
)

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    color_mode='grayscale',
    classes=['forward', 'idle', 'left', 'right'],
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("first_try.pkl")

sgd = SGD(lr=LEARNING_RATE)

model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

score = model.evaluate_generator(test_generator, TEST_SIZE // BATCH_SIZE)

print score
