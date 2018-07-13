from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from constants import *

model = Sequential()
model.add(Flatten(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNEL)))
model.add(Dense(100))
model.add(Activation('sigmoid'))
model.add(Dense(4))
model.add(Activation('sigmoid'))

sgd = SGD(lr=LEARNING_RATE)

model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)


train_datagen = ImageDataGenerator(
    rescale = 1./255,
#    rotation_range=30,
#    shear_range=0.2,
#    zoom_range=0.2
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    color_mode='grayscale',
    classes=['forward', 'idle', 'left', 'right'],
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

validation_generator = validation_datagen.flow_from_directory(
    'data/validation',
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    color_mode='grayscale',
    classes=['forward', 'idle', 'left', 'right'],
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

filepath = 'first_try.pkl'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(
    train_generator,
    steps_per_epoch=TRAIN_SIZE//BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks_list,
    validation_data=validation_generator,
    validation_steps=VALIDATION_SIZE//BATCH_SIZE,
    use_multiprocessing=True,
    workers=4
)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

#model.save_weights('first_try.pkl')
