import glob, time
from PIL import Image
from predict import Predictor

directions = ['forward', 'idle', 'left', 'right']

predict_from = 'data/test'

save_to = '/home/shahbaj/misclassified_data'

p = Predictor()

for direction in directions:
    for f in glob.glob(predict_from + '/' + direction + '/*.jpg'):
        command = p.predict(f)
        if command != direction:
            img = Image.open(f)
            img.save(save_to + '/' + direction + '/' + command + '-%s.jpg' % (str(time.time())))
            img.close()
            
