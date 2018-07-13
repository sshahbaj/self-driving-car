import os, argparse, glob, shutil, numpy as np, time
from PIL import Image

main_directory = ['train', 'test', 'validation']
sub_directory = directions = ['forward', 'left', 'right', 'idle']

def main(path):

    '''Delete the reverse images'''
    for image in glob.glob(path + '/*' + 'reverse' + '*.jpg'):
        os.remove(image)

    
    '''Augument Data'''
    os.makedirs(path+'/temp')
    for direction in directions:
        image_list = glob.glob(path + '/*' + direction + '-*.jpg')
        for image in image_list:
            test_image = Image.open(image)
            rotated_image = test_image.transpose(Image.FLIP_LEFT_RIGHT)

            d = direction
            if d == 'right':
                d = 'left'
            elif d == 'left':
                d = 'right'
            
            saved_location = path + '/temp/' + 'image%s.jpg' % ("-" + d + "-"+ str(time.time()))    
            rotated_image.save(saved_location, format = 'JPEG')
            test_image.close()
            rotated_image.close()

    for f in glob.glob(path+'/temp/*.jpg'):
        shutil.move(f, path)
    os.rmdir(path+'/temp')

    
    '''Create 'train', 'validation' and 'test' folder along with the sub directories'''
    for main_dir in main_directory:

        if not os.path.exists(path+'/'+main_dir):
            os.makedirs(path+'/'+main_dir)
        
        for sub_dir in sub_directory:
            if not os.path.exists(path+'/'+main_dir+'/'+sub_dir):
                os.makedirs(path+'/'+main_dir+'/'+sub_dir)


    '''Segregate the images'''
    for direction in directions:
        names = glob.glob(path + '/*' + direction + '-*.jpg')
        length = len(names)
        lst = np.random.choice(range(0,length), int(length*0.4), replace=False)
        test_names = []

        for i in lst:
            test_names.append(names[i])
        
        for f in test_names:
            shutil.move(f, path + '/' + 'test' + '/' + direction)
        
        for f in glob.glob(path + '/*' + direction + '-*.jpg'):
            shutil.move(f, path + '/' + 'train' + '/' + direction)
    
        names = glob.glob(path + '/test/' + direction + '/*.jpg')
        length = len(names)
        lst = np.random.choice(range(0,length), int(length*0.5), replace=False)
        test_names = []
        
        for i in lst:
            test_names.append(names[i])

        for f in test_names:
            shutil.move(f, path + '/validation/' + direction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segregate images')
    parser.add_argument('-p', '--path', help='Input folder', type=str, required=True)
    args = parser.parse_args()

    main(path=args.path)
