import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator



def load_img_train(batch, train_dir, imgSize):
        
    train_datagen = ImageDataGenerator()
    train = train_datagen.flow_from_directory(
        train_dir,
        target_size=(imgSize, imgSize),
        color_mode='grayscale',
        batch_size=batch,
        class_mode='categorical')
    
    return train




def showX(X, image_size, rows=1):
    
    assert X.shape[0]%rows == 0
    int_X = ( (X+1)/2*255).clip(0,255).astype('uint8')
    int_X = np.moveaxis(int_X.reshape(-1,image_size,image_size,1), 1, 1)
    int_X = int_X.reshape(rows, -1, image_size, image_size, 1).swapaxes(1,2).reshape(rows*image_size,-1, 1)

    return int_X




class loadImg4Classification:
        
    def train(isize, batch, train_dir, 
              rescale=1./255, rotation_range=20, width_shift_range=0.2, 
              height_shift_range=0.2, horizontal_flip=True):
        
        train_datagen = ImageDataGenerator(
            rescale=rescale,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            horizontal_flip=horizontal_flip,
            fill_mode='nearest')
        
        train = train_datagen.flow_from_directory(
            train_dir,
            target_size=(isize, isize),
            color_mode='grayscale',
            batch_size=batch,
            class_mode='categorical',
            shuffle=True)
        
        return train
        
        
        
    def valid(isize, batch, valid_dir, rescale=1./255):
        
        validation_datagen = ImageDataGenerator(rescale=rescale)
        validation_generator = validation_datagen.flow_from_directory(
            valid_dir,
            target_size=(isize, isize),
            color_mode='grayscale',
            batch_size=batch,
            class_mode='categorical',
            shuffle=True)
        
        return validation_generator
        
        
        
    def test(isize, batch, test_dir):
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(isize, isize),
            color_mode='grayscale',
            batch_size=batch,
            class_mode='categorical',
            shuffle=False)
        
        return test_generator
    

    
    
def prepareTestingData(testImgPath, imgSize):
    
    imgNumber = 0
    for imgClass in os.listdir(testImgPath):
        imgNumber += (len(os.listdir(testImgPath+imgClass)))
        
    test_batches = ImageDataGenerator().flow_from_directory(testImgPath,
                                                            target_size=(imgSize, imgSize), 
                                                            batch_size=imgNumber,
                                                            class_mode = 'categorical',
                                                            color_mode='grayscale',
                                                            shuffle=False)


    positive_image_number = len(test_batches.classes[test_batches.classes==0])
    negative_image_number = len(test_batches.classes[test_batches.classes==1])
    print('loading images...')
    imgs, labels = next(test_batches)
    
    return imgs, labels, positive_image_number, negative_image_number