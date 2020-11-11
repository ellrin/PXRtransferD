from keras.preprocessing.image import ImageDataGenerator
import numpy as np




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