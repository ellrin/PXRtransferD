from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, Cropping2D
from keras.layers import BatchNormalization, Input, Flatten
from keras.layers import Conv2DTranspose, Reshape, Activation
from keras.layers.advanced_activations import LeakyReLU




def DCGAN_D(isize, nz, nc, conv_init, ndf=128):
    
    x = inputs = Input(shape=(isize, isize, nc))
    x = Conv2D(filters=ndf, kernel_size=4, strides=1, use_bias=False,
               padding = "same", kernel_initializer = conv_init)(x)
    x = Conv2D(filters=ndf, kernel_size=4, strides=2, use_bias=False,
               padding = "same", kernel_initializer = conv_init)(x) 
    x = LeakyReLU(alpha=0.2)(x)
    
    
    x = Conv2D(filters=ndf*2, kernel_size=4, strides=2, use_bias=False,
               padding = "same", kernel_initializer = conv_init)(x)
    x = Conv2D(filters=ndf*2, kernel_size=4, strides=1, use_bias=False,
               padding = "same", kernel_initializer = conv_init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    

    x = Conv2D(filters=ndf*4, kernel_size=4, strides=2, use_bias=False,
               padding = "same", kernel_initializer = conv_init)(x)
    x = Conv2D(filters=ndf*4, kernel_size=4, strides=1, use_bias=False,
               padding = "same", kernel_initializer = conv_init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    
    x = Conv2D(filters=nc, kernel_size=3, strides=1, use_bias=False,
               padding = "same",kernel_initializer = conv_init)(x) 
    x = Conv2D(filters=1, kernel_size=int(x.shape[1]), strides=1, use_bias=False,
               kernel_initializer = conv_init)(x)
    
    outputs = Flatten()(x)
    
    return Model(inputs=inputs, outputs=outputs)




def DCGAN_G(isize, nz, nc, conv_init, gamma_init, init_filters=1024):
    
    assert isize%32 == 0
    
    init_KernelSize = isize//32
    
    x = inputs = Input(shape=[nz])
    x = Reshape((1,1,nz))(x)
    
    x = Conv2DTranspose(filters=init_filters, kernel_size=init_KernelSize, 
                        strides=1, use_bias=False, kernel_initializer=conv_init)(x)
    x = BatchNormalization(gamma_initializer = gamma_init, momentum=0.9, axis=1)(x,training=True)
    x = Activation("relu")(x)
      
    for idx in range(4):

        filtersNumber = init_filters//(2**(idx+1))
        x = Conv2DTranspose(filters=filtersNumber, kernel_size=4, strides=2, use_bias=False, 
                            kernel_initializer=conv_init, padding='same')(x)
        x = BatchNormalization(gamma_initializer = gamma_init, momentum=0.9, axis=1)(x, training=True)
        x = Activation("relu")(x)

    x = Conv2DTranspose(filters=nc, kernel_size=4, strides=2, use_bias=False,
                        kernel_initializer = conv_init, padding="same")(x)

    outputs = Activation("tanh")(x)
    
    return Model(inputs=inputs, outputs=outputs)