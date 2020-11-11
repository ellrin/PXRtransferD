# import packages

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras.backend as K
from keras.layers import Input
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from utils.model import DCGAN_D, DCGAN_G
from utils.preprocessing import load_img_train, showX


# initial settings
with open('gan_info.json') as info:
    train_info = json.load(info)


Diters             = train_info['Diters']
λ                  = int(train_info['lambda_number'])
lrD                = train_info['lrD']
lrG                = train_info['lrG']
gpu_number         = train_info['gpu_number']
imgSize            = train_info['image_size']
batch_size         = train_info['batch_size']
epochs             = train_info['epochs']
initial_dir        = train_info['initial_dir_path']
train_dir          = train_info['train_dir_path']
latent_code_number = nz = train_info['latent_code_number']
image_channel      = nc = train_info['image_channel']

if gpu_number != None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)

init_gen_iterations = 0
init_errG = 0
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)
gan_log_path = os.getcwd()+'/gan_loss_'+str(imgSize)+'.csv'

print('gpu:',gpu_number,
      '\nimg Size:',imgSize,
      '\nchannel:',nc,
      '\nz:',nz,
      '\nbatchs:', batch_size,
      '\nepochs:', epochs,
      '\nDiters:', Diters)




# mkdir
try:
    gan_path = initial_dir+'gan_'+str(imgSize)
    os.mkdir(gan_path)
    os.mkdir(gan_path+'/D')
    os.mkdir(gan_path+'/G')
    os.mkdir(gan_path+'/images')
    
except:
    pass

D_path = gan_path+'/D'
G_path = gan_path+'/G'
img_path = gan_path+'/images'




# model setting

netD = DCGAN_D(imgSize, nz, nc, conv_init)
netG = DCGAN_G(imgSize, nz, nc, conv_init, gamma_init)
print('\n\n\n\n\nnetD')
netD.summary()
print('\n\n\n\n\nnetG')
netG.summary()

netD_real_input = Input(shape=(imgSize, imgSize, nc))
noisev = Input(shape=(nz,))
netD_fake_input = netG(noisev)

ϵ_input          = K.placeholder(shape=(None,1,1,1))
netD_mixed_input = Input(shape=(imgSize, imgSize, nc), tensor=ϵ_input * netD_real_input + (1-ϵ_input) * netD_fake_input)  
loss_real        = K.mean(netD(netD_real_input))
loss_fake        = K.mean(netD(netD_fake_input))

grad_mixed       = K.gradients(netD(netD_mixed_input), [netD_mixed_input])[0]
norm_grad_mixed  = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2,3]))
grad_penalty     = K.mean(K.square(norm_grad_mixed -1))

loss = loss_fake - loss_real + λ * grad_penalty


training_updates = Adam(lr=lrD, beta_1=0.0, beta_2=0.9).get_updates(netD.trainable_weights,[],loss)
netD_train = K.function([netD_real_input, noisev, ϵ_input], [loss_real, loss_fake], training_updates)



loss = -loss_fake 
training_updates = Adam(lr=lrG, beta_1=0.0, beta_2=0.9).get_updates(netG.trainable_weights,[], loss)
netG_train = K.function([noisev], [loss], training_updates)




train = load_img_train(1, train_dir, imgSize)
oneMatrix = np.ones((train.samples, imgSize, imgSize, 1))

for i in range(train.samples):
    oneMatrix[i] = (train[i][0][0] - 127.5) / 127.5
    
train_all = oneMatrix






t0 = time.time()
errG = init_errG
gen_iterations = init_gen_iterations
fixed_noise = np.random.normal(size=(batch_size, nz)).astype('float32')

training_record = []
targetD = np.float32([2]*batch_size+[-2]*batch_size)[:, None]
targetG = np.ones(batch_size, dtype=np.float32)[:, None]

for epoch in range(epochs):
    
    i = 0
    print('[epoch : %d]'%(epoch+1))
    
    np.random.shuffle(train_all)
    batches = train_all.shape[0]//batch_size
    
    while i < batches:
        if gen_iterations < 32 or gen_iterations % 50 == 0:
            _Diters = 100
        else:
            _Diters = Diters
            
        j = 0
        
        while j < _Diters and i < batches:
            
            j+=1
            real_data = train_all[i*batch_size:(i+1)*batch_size]
            
            i+=1
            noise = np.random.normal(size=(batch_size, nz))
            ϵ = np.random.uniform(size=(batch_size, 1, 1 ,1))
            errD_real, errD_fake  = netD_train([real_data, noise, ϵ])
            errD = errD_real - errD_fake
       
        if gen_iterations%500==0:
            
            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (epoch+1, epochs, i, batch_size, gen_iterations,errD, errG, errD_real, errD_fake), time.time()-t0)
            
            fake = netG.predict(fixed_noise)
            epoch_img = showX(fake, imgSize, 4)
            plt.imsave(img_path+'/epoch_%d'%(epoch)+'.png', epoch_img[:,:,0], cmap='gray')
            netD.save(D_path+'/net_D_epoch_%d'%(epoch))
            netG.save(G_path+'/net_G_epoch_%d'%(epoch))
            
            training_record.append((epoch+1, errD, errG, time.time()-t0))
            training_record_df = pd.DataFrame(training_record, columns=['epoch', 'errD', 'errG', 'time'])
            training_record_df.to_csv(gan_log_path)
        
        noise = np.random.normal(size=(batch_size, nz))        
        errG, = netG_train([noise])
        gen_iterations+=1
