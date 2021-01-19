import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model



def img_resize(img, desired_size):
    
    assert len(img.shape) ==3
        
    init_size = img.shape[:2]  # arr.shape = (12,24,36), arr.shape[:2] = (12,24)
    ratio = float(desired_size) / max(init_size)
    new_size = tuple([int(x*ratio) for x in init_size])  # tuple is a python way to save the string
    img = cv2.resize(img, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h - (delta_h//2)
    left, right = delta_w//2, delta_w - (delta_w//2)
        
    color = [0,0,0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return new_img



def get_test_img(number, imgs):
    
    img = imgs[number]
    img = np.expand_dims(img, axis=0)
    
    return img




def ScoreCam(model, img_array, layer_name, max_N=-1):

    cls = np.argmax(model.predict(img_array))
    act_map_array = Model(inputs=model.input, outputs=model.get_layer(layer_name).output).predict(img_array)
    
    # extract effective maps
    if max_N != -1:
        act_map_std_list = [np.std(act_map_array[0,:,:,k]) for k in range(act_map_array.shape[3])]
        unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_N)[:max_N]
        max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
        act_map_array = act_map_array[:,:,:,max_N_indices]

    input_shape = model.layers[0].output_shape[1:]  # get input shape
    
    # 1. upsample 
    act_map_resized_list = [cv2.resize(act_map_array[0,:,:,k], input_shape[:2], 
                                       interpolation=cv2.INTER_LINEAR) for k in range(act_map_array.shape[3])]
    
    # 2. normalize
    act_map_normalized_list = []
    
    for act_map_resized in act_map_resized_list:
        
        if np.max(act_map_resized) - np.min(act_map_resized) != 0:
            act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
            
        else:
            act_map_normalized = act_map_resized
        act_map_normalized_list.append(act_map_normalized)
        
    # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
    
    masked_input_list = []
    for act_map_normalized in act_map_normalized_list:
        masked_input = np.copy(img_array)
        for k in range(1):
            masked_input[0,:,:,k] *= act_map_normalized
        masked_input_list.append(masked_input)
    masked_input_array = np.concatenate(masked_input_list, axis=0)
    
    
    # 4. feed masked inputs into CNN model and softmax
    def softmax(x):
        f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
        return f
    pred_from_masked_input_array = softmax(model.predict(masked_input_array))
    
    
    # 5. define weight as the score of target class
    weights = pred_from_masked_input_array[:,cls]
    
    
    # 6. get final class discriminative localization map as linear weighted combination of all activation maps
    cam = np.dot(act_map_array[0,:,:,:], weights)
    cam = np.maximum(0, cam)  # Passing through ReLU
    cam /= np.max(cam)  # scale 0 to 1.0
    
    return cam




def GradCam(model, img_array, layer_name):
    
    cls = np.argmax(model.predict(img_array))
    
    """GradCAM method for visualizing input saliency."""
    y_c = model.output[0, cls]
    conv_output = model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # grads = normalize(grads)

    gradient_function = K.function([model.input], [conv_output, grads])
    output, grads_val = gradient_function([img_array])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    weights = np.mean(grads_val, axis=(0, 1))

    cam = np.dot(output, weights)
    cam = np.maximum(cam, 0)  # Passing through ReLU
    cam /= np.max(cam)  # scale 0 to 1.0  

    return cam




def superimpose(original_img, cam, a=100, b=0.5, c=1, emphasize=False):
    
    heatmap = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    if emphasize:

        def sigmoid(x, a=a, b=b, c=c):
            return c / (1 + np.exp(-a * (x-b)))

        heatmap = sigmoid(heatmap, a,b,c)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    hif = .4
    superimposed_img = heatmap * hif + original_img
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  
    superimposed_img_opt = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return superimposed_img_opt




def get_heatmap_imgs(img_number, test_img, prediction_model, intermediate_layername):

    img_array = get_test_img(img_number, test_img)
    pred = prediction_model.predict(img_array)
    
    grad_cam  = GradCam(model=prediction_model, img_array=img_array, layer_name=intermediate_layername)
    score_cam = ScoreCam(model=prediction_model, img_array=img_array, layer_name=intermediate_layername)
    
    grad_cam_resized      = cv2.resize(grad_cam, (test_img[img_number].shape[1], test_img[img_number].shape[0]))
    grad_cam_superimposed = superimpose(test_img[img_number]*255, grad_cam)
    grad_cam_emphasize    = superimpose(test_img[img_number]*255, grad_cam, emphasize=True)
    
    score_cam_resized      = cv2.resize(score_cam, (test_img[img_number].shape[1], test_img[img_number].shape[0]))
    score_cam_superimposed = superimpose(test_img[img_number]*255, score_cam)
    score_cam_emphasize    = superimpose(test_img[img_number]*255, score_cam, emphasize=True)


    # comb = (grad_cam_resized+score_cam_resized)/2
    comb = (grad_cam_resized+score_cam_resized)/2
    comb_img = superimpose(test_img[img_number]*255, comb, emphasize=True)
    
    fig = plt.figure(figsize=(20,10))
    plt.subplot(2,3,1)
    plt.title('Frature probability : %f'%(pred[0][0]))
    plt.imshow(img_resize(img_array[0], 512), cmap='gray')
    plt.subplot(2,3,2)
    plt.title('grad cam')
    plt.imshow(grad_cam_superimposed)
    plt.subplot(2,3,3)
    plt.title('score cam')
    plt.imshow(score_cam_superimposed)
    plt.subplot(2,3,4)
    plt.title('combination heatmap')
    plt.imshow(comb_img)
    plt.subplot(2,3,5)
    plt.title('grad cam emphasize')
    plt.imshow(grad_cam_emphasize)
    plt.subplot(2,3,6)
    plt.title('score cam emphasize')
    plt.imshow(score_cam_emphasize)
    plt.close()
    
    return fig