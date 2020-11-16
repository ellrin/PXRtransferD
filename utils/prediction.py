import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from keras.models import Model




def get_results(testLabel, testProbability, positiveDataNumber, negativeDataNumber, youden_index):

    TP = 0
    TN = 0
    FNidx = []
    FPidx = []
    predicted_bool = []
    testLabel = np.array(testLabel[:,0], dtype=int)
    
    for pred_value in ((testProbability)[:,0]+(1-youden_index)):
        predicted_bool.append(int(pred_value))
        

    for idx, label in enumerate(testLabel[0:positiveDataNumber]):
        if predicted_bool[idx] == label:
            TP+=1
        else:
            FNidx.append(idx)

    for idx, label in enumerate(testLabel[positiveDataNumber:]):
        if predicted_bool[positiveDataNumber+idx] == label:
            TN+=1
        else:
            FPidx.append(idx)

    FP = negativeDataNumber - TN
    FN = positiveDataNumber - TP

    Accuracy = (TP+TN)/(negativeDataNumber+positiveDataNumber)
    Sensitivity = (TP)/(TP+FN)
    Specificity = (TN)/(TN+FP)
    f1 = (2*TP)/(2*TP+FP+FN)
    
    return TP, TN, FP, FN, FPidx, FNidx, Accuracy, Sensitivity, Specificity, f1




def get_tSNE(model, InterMeDiateLayerName, imgs, labels, batch_size, fpidx, fnidx, 
             colors  = ['purple', 'skyblue'], 
             classes = ['hip fracture','no hip fracture'], 
             markers = ['.','+'],
             graphTitle="GAN pre-trained model's tSNE"):
    
    # get the intermediate layer's (ITMDL) output
    testLabel = np.array(labels[:,0], dtype=int)
    ITMDLname = InterMeDiateLayerName
    ITMDLModel    = Model(inputs=model.input, outputs=model.get_layer(ITMDLname).output)
    ITMDL_output  = ITMDLModel.predict(imgs, batch_size=batch_size, verbose=1)
    
    # use tsne for the dimension reduction, and save the results as a dataframe group
    X_transformed = TSNE(n_components=2, verbose=1).fit_transform(ITMDL_output.reshape(ITMDL_output.shape[0],-1))
    transformedDf = pd.DataFrame(dict(x=X_transformed[:,0], y=X_transformed[:,1], label=testLabel))
    tsne_groups   = transformedDf.groupby('label')
    
    # save the false positive(FP) false negative(FN) results as 2 dataframe groups
    FNdf = pd.DataFrame(dict(x=X_transformed[:,0][fnidx], y=X_transformed[:,1][fnidx], label=testLabel[fnidx]))
    FPdf = pd.DataFrame(dict(x=X_transformed[:,0][fpidx], y=X_transformed[:,1][fpidx], label=testLabel[fpidx]))
    FN_groups = FNdf.groupby('label')
    FP_groups = FPdf.groupby('label')
    
    
    # plot the tsne graph
    fig, ax = plt.subplots(figsize=(12,10))
    
    colors     = colors
    class_name = classes
    markers    = markers
    ax.margins(0.05)

    for label, group in tsne_groups:
        name = class_name[label]
        point, = ax.plot(group.x, group.y, marker=markers[label], 
                         linestyle='', ms=8, label=name, alpha=0.8, color = colors[label])

    for label, group in FN_groups:
        name = class_name[label]
        point, = ax.plot(group.x, group.y, marker='^', linestyle='', ms=11, label='FN', alpha=0.8, color = 'orangered')


    for label, group in FP_groups:
        name = class_name[label]
        point, = ax.plot(group.x, group.y, marker='o', linestyle='', ms=11, label='FP', alpha=0.8, color = 'orange')

    plt.title(graphTitle, fontsize=30)
    # plt.ylim((-glim, glim))
    # plt.xlim((-glim, glim))
    plt.grid(ls='--')
    ax.legend(prop={'size': 12})
    plt.close()
    
    return fig