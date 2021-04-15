import os
from tqdm import tqdm
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''

    heatmap = np.zeros((n_rows, n_cols))
    T_height, T_width = np.shape(T)
    for i in tqdm(range(n_rows-T_height)):
        if i % 50 == 0:
            plt.imshow(heatmap)
            plt.savefig('%d-saved_figure.png' % i)

        for j in range(n_cols-T_width):
            test_I = I[i:i+T_height,j:j+T_width]
            for test_i in range(T_height):
                for test_j in range(T_width):
                    test_T = np.ones(3)*T[test_i, test_j]
                    dist = np.dot(test_I[test_i, test_j], test_T)
                    heatmap_i = i + T_height // 2
                    heatmap_j = j + T_width // 2
                    heatmap[heatmap_i, heatmap_j] = np.arccos(dist) / np.pi

    plt.imshow(heatmap)
    plt.savefig('heatmap.png') 

    '''
    END YOUR CODE
    '''

    return heatmap


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    (n_rows,n_cols) = np.shape(heatmap)
    box_dims = [(100, 50), (20, 10), (10, 5)]
    threshold = 10

    for box_dim in box_dims:
        box_height, box_width = box_dim
        for i in range(n_rows - box_height):
            for j in range(n_cols - box_width):
                candidate_box = heatmap[i:i+box_height, j:j+box_width]
                score = np.mean(candidate_box)
                if score > threshold:
                    tl_row = i
                    tl_col = j
                    br_row = i + box_height
                    br_col = j + box_width
                    output.append([tl_row,tl_col,br_row,br_col, score])
    plt.imshow(heatmap)
    plt.savefig('heatmap.png') 

    '''
    END YOUR CODE
    '''

    return output

def show_img(I, img_name, rect_list):
    # Create figure and axes
    fig, ax = plt.subplots()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    # Display the image
    ax.imshow(I)
    for rect in rect_list:
        tl_row, tl_col, br_row, br_col = rect
        # Create a Rectangle patch
        rect = patches.Rectangle((tl_col-1, tl_row-1), br_col-tl_col, br_row-tl_row,
                                 linewidth=1, edgecolor='r', facecolor='none')
        
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.title(img_name)
    plt.savefig(img_name)

def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    template_height = 8
    template_width = 6

    # You may use multiple stages and combine the results
    T = np.random.random((template_height, template_width))

    heatmap = compute_convolution(I, T)
    print('computed heatmap')
    output = predict_boxes(heatmap)
    print('computed output')

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

user_profile = os.environ['HOME']

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '%s/data/EE148/RedLights2011_Medium' % user_profile

# load splits: 
split_path = '%s/data/EE148/hw02_splits' % user_profile
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '%s/data/EE148/hw02_preds' % user_profile
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)
    print(preds_train[file_names_train[i]])

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
