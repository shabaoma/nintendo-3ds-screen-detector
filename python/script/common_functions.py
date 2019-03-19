import numpy as np
import pandas as pd
from keras.preprocessing import image

def train_val_test_data(data_x, data_y, p1, p2):
    """
    input
        data_x: (numpy) shape = (n, d1, d2, ...)
        data_y: (numpy) shape = (n, d1, d2, ...)
        p1, p2: [0, p1) train [p1, p2) val [p2, 1] test
    output
        x_train, x_val, x_test: (numpy) shape = (n, d1, d2, ...)
        y_train, y_val, y_test: (numpy) shape = (n, d1, d2, ...)
    """
    
    index_df = pd.DataFrame({'index': range(len(data_x))}).sample(frac=1)
    total_length = len(index_df)
    train_index = index_df.iloc[:int(total_length * p1)].index
    val_index = index_df.iloc[int(total_length * p1): int(total_length * p2)].index
    test_index = index_df.iloc[int(total_length * p2):].index
    
    x_train = np.concatenate([[data_x[index]] for index in train_index])
    x_val = np.concatenate([[data_x[index]] for index in val_index])
    x_test = np.concatenate([[data_x[index]] for index in test_index])
    print(x_train.shape, x_val.shape, x_test.shape)

    y_train = np.concatenate([[data_y[index]] for index in train_index])
    y_val = np.concatenate([[data_y[index]] for index in val_index])
    y_test = np.concatenate([[data_y[index]] for index in test_index])
    print(y_train.shape, y_val.shape, y_test.shape)
    
    return x_train, x_val, x_test, y_train, y_val, y_test

def coord_to_mask(coord_array):
    """
    input
        coord_array: (numpy) shape = (1, 8), value = [0, 1]
    output
        mask: (numpy) shape = (len_x, len_y, 1), value = [0 | 1]
    """
    points = np.array([[int(coord_array[0]*255.0), int(coord_array[1]*255.0)], 
                       [int(coord_array[2]*255.0), int(coord_array[3]*255.0)], 
                       [int(coord_array[4]*255.0), int(coord_array[5]*255.0)], 
                       [int(coord_array[6]*255.0), int(coord_array[7]*255.0)]])
    single_image = np.zeros((256, 256, 1)).astype(np.float32)
    cv2.fillConvexPoly(single_image, points, 1)
    return single_image

def show_image(array):
    """
    input
        array: (numpy) (len_x, len_y, 1), value = [0 | 1]
    """
    display(image.array_to_img(array))
