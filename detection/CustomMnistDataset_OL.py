from detection.utils_detection import get_center
from detection.utils_detection import random_corners
from torch.utils.data import Dataset
import numpy as np



class CustomMnistDataset_OL(Dataset):
    def __init__(self, df, test=False):
        '''
        df is a pandas dataframe with 28x28 columns for each pixel value in MNIST
        '''
        self.df = df
        self.test = test
        print(df.shape)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        #img into 28x28 array
        if self.test:
            image = np.reshape(np.array(self.df.iloc[idx,:]), (28,28)) / 255.
        else:#if train, need to skip the first column which represents the label
            image = np.reshape(np.array(self.df.iloc[idx,1:]), (28,28)) / 255.
        
        new_size = 90    
        # create the new image
        new_img = np.zeros((new_size, new_size)) # images will be 90x90
        # randomly select a top left corner to use for img
        x_min,y_min,x_max,y_max=random_corners(new_size,image)
        x_center,y_center=get_center(x_min,y_min,x_max,y_max)

        # try normalizing this part of the output
        #x_center = x_center #/ 90.
        #y_center = y_center #/ 90.
        

        new_img[x_min:x_max, y_min:y_max] = image
        
        #NEW
        new_img = np.reshape(new_img, (1,90,90)) #batch
        
        #label = [int(self.df.iloc[idx,0]), np.array([x_center, y_center]).astype('float32')] # the label consists of the digit and the center of the number, bottom left 
        #NEED TO MODIFY HERE, ADD CORNER POINTS TO MAKE IT EASY FOR THE IoU COMPUTATION : 
        label = [int(self.df.iloc[idx,0]), np.array([x_min,y_min,x_max,y_max]).astype('float32')]
        sample = {"image": new_img, "label": label}
        
        return sample['image'], sample['label']
