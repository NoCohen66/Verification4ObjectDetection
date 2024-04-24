from solver.utils import get_center
from solver.utils import random_corners
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
        x_min,y_min,x_max,y_max=random_corners(new_size,image)
        x_center,y_center=get_center(x_min,y_min,x_max,y_max)

        

        new_img[x_min:x_max, y_min:y_max] = image
        
     
        new_img = np.reshape(new_img, (1,90,90)) #batch

        label = [int(self.df.iloc[idx,0]), np.array([x_min,y_min,x_max,y_max]).astype('float32')]
        sample = {"image": new_img, "label": label}
        
        return sample['image'], sample['label']
