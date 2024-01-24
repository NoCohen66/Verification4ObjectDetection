import numpy as np
from matplotlib import pyplot as plt
from random import randrange
import torch 


def show_img_from_df(df, ind):
    plt.imshow(np.reshape(np.array(df.iloc[ind,1:]), (28,28)), cmap="gray")

def random_corners(new_size,img):
    # randomly select a top left corner to use for img
    x_min, y_min = randrange(new_size - img.shape[0]), randrange(new_size - img.shape[0])
    # compute bottom right corner
    x_max, y_max = x_min + img.shape[0], y_min + img.shape[0]
    return x_min,y_min,x_max,y_max #return top left, bottom right coordinates

def get_center(x_min,y_min,x_max,y_max):
    #compute center coordinates given corner points 
    x_center = x_min + (x_max-x_min)/2
    y_center = y_min + (y_max-y_min)/2
    return x_center,y_center


def train(dataloader, model, loss_fn, loss_box, optimizer, alpha, beta):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train() # very important... This turns the model back to training mode
    size = len(dataloader.dataset) #len(train_dataloader.dataset)

    
    loss_dig_list = []
    loss_iou_list = []
    
    for batch, (X, y) in enumerate(dataloader):
        #print(f"X.shape {X.shape}")
        X, y0, y1 = X.to(device), y[0].to(device), y[1].to(device) #y0, y1 : ground truth values

        y0_pred, y1_pred = model(X.float()) #y0_pred corresponds to the digit, y1_pred corresponds to the list [x_min, y_min, x_max, y_max]
        
        loss_dig = loss_fn(y0_pred, y0)
        #print(y1_pred.detach().numpy().shape)
        loss_iou = loss_box(y1_pred, y1.float()) #[0].detach().numpy()
        loss = alpha*loss_dig + beta*loss_iou

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            
            loss_dig = loss_dig.item()
            loss_iou = loss_iou.item()
            
            loss_dig_list.append(loss_dig)
            loss_iou_list.append(loss_iou)
            
            print(f"MAIN loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"Digit prediction loss: {loss_dig:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"Coordinate prediction loss (MSE): {loss_iou:>7f}  [{current:>5d}/{size:>5d}]")
            print("-----------")
            

# TODO: make this work for three outputs....

def test(dataloader, model, loss_fn, loss_box, alpha=100, beta=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = len(dataloader.dataset)
    model.eval()
    test_loss, test_loss_y0, test_loss_y1, correct = 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y0, y1 = X.to(device), y[0].to(device), y[1].to(device)
            y0_pred, y1_pred = model(X.float())
            #test_loss += alpha*loss_fn(y0_pred, y0).item() + beta*loss_box(y1_pred, y1.float()).item()
            test_loss_y0 += loss_fn(y0_pred, y0).item()
            test_loss_y1 += loss_box(y1_pred, y1.float()).item()
            test_loss += alpha*test_loss_y0 + beta*test_loss_y1
            
            correct += (y0_pred.argmax(1) == y0).type(torch.float).sum().item() # only for digit predictions
            
    # average the loss and accuracy among all records used in the dataset
    test_loss /= size
    test_loss_y0 /= size
    test_loss_y1 /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg digit loss: {test_loss_y0:>8f}, Avg coordinate loss: {test_loss_y1:>8f} \n")
    acc=round(100*correct)
    avg_digit_loss=round(test_loss_y0,6)
    avg_coordinate_loss=round(test_loss_y1,6)
    return acc, avg_digit_loss, avg_coordinate_loss