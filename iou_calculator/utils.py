import plotly.graph_objects as go
from PIL import Image, ImageDraw


def clip_corner(corner, LARD=False):
    """
    Be careful this is adapted to MNIST dataset !
    """
    if corner<0:
        return(0)
    
    if LARD == True: 
        if corner>256:
            return(256)
    else:
        if corner>90:
            return(90)
        
    return(corner)
    
def Merge(dict1, dict2): 
    res = {**dict1, **dict2}
    return res

def show_im(X, name, gt_box):
    X_t = X.squeeze()
    X_t = X_t.mul(255).byte()
    pil_image = Image.fromarray(X_t.cpu().numpy(), mode="L")
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle(gt_box, outline="red")
    pil_image.save(name)
    #pil_image.show()

def check_gt_box(gt_box):
    x_bl_gt=gt_box[0]
    x_tr_gt=gt_box[2]
    y_bl_gt=gt_box[1]
    y_tr_gt=gt_box[3]
    if x_bl_gt > x_tr_gt:
        raise ValueError("Ground truth: x_min is greater than x_max.")
    elif y_bl_gt > y_tr_gt:
        raise ValueError("Ground truth: y_min is greater than y_max.")


def check_box_elem(elem):
    if elem>=90 or elem<=0:
        return(True)
    else:
        return(False)
    
def check_box(box):
    sum_elem = sum([check_box_elem(elem) for elem in box])
    if sum_elem>0:
        return(False)
    else:
        return(True)
    

def show_im_origin(X, name, gt_box, pred1 = [0,0,0,0], pred2 = [0,0,0,0]):
    xmin, ymin, xmax, ymax = gt_box
    pred1_xmin, pred1_ymin, pred1_xmax, pred1_ymax = pred1
    pred2_xmin, pred2_ymin, pred2_xmax, pred2_ymax = pred2
    X_t = X.squeeze()
    X_t = X_t.mul(255).byte()
    pil_image = Image.fromarray(X_t.cpu().numpy(), mode="L").convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle([ymin, xmin, ymax, xmax], outline="green")
    draw.rectangle([pred1_ymin, pred1_xmin, pred1_ymax, pred1_xmax], outline="red")
    draw.rectangle([pred2_ymin, pred2_xmin, pred2_ymax, pred2_xmax], outline="orange")
    pil_image.save(name)
    #pil_image.show()


def display(df3):
    # Create a figure
    fig = go.Figure()


    # Add shaded area for condition_a
    fig.add_trace(go.Scatter(
        x=df3.index,
        y=df3['IoU_vanilla_reluval_upper'],
        fill=None,
        mode='lines',
        line_color='lightblue',
        name='Vanilla IoU Upper'
    ))

    fig.add_trace(go.Scatter(
        x=df3.index,
        y=df3['IoU_vanilla_reluval_lower'],
        fill='tonexty',  # fill area between trace0 and trace1
        mode='lines',
        line_color='lightblue',
        name='Vanilla IoU Lower'
    ))

    # Add shaded area for condition_b
    fig.add_trace(go.Scatter(
        x=df3.index,
        y=df3['IoU_optim_2_upper'],
        fill=None,
        mode='lines',
        line_color='lightcoral',
        name='IoU extension Upper'
    ))

    fig.add_trace(go.Scatter(
        x=df3.index,
        y=df3['IoU_optim_2_lower'],
        fill='tonexty',  # fill area between trace2 and trace3
        mode='lines',
        line_color='lightcoral',
        name='IoU extension Lower'
    ))

    # Add scatter points for the average of each interval
    fig.add_trace(go.Scatter(
        x=df3.index,
        y=(df3['IoU_vanilla_reluval_lower'] + df3['IoU_vanilla_reluval_upper']) / 2,
        mode='markers',
        marker_color='blue',
        name='Vanilla IoU average'
    ))

    fig.add_trace(go.Scatter(
        x=df3.index,
        y=(df3['IoU_optim_2_lower'] + df3['IoU_optim_2_upper']) / 2,
        mode='markers',
        marker_color='red',
        name='IoU extension average'
    ))

    # Set the layout
    fig.update_layout(
        title="IoU fragility against adversarial perturbations with the two methods",
        xaxis_title="Perturbtion",
        yaxis_title="IoU",
        showlegend=True
    )

    fig.show()