from config import device
import torch
from tqdm import tqdm
from model import YOLOv3, YOLOLoss
import  dataloader
import model2
import config
import torch.optim as optim 
import helper
import utils
#from dataloader import loader
import torch

print(torch.cuda.is_available())

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"): 
    print("==> Saving checkpoint") 
    checkpoint = { 
        "state_dict": model.state_dict(), 
        "optimizer": optimizer.state_dict(), 
    } 
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr): 
    print("==> Loading checkpoint") 
    checkpoint = torch.load(checkpoint_file, map_location=device) 
    model.load_state_dict(checkpoint["state_dict"]) 
    optimizer.load_state_dict(checkpoint["optimizer"]) 
  
    for param_group in optimizer.param_groups: 
        param_group["lr"] = lr 
    

# Define the train function to train the model 
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors): 
    # Creating a progress bar 
    progress_bar = tqdm(loader, leave=True) 
  
    # Initializing a list to store the losses 
    losses = [] 
  
    # Iterating over the training data 
    for _, (x, y) in enumerate(progress_bar): 
        x = x.to(device) 
        y0, y1, y2 = ( 
            y[0].to(device), 
            y[1].to(device), 
            y[2].to(device), 
        ) 
  
        with torch.cuda.amp.autocast(): 
            # Getting the model predictions 
            outputs = model(x) 
            # Calculating the loss at each scale 
            loss = ( 
                  loss_fn(outputs[0], y0, scaled_anchors[0]) 
                + loss_fn(outputs[1], y1, scaled_anchors[1]) 
                + loss_fn(outputs[2], y2, scaled_anchors[2]) 
            ) 
  
        # Add the loss to the list 
        losses.append(loss.item()) 
  
        # Reset gradients 
        optimizer.zero_grad() 
  
        # Backpropagate the loss 
        scaler.scale(loss).backward() 
  
        # Optimization step 
        scaler.step(optimizer) 
  
        # Update the scaler for next iteration 
        scaler.update() 
  
        # update progress bar with loss 
        mean_loss = sum(losses) / len(losses) 
        progress_bar.set_postfix(loss=mean_loss)


# Testing YOLO v3 model 
if __name__ == "__main__": 
    # Setting number of classes and image size 
    num_classes = 1
    IMAGE_SIZE = 416
  
    # Creating model and testing output shapes 
    model = YOLOv3(num_classes=num_classes) 
    x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE)) 
    out = model(x) 
    print(out[0].shape) 
    print(out[1].shape) 
    print(out[2].shape) 
  
    # Asserting output shapes 
    assert model(x)[0].shape == (1, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5) 
    assert model(x)[1].shape == (1, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5) 
    assert model(x)[2].shape == (1, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5) 
    print("Output shapes are correct!")


# Creating the model from YOLOv3 class 
model = YOLOv3().to(device) 
  
# Defining the optimizer 
optimizer = optim.Adam(model.parameters(), lr = config.learning_rate) 
  
# Defining the loss function 
loss_fn = YOLOLoss() 
  
# Defining the scaler for mixed precision training 
scaler = torch.cuda.amp.GradScaler() 

# Defining the train dataset 
train_dataset = dataloader.Dataset( 
    csv_file= '/Users/pavithrak/CUAD/Pavithra_Repo/avg_mast_data.csv',
    anchors=config.ANCHORS, 
    transform=dataloader.train_transform
)

# Defining the train data loader 
train_loader = torch.utils.data.DataLoader( 
    train_dataset, 
    batch_size = config.batch_size, 
    # num_workers = 1, 
    shuffle = True, 
    pin_memory = True, 
) 
  

# Scaling the anchors 
scaled_anchors = ( 
    torch.tensor(config.ANCHORS) *
    torch.tensor(config.s).unsqueeze(1).unsqueeze(1).repeat(1,3,2) 
).to(device) 
  
# Training the model 
for e in range(1, config.epochs+1): 
    print("Epoch:", e) 
    training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors) 

    # Saving the model 
    if config.save_model: 
        save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")


#Testing data
# Setting the load_model to True 
load_model = True
  
# Defining the model, optimizer, loss function and scaler 
model = YOLOv3().to(device) 

optimizer = optim.Adam(model.parameters(), lr = config.learning_rate) 
loss_fn = YOLOLoss() 
scaler = torch.cuda.amp.GradScaler() 
  
# Loading the checkpoint 
if load_model: 
    load_checkpoint(config.checkpoint_file, model, optimizer, config.learning_rate) 
  
# Defining the test dataset and data loader 
test_dataset = dataloader.Dataset( 
    csv_file= '/Users/pavithrak/CUAD/Pavithra_Repo/new_test_data.csv',#"/Users/pavithrak/CUAD/Pavithra_Repo/avg_test_mast_data.csv",  
    anchors=config.ANCHORS, 
    transform=dataloader.test_transform 
) 
test_loader = torch.utils.data.DataLoader( 
    test_dataset, 
    batch_size = 32, 
    # num_workers = 1, 
    shuffle = True, 
) 
  
# Getting a sample image from the test data loader 
x, y = next(iter(test_loader)) 
x = x.to(device) 
  
model.eval() 
with torch.no_grad(): 
    # Getting the model predictions 
    output = model(x) 
    # Getting the bounding boxes from the predictions 
    bboxes = [[] for _ in range(x.shape[0])] 
    
    anchors = ( 
            torch.tensor(config.ANCHORS) 
                * torch.tensor(config.s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) 
            ).to(device) 
  
    # Getting bounding boxes for each scale 
    for i in range(3): 
        batch_size, A, S, _, _ = output[i].shape 
        anchor = anchors[i] 
        boxes_scale_i = helper.convert_cells_to_bboxes( 
                            output[i], anchor, s=S, is_predictions=True
                        ) 
        for idx, (box) in enumerate(boxes_scale_i): 
            bboxes[idx] += box 
model.train() 
  
# Plotting the image with bounding boxes for each image in the batch 
for i in range(batch_size): 
    # Applying non-max suppression to remove overlapping bounding boxes 
    nms_boxes = helper.nms(bboxes[i], iou_threshold=0.5, threshold=0.6) 
    # Plotting the image with bounding boxes 
    utils.plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)