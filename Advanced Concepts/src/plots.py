import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import make_grid
import torch
import numpy as np

DATA_MEAN = (0.4914, 0.4822, 0.4465)
DATA_STD = (0.247, 0.2435, 0.2616)

class Plots:
    def __init__(self):
        pass

    def sampleVisual(dataset):
        batch = next(iter(dataset))
        images, labels = batch
        images = images["image"]
        batch_grid = make_grid(images)
        images = batch_grid.numpy().transpose((1, 2, 0)) # (C, H, W) --> (H, W, C)
        # Convert mean and std to numpy array
        mean = np.asarray(DATA_MEAN)
        std = np.asarray(DATA_STD)
        # unnormalize the image
        images = DATA_STD * images + DATA_MEAN
        images = np.clip(images, 0, 1)
        fig = plt.figure() # Create a new figure
        fig.set_figheight(15)
        fig.set_figwidth(15)
        ax = fig.add_subplot(111)
        ax.axis("off") # Sqitch off the axis
        ax.imshow(images)
        #return plt.imshow(batch_grid[0].squeeze(), cmap='gray_r')
        # return ax.imshow(images)
        
    def plotting(model, loader, device):
        wrong_images = []
        wrong_label = []
        correct_label = []

        with torch.no_grad():
            for img, label in loader:
                img, label = img.to(device), label.to(device)
                pred_label = model(img.to(device))
                pred = pred_label.argmax(dim=1, keepdim=True)

                wrong_pred = (pred.eq(target.view_as(pred)) == False)
                wrong_images.append(data[wrong_pred])
                wrong_label.append(pred[wrong_pred])
                correct_label.append(target.view_as(pred)[wrong_pred])

                wrong_predictions = list(
                    zip(torch.cat(wrong_images), torch.cat(wrong_label), torch.cat(correct_label)))
                fig = plt.figure(figsize=(8, 10))
                fig.tight_layout()
                for i, (img, pred, correct) in enumerate(wrong_predictions[:10]):
                    img, pred, target = img.cpu().numpy(), pred.cpu(), correct.cpu()
                    ax = fig.add_subplot(5, 2, i+1)
                    ax.axis('off')
                    ax.set_title(
                        f'\nactual {target.item()}\npredicted {pred.item()}', fontsize=10)
                    ax.imshow(img.squeeze(), cmap='gray_r')

                plt.show()
            return len(wrong_predictions)
            
    def stat_graph(train_acc, train_losses, test_acc, test_losses):
      fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,10))
      ax1 = ax[0,0]
      ax1.set_title("TrainAccuracy")
      ax1.plot(train_acc, color="blue")
      ax2 = ax[0,1]
      ax2.set_title("TestAccuracy")
      ax2.plot(test_acc, color="blue")
      ax3 = ax[1,0]
      ax3.set_title("TrainLoss")
      ax3.plot(train_losses, color="blue")
      ax4 = ax[1,1]
      ax4.set_title("TestLoss")
      ax4.plot(test_losses, color="blue")
      plt.show()

    def imshow(img):
        images = img.cpu().numpy().transpose((1, 2, 0))
        mean = np.asarray(DATA_MEAN)
        std = np.asarray(DATA_STD)
        # unnormalize the image
        images = DATA_STD * images + DATA_MEAN
        images = np.clip(images, 0, 1)
        plt.imshow(images)
        plt.show()

    def misclassifications(model,test_loader,device):
      wrong_images=[]
      wrong_label=[]
      correct_label=[]
      with torch.no_grad():
          for data, target in test_loader:
              data, target = data["image"].to(device), target.to(device)
              output = model(data)        
              #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              _, predicted = torch.max(output.data, 1)
              # c = (predicted == labels).squeeze()
              # for i in range(4):
              #   label = labels[i]
              #   class_correct[label] += c[i].item()
              #   class_total[label] += 1
                
              wrong_pred = (predicted.eq(target.view_as(predicted)) == False)
              wrong_images.append(data[wrong_pred])
              wrong_label.append(predicted[wrong_pred])
              correct_label.append(target.view_as(predicted)[wrong_pred])  

              wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_label),torch.cat(correct_label)))    
          print(f'Total wrong predictions are {len(wrong_predictions)}')
          
          
          fig = plt.figure(figsize=(2,2))
          fig.tight_layout()
          for i, (img, predicted, correct) in enumerate(wrong_predictions[:10]):
            Plots.imshow(img)
              # img, predicted, target = img.cpu().numpy(), predicted.cpu(), correct.cpu()
              # img = torch.from_numpy(img)
              # ax = fig.add_subplot(5, 2, i+1)
              # ax.axis('off')
              # ax.set_title(f'\nactual {target.item()}\npredicted {predicted.item()}',fontsize=10)  
              # #plt.imshow(np.transpose(npimg, (1, 2, 0)))
              # #ax.imshow(img.squeeze(), cmap='gray_r')  
              # img = img / 2 + 0.5     # unnormalize
              # npimg = img.numpy()
              # plt.imshow(np.transpose(npimg, (1, 2, 0)))


    def miscImages(model,test_loader,device):
      classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
      wrong_images=[]
      wrong_label=[]
      correct_label=[]
      with torch.no_grad():
          for data, target in test_loader:
              data, target = data["image"].to(device), target.to(device)
              output = model(data)        
              _, predicted = torch.max(output.data, 1)
              wrong_pred = (predicted.eq(target.view_as(predicted)) == False)
              wrong_images.append(data[wrong_pred])
              wrong_label.append(predicted[wrong_pred])
              correct_label.append(target.view_as(predicted)[wrong_pred])  

              wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_label),torch.cat(correct_label)))    
          print(f'Total wrong predictions are {len(wrong_predictions)}')
          
          
          fig = plt.figure(figsize=(15,10))
          fig.tight_layout()
          for i, (img, pred, correct) in enumerate(wrong_predictions[:50]):
            img, pred, target = img.cpu().numpy(), pred.cpu(), correct
            ax = fig.add_subplot(5, 10, i+1)
            ax.axis('off')
            ax.set_title(f'\nactual {classes[target.item()]}\npredicted {classes[pred.item()]}',fontsize=10)
            ax.imshow(img.transpose(1,2,0).squeeze(), cmap='gray_r',vmin=0, vmax=255)      
          plt.show()



