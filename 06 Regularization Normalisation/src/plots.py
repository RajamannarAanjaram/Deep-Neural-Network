import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch


class Plots:
    def __init__(self):
        pass

    def sampleVisual(dataset):
        batch = next(iter(dataset))
        images, labels = batch
        batch_grid = make_grid(images)
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        return plt.imshow(batch_grid[0].squeeze(), cmap='gray_r')

    def misclassification(model, test_loader, device):
        wrong_images = []
        wrong_label = []
        correct_label = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)

                wrong_pred = (pred.eq(target.view_as(pred)) == False)
                wrong_images.append(data[wrong_pred])
                wrong_label.append(pred[wrong_pred])
                correct_label.append(target.view_as(pred)[wrong_pred])

                wrong_predictions = list(
                    zip(torch.cat(wrong_images), torch.cat(wrong_label), torch.cat(correct_label)))
            print(f'Total wrong predictions are {len(wrong_predictions)}')

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
        return
