from collections import defaultdict

from torch import nn
import torchvision.models as models
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler

from torchvision import datasets, models, transforms
import os
import time
import copy
import torch.optim as optim


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    print(count)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def data_loader(batch_size):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(244),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(244),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(244),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = '../../../analyzing-gui-data/data/icons_split_v4'
    # random split

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}

    weights = make_weights_for_balanced_classes(image_datasets['train'].imgs, len(image_datasets['train'].classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                                               sampler=sampler, num_workers=4, pin_memory=True),
                   'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size,
                                                  shuffle=True, num_workers=1),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=1,
                                                  shuffle=True, num_workers=1)}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    print(dataset_sizes)
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, len(class_names)


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, batch, num_epochs=40):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            step = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if step % 100 == 0:
                    print("{}/{}".format(step, dataset_sizes[phase] / batch))
                step += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    if not os.path.exists("models"):
        os.mkdir("models")
    torch.save(model.state_dict(), "models/best_26.pkl")
    return model


def test(model, dataloader, size):
    model.eval()
    running_corrects = 0
    total = 0
    corrects_per_class = defaultdict(int)
    total_per_class = defaultdict(int)
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        for idx, l in enumerate(labels):
            total += 1
            corrects_per_class[l.item()] += 1 if preds[idx].item() == l.item() else 0
            total_per_class[l.item()] += 1
    for key in corrects_per_class:
        print(key, corrects_per_class[key] / total_per_class[key])
    print('avg', running_corrects / size)
    return running_corrects / total


if __name__ == '__main__':
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    batch = 40
    dataloaders, dataset_sizes, num_class = data_loader(batch)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, device, batch)
    test(model, dataloaders['test'], dataset_sizes['test'])
