def create_model(model_name, data_directory):
    from __future__ import print_function, division

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    import numpy as np
    import torchvision
    from torchvision import datasets, models, transforms
    import matplotlib.pyplot as plt
    import time
    import os
    import copy
    import sys

    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    plt.ion()   # interactive mode normalization for training

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    image_datasets = {x: datasets.ImageFolder(os.path.join(data_directory, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
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
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
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
        return model


    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    ## training

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)

    torch.save(model_ft.state_dict(), f"models/{model_name}.pt")

def test_model(model_path, data_directory, project_name, model_info, output_csv_path=None):
    ## Imports
    from __future__ import print_function, division

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    import numpy as np
    import torchvision
    from torchvision import datasets, models, transforms
    import matplotlib.pyplot as plt
    import time
    import os
    import copy
    import sys
    import pandas as pd
    from pathlib import Path

    plt.ion()   # interactive mode normalization for training

    ## CONFIG
    ## load model

    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    ## Create custom class to maintain path names

    class ImageFolderWithPaths(datasets.ImageFolder):
        """Custom dataset that includes image file paths. Extends
        torchvision.datasets.ImageFolder
        """

        # override the __getitem__ method. this is the method that dataloader calls
        def __getitem__(self, index):
            # this is what ImageFolder normally returns 
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path
            path = self.imgs[index][0]
            # make a new tuple that includes original and the path
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path

    ## setting up dataloaders

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image_dataset = ImageFolderWithPaths(f"{data_directory}/test", transform)
    dataloader    = torch.utils.data.DataLoader(image_dataset, batch_size=4, num_workers=4)
    dataset_size  = len(image_dataset)
    class_names   = image_dataset.classes

    ## 

    predicteds  = []
    actuals     = []
    img_name    = []
    confidences = []

    with torch.no_grad():
        for inputs, labels, paths in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            # print(outputs)
            vals, preds = torch.max(outputs, 1)
            prediction_confidences = torch.functional.F.softmax(outputs, dim=-1)
            selected_confidences = torch.max(prediction_confidences, 1)
            for j in range(inputs.size()[0]):
                predicteds.append(class_names[preds[j]])
                actuals.append(class_names[labels[j]])
                img_name.append(Path(paths[j]).stem)
                confidences.append(selected_confidences.values[j].item())

    ##

    vals = pd.DataFrame({"predicted": predicteds, "actual": actuals, "image": img_name, "confidence": confidences})

    vals["project"]    = project_name
    vals["model_type"] = model_info

    vals['same'] = vals['predicted'] == vals['actual']
    if output_csv_path:
        vals.to_csv(output_csv_path, index=False)
    else:
        print(vals)
        print(vals.same.mean())
    print("Done")
