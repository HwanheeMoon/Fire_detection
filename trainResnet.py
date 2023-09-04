from torchfusion_utils.fp16 import convertToFP16
from torchfusion_utils.metrics import Accuracy
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


classes = ['fire','neutral','smoke']
transforms_train = transforms.Compose([transforms.Resize(225),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])

transforms_test = transforms.Compose([transforms.Resize(225),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])

batch_sizes = 64
test_data_dir = './FIRE-SMOKE-DATASET/Test'
train_data_dir = './FIRE-SMOKE-DATASET/Train'

train_data = datasets.ImageFolder(root=train_data_dir, transform=transforms_train)
test_data = datasets.ImageFolder(root=test_data_dir, transform=transforms_test)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_sizes, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_sizes, shuffle=True)

images, labels = next(iter(train_data_loader))


def image_display(image, title=None):
    image = image / 2 + 0.5
    numpy_image = image.numpy()
    transposed_numpy_image = np.transpose(numpy_image, (1, 2, 0))
    plt.figure(figsize=(20, 4))
    plt.imshow(transposed_numpy_image)
    plt.yticks([])
    plt.xticks([])
    if title:
        plt.title(title)
    plt.show


image_display(torchvision.utils.make_grid(images))




ResNet = models.resnet18(num_classes=3)
Model = ResNet

Model = Model.to(device)

lr = 0.001

criteria = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(Model.parameters(), lr=lr)

Model,optimizer = convertToFP16(Model, optimizer)

milestones = [100, 150]

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)


def model_traing_and_validation_loop(Model, n_epochs):
    n_epochs = n_epochs

    saving_criteria_of_model = 0

    training_loss_array = []

    validation_loss_array = []

    train_acc = Accuracy()

    validation_acc = Accuracy(topK=1)

    for i in range(n_epochs):

        total_test_data = 0

        total_train_data = 0

        correct_test_data = 0

        training_loss = 0

        validation_loss = 0

        train_acc.reset()
        now = 0
        for data, target in train_data_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            predictions = Model(data)

            loss = criteria(predictions, target)

            optimizer.backward(loss)

            optimizer.step()

            training_loss += loss.item() * data.size(0)

            train_acc.update(predictions, target)
            now += 1
            print(f'\rEpoch (train) {i} : {now} / {len(train_data_loader)}',end='')
        scheduler.step()
        now = 0
        print('')
        with torch.no_grad():

            validation_acc.reset()

            for data, target in test_data_loader:
                data, target = data.to(device), target.to(device)

                predictions = Model(data)

                loss = criteria(predictions, target)

                validation_acc.update(predictions, target)

                total_test_data += target.size(0)

                validation_loss += loss.item() * data.size(0)
                now += 1
                print(f'\rEpoch (val) {i} : {now} / {len(test_data_loader)}', end='')
        training_loss = training_loss / len(train_data)

        validation_loss = validation_loss / total_test_data

        training_loss_array.append(training_loss)

        validation_loss_array.append(validation_loss)

        print(f'\n{i + 1} / {n_epochs} Training loss: {training_loss:.4f}, Tran_Accuracy: {100 * train_acc.getValue():.4f} % , Validation_loss: {validation_loss:.4f}, Validation_Accuracy: {100* validation_acc.getValue():.4f} %')

        if saving_criteria_of_model < validation_acc.getValue():
            torch.save(Model.state_dict(), 'model2.pt')

            saving_criteria_of_model = validation_acc.getValue()

            print('--------------------------Saving Model---------------------------')

    plt.figure(figsize=(20, 4))

    x_axis = (range(n_epochs))

    plt.plot(x_axis, training_loss_array, 'r', validation_loss_array, 'b')

    plt.title('A gragh of training loss vs validation loss')

    plt.legend(['train loss', 'validation loss'])

    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')

    return Model
model_traing_and_validation_loop(Model,100)
