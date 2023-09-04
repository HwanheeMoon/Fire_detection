import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchfusion_utils.metrics import Accuracy
from models.seqModel import SeqClassifier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



classes = ['default','fire','fire increase','smoke','smoke increase']

input_size = 512  # 이미지 특징의 크기
hidden_size = 128  # GRU hidden 크기
num_classes = len(classes)  # 분류할 클래스 수
num_layers = 2  # GRU 레이어 개수

# 모델 생성
final_model = SeqClassifier(input_size, hidden_size, num_classes, num_layers, use_LSTM=True).to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 조정
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 폴더 경로
data_dir = "new data/"

# ImageFolder를 사용하여 데이터셋 생성
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 데이터 로딩을 위한 데이터로더 설정
batch_size = 16
dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=True)



learning_rate = 0.001
num_epochs = 10
loss = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)

print("Training Start...")

train_acc = Accuracy()
max_acc = -9999
min_loss = 1000

for epoch in range(num_epochs):
    now = 0
    correct = 0
    acc = 0.0
    train_acc.reset()
    for inputs, labels in dataloader:
        total = len(dataloader)
        correct = 0
        optimizer.zero_grad()
        outputs = final_model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        now += 1
        train_acc.update(outputs, labels)
        if train_acc.getValue() > max_acc and loss.item() < min_loss:
            max_acc = train_acc.getValue()
            min_loss = loss.item()
            model_path = 'best_train-lstm.pt'
            torch.save(final_model.state_dict(), model_path)
            with open(f'log.txt', 'w') as f:
                f.write(f'model : {model_path[:-3]} best Accuracy : {100* train_acc.getValue():.3f} %, lowest Loss: {loss.item():.4f}')
        print(f"\r epoch {epoch} : {now}/{total} , Accuracy : {100* train_acc.getValue():.3f} %, Loss: {loss.item():.4f}", end=" ")
    print(f"\nepoch {epoch} Accuracy : {100 * train_acc.getValue():.3f} %, Loss: {loss.item():.4f}")

model_path = 'last-lstm.pt'
torch.save(final_model.state_dict(), model_path)
