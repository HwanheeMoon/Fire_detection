import torchvision.models as models
import torch.nn as nn
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cnn = models.resnet18(pretrained=True).to(device)
cnn.load_state_dict(torch.load('model2.pt'),strict=False)
cnn = nn.Sequential(*list(cnn.children())[:-1])  # 마지막 fully connected layer 제거
cnn.eval()

class SeqClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, use_LSTM = False):
        super(SeqClassifier, self).__init__()

        if use_LSTM:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 이미지 특징 추출
        with torch.no_grad():
            features = cnn(x)

        # RNN에 입력하기 위해 특징들을 시퀀스로 변환
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, channels, height * width).permute(0, 2, 1)

        # RNN 입력
        rnn_output, _ = self.rnn(features)

        # LSTM의 마지막 출력을 사용하여 분류
        output = self.fc(rnn_output[:, -1, :])

        return output