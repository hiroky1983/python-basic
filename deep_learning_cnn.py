import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ディープラーニング（CNN）

# 再現性のために乱数のシードを設定
torch.manual_seed(42)

# ハイパーパラメータ
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
NUM_CLASSES = 10

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 畳み込み層のブロック1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 畳み込み層のブロック2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 全結合層
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # フラット化
        x = self.fc(x)
        return x

class DataManager:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def get_data_loaders(self):
        # 訓練データのロード
        train_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            transform=self.transform,
            download=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # テストデータのロード
        test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            transform=self.transform
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_loader, test_loader

class Trainer:
    def __init__(self, model, learning_rate, device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
    
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        return epoch_loss, epoch_accuracy
    
    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy

class Visualizer:
    @staticmethod
    def plot_results(train_losses, train_accuracies, test_accuracies):
        plt.figure(figsize=(15, 5))
        
        # 損失のプロット
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('訓練損失の推移')
        plt.xlabel('エポック')
        plt.ylabel('損失')
        
        # 精度のプロット
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='訓練精度')
        plt.plot(test_accuracies, label='テスト精度')
        plt.title('精度の推移')
        plt.xlabel('エポック')
        plt.ylabel('精度 (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用デバイス: {device}')
    
    # データの準備
    data_manager = DataManager(BATCH_SIZE)
    train_loader, test_loader = data_manager.get_data_loaders()
    
    # モデルの初期化
    model = CNN().to(device)
    trainer = Trainer(model, LEARNING_RATE, device)
    
    print("訓練を開始します...")
    for epoch in range(EPOCHS):
        print(f'\nエポック {epoch+1}/{EPOCHS}')
        
        # 訓練
        loss, train_acc = trainer.train_epoch(train_loader)
        trainer.train_losses.append(loss)
        trainer.train_accuracies.append(train_acc)
        
        # 評価
        test_acc = trainer.evaluate(test_loader)
        trainer.test_accuracies.append(test_acc)
        
        print(f'訓練損失: {loss:.4f}')
        print(f'訓練精度: {train_acc:.2f}%')
        print(f'テスト精度: {test_acc:.2f}%')
    
    print("\n訓練が完了しました！")
    
    # 結果の可視化
    visualizer = Visualizer()
    visualizer.plot_results(
        trainer.train_losses,
        trainer.train_accuracies,
        trainer.test_accuracies
    )

if __name__ == '__main__':
    main() 