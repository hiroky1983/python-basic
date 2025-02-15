import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ハイパーパラメータの設定
BATCH_SIZE = 100
LEARNING_RATE = 0.001
EPOCHS = 5

# ニューラルネットワークのモデル定義
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 入力層(28x28=784) → 隠れ層(128) → 出力層(10)の単純なニューラルネットワーク
        self.flatten = nn.Flatten()  # 2次元画像を1次元に変換
        self.fc1 = nn.Linear(28 * 28, 128)  # 全結合層1
        self.relu = nn.ReLU()  # 活性化関数
        self.fc2 = nn.Linear(128, 10)  # 全結合層2（出力層）
    
    def forward(self, x):
        # データの流れを定義
        x = self.flatten(x)  # 画像を1次元に変換
        x = self.fc1(x)      # 第1層で変換
        x = self.relu(x)     # 活性化関数を適用
        x = self.fc2(x)      # 最終的な出力（10クラス）
        return x

# データの前処理とローダーの設定
def get_data_loaders():
    # 画像データを正規化する変換を定義
    transform = transforms.Compose([
        transforms.ToTensor(),  # PIL画像をTensorに変換
        transforms.Normalize((0.1307,), (0.3081,))  # MNISTの平均と標準偏差で正規化
    ])
    
    # 訓練データのロード
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        transform=transform,
        download=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # テストデータのロード
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    return train_loader, test_loader

# モデルの訓練関数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()  # 訓練モードに設定
    for batch_idx, (data, target) in enumerate(train_loader):
        # データをGPUに転送（利用可能な場合）
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()  # 勾配をリセット
        output = model(data)   # 順伝播
        loss = criterion(output, target)  # 損失を計算
        loss.backward()        # 逆伝播
        optimizer.step()       # パラメータの更新
        
        # 進捗表示
        if batch_idx % 100 == 0:
            print(f'Training: [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.6f}')

# モデルの評価関数
def evaluate_model(model, test_loader, device):
    model.eval()  # 評価モードに設定
    correct = 0
    total = 0
    
    with torch.no_grad():  # 勾配計算を無効化
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'テストデータの精度: {accuracy:.2f}%')
    return accuracy

def main():
    # デバイスの設定（GPUが利用可能な場合はGPUを使用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用デバイス: {device}')
    
    # データローダーの取得
    train_loader, test_loader = get_data_loaders()
    
    # モデル、損失関数、オプティマイザーの設定
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 訓練ループ
    print("訓練を開始します...")
    for epoch in range(EPOCHS):
        print(f'\nエポック {epoch+1}/{EPOCHS}')
        train_model(model, train_loader, criterion, optimizer, device)
        evaluate_model(model, test_loader, device)
    
    print("\n訓練が完了しました！")

if __name__ == '__main__':
    main() 