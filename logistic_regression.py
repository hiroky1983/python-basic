import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 再現性のために乱数のシードを設定
np.random.seed(42)
torch.manual_seed(42)

class DataGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
    
    def generate_data(self):
        """2クラスの分類用データを生成"""
        # クラス0のデータ生成（左下のクラスタ）
        X0 = np.random.randn(self.num_samples // 2, 2) * 1.5
        y0 = np.zeros(self.num_samples // 2)
        
        # クラス1のデータ生成（右上のクラスタ）
        X1 = np.random.randn(self.num_samples // 2, 2) * 1.5 + np.array([4, 4])
        y1 = np.ones(self.num_samples // 2)
        
        # データの結合
        X = np.vstack([X0, X1])
        y = np.hstack([y0, y1])
        
        # NumPy配列をPyTorchのテンソルに変換
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        return X_tensor, y_tensor

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # 入力次元:2, 出力次元:1
        self.sigmoid = nn.Sigmoid()     # シグモイド関数
    
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
    def predict(self, X, threshold=0.5):
        """予測確率を2値クラスに変換"""
        with torch.no_grad():
            probabilities = self(X)
            predictions = (probabilities >= threshold).float()
        return predictions

class LogisticRegressionTrainer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.criterion = nn.BCELoss()  # 二値クロスエントロピー損失
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    def train(self, X, y, epochs=1000):
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            # 予測
            y_pred = self.model(X)
            
            # 損失の計算
            loss = self.criterion(y_pred, y)
            losses.append(loss.item())
            
            # 精度の計算
            accuracy = self.calculate_accuracy(y_pred, y)
            accuracies.append(accuracy)
            
            # 勾配をゼロにリセット
            self.optimizer.zero_grad()
            
            # 逆伝播
            loss.backward()
            
            # パラメータの更新
            self.optimizer.step()
            
            # 学習経過の表示
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
        
        return losses, accuracies
    
    def calculate_accuracy(self, y_pred, y_true):
        """予測精度を計算"""
        predictions = (y_pred >= 0.5).float()
        correct = (predictions == y_true).float().sum()
        accuracy = correct / len(y_true) * 100
        return accuracy.item()

class Visualizer:
    @staticmethod
    def plot_results(X, y, model, losses, accuracies):
        plt.figure(figsize=(15, 5))
        
        # データ点と決定境界のプロット
        plt.subplot(1, 3, 1)
        Visualizer.plot_decision_boundary(X, y, model)
        plt.title('ロジスティック回帰の決定境界')
        
        # 損失の推移をプロット
        plt.subplot(1, 3, 2)
        plt.plot(losses)
        plt.xlabel('エポック')
        plt.ylabel('損失')
        plt.title('損失の推移')
        
        # 精度の推移をプロット
        plt.subplot(1, 3, 3)
        plt.plot(accuracies)
        plt.xlabel('エポック')
        plt.ylabel('精度 (%)')
        plt.title('精度の推移')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_decision_boundary(X, y, model):
        """決定境界を描画"""
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        # メッシュ点での予測
        with torch.no_grad():
            Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
            Z = (Z >= 0.5).float().numpy()
        Z = Z.reshape(xx.shape)
        
        # 決定境界をプロット
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0][y.reshape(-1) == 0], X[:, 1][y.reshape(-1) == 0], 
                   c='blue', label='クラス 0')
        plt.scatter(X[:, 0][y.reshape(-1) == 1], X[:, 1][y.reshape(-1) == 1], 
                   c='red', label='クラス 1')
        plt.xlabel('特徴 1')
        plt.ylabel('特徴 2')
        plt.legend()

def main():
    # データの生成
    data_generator = DataGenerator(num_samples=200)
    X, y = data_generator.generate_data()
    
    # モデルの初期化
    model = LogisticRegressionModel()
    
    # トレーナーの初期化と学習の実行
    trainer = LogisticRegressionTrainer(model, learning_rate=0.1)
    losses, accuracies = trainer.train(X, y, epochs=1000)
    
    # 学習後のモデルパラメータの表示
    w1, w2 = model.linear.weight.data[0].tolist()
    b = model.linear.bias.item()
    print(f'\n学習後のパラメータ:')
    print(f'重み: w1={w1:.4f}, w2={w2:.4f}')
    print(f'バイアス: b={b:.4f}')
    
    # 結果の可視化
    visualizer = Visualizer()
    visualizer.plot_results(X, y, model, losses, accuracies)

if __name__ == '__main__':
    main() 