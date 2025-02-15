import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 線形回帰

# 再現性のために乱数のシードを設定
np.random.seed(42)
torch.manual_seed(42)

# サンプルデータの生成
class DataGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        
    def generate_data(self):
        # x値を生成 (0から10の範囲でランダムに100点)
        X = np.random.rand(self.num_samples, 1) * 10
        # y = 2x + 1 + ノイズ の形で目標値を生成
        y = 2 * X + 1 + np.random.randn(self.num_samples, 1) * 1.5
        
        # NumPy配列をPyTorchのテンソルに変換
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        return X_tensor, y_tensor

# 線形回帰モデルの定義
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 入力次元:1, 出力次元:1
    
    def forward(self, x):
        return self.linear(x)

# モデルの学習を行うクラス
class LinearRegressionTrainer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.criterion = nn.MSELoss()  # 平均二乗誤差を損失関数として使用
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
    def train(self, X, y, epochs=1000):
        losses = []
        
        for epoch in range(epochs):
            # 予測
            y_pred = self.model(X)
            
            # 損失の計算
            loss = self.criterion(y_pred, y)
            losses.append(loss.item())
            
            # 勾配をゼロにリセット
            self.optimizer.zero_grad()
            
            # 逆伝播
            loss.backward()
            
            # パラメータの更新
            self.optimizer.step()
            
            # 学習経過の表示
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')
        
        return losses

# 結果を可視化するクラス
class Visualizer:
    @staticmethod
    def plot_results(X, y, model, losses):
        plt.figure(figsize=(12, 5))
        
        # 学習データと予測線のプロット
        plt.subplot(1, 2, 1)
        plt.scatter(X.numpy(), y.numpy(), color='blue', label='データ点')
        
        # モデルの予測線
        with torch.no_grad():
            X_test = torch.linspace(0, 10, 100).reshape(-1, 1)
            y_pred = model(X_test)
            plt.plot(X_test.numpy(), y_pred.numpy(), color='red', label='予測線')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('線形回帰の結果')
        plt.legend()
        
        # 損失の推移をプロット
        plt.subplot(1, 2, 2)
        plt.plot(losses)
        plt.xlabel('エポック')
        plt.ylabel('損失')
        plt.title('損失の推移')
        
        plt.tight_layout()
        plt.show()

def main():
    # データの生成
    data_generator = DataGenerator(num_samples=100)
    X, y = data_generator.generate_data()
    
    # モデルの初期化
    model = LinearRegressionModel()
    
    # トレーナーの初期化と学習の実行
    trainer = LinearRegressionTrainer(model, learning_rate=0.01)
    losses = trainer.train(X, y, epochs=1000)
    
    # 学習後のモデルパラメータの表示
    w, b = model.linear.weight.item(), model.linear.bias.item()
    print(f'\n学習後のパラメータ:')
    print(f'傾き (w): {w:.4f}')
    print(f'切片 (b): {b:.4f}')
    
    # 結果の可視化
    visualizer = Visualizer()
    visualizer.plot_results(X, y, model, losses)

if __name__ == '__main__':
    main() 