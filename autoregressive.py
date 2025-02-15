import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 自己回帰モデル

# 再現性のために乱数のシードを設定
np.random.seed(42)
torch.manual_seed(42)

class TimeSeriesGenerator:
    def __init__(self, num_points=1000, ar_params=[0.6, -0.3], noise_std=0.1):
        """
        時系列データの生成器
        
        Parameters:
        -----------
        num_points : int
            生成するデータポイントの数
        ar_params : list
            自己回帰パラメータ（例：[0.6, -0.3]はAR(2)モデル）
        noise_std : float
            ノイズの標準偏差
        """
        self.num_points = num_points
        self.ar_params = ar_params
        self.noise_std = noise_std
        self.ar_order = len(ar_params)
    
    def generate_data(self):
        """AR過程に従う時系列データを生成"""
        # 初期値を0で初期化
        data = np.zeros(self.num_points)
        # ノイズを生成
        noise = np.random.normal(0, self.noise_std, self.num_points)
        
        # AR過程でデータを生成
        for t in range(self.ar_order, self.num_points):
            # 過去の値との線形結合
            for i, param in enumerate(self.ar_params):
                data[t] += param * data[t-i-1]
            # ノイズを加える
            data[t] += noise[t]
        
        return data

class ARModel(nn.Module):
    def __init__(self, ar_order):
        """
        自己回帰モデル
        
        Parameters:
        -----------
        ar_order : int
            自己回帰の次数（過去何点を使用するか）
        """
        super(ARModel, self).__init__()
        self.ar_order = ar_order
        # 自己回帰係数を学習するための線形層
        self.linear = nn.Linear(ar_order, 1)
    
    def forward(self, x):
        """
        予測を行う
        x: (batch_size, ar_order)の入力
        """
        return self.linear(x)

class ARTrainer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def prepare_data(self, time_series):
        """時系列データを学習用の入力と出力に変換"""
        X, y = [], []
        for t in range(len(time_series) - self.model.ar_order):
            # 入力: t-ar_order から t-1 までのデータ
            X.append(time_series[t:t+self.model.ar_order])
            # 出力: t時点のデータ
            y.append(time_series[t+self.model.ar_order])
        
        return torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1)
    
    def train(self, time_series, epochs=1000):
        X, y = self.prepare_data(time_series)
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
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
        
        return losses

class Visualizer:
    @staticmethod
    def plot_results(time_series, model, losses):
        plt.figure(figsize=(15, 5))
        
        # 時系列データと予測のプロット
        plt.subplot(1, 2, 1)
        plt.plot(time_series, label='実データ', alpha=0.7)
        
        # モデルによる予測
        trainer = ARTrainer(model)
        X, _ = trainer.prepare_data(time_series)
        with torch.no_grad():
            predictions = model(X).numpy()
        
        # 予測値をプロット（最初のar_order点は予測なし）
        plt.plot(range(model.ar_order, len(time_series)), 
                predictions, label='予測', alpha=0.7)
        plt.xlabel('時間')
        plt.ylabel('値')
        plt.title('時系列データと予測')
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
    generator = TimeSeriesGenerator(num_points=500, ar_params=[0.6, -0.3])
    time_series = generator.generate_data()
    
    # モデルの初期化
    model = ARModel(ar_order=2)
    
    # トレーナーの初期化と学習の実行
    trainer = ARTrainer(model, learning_rate=0.01)
    losses = trainer.train(time_series, epochs=1000)
    
    # 学習後のモデルパラメータの表示
    ar_coefficients = model.linear.weight.data[0].tolist()
    bias = model.linear.bias.item()
    print(f'\n学習後のパラメータ:')
    print(f'AR係数: {ar_coefficients}')
    print(f'バイアス: {bias:.4f}')
    
    # 結果の可視化
    visualizer = Visualizer()
    visualizer.plot_results(time_series, model, losses)

if __name__ == '__main__':
    main() 