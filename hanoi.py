class HanoiTower:
    def __init__(self, n_disks):
        self.n_disks = n_disks
        # 3本の棒を表現するリスト
        self.pegs = [[], [], []]
        # 最初の棒に円盤を大きい順に配置
        self.pegs[0] = list(range(n_disks, 0, -1))
    
    def move_disk(self, from_peg, to_peg):
        """円盤を1つ移動する"""
        if not self.pegs[from_peg]:
            return False
        if self.pegs[to_peg] and self.pegs[from_peg][-1] > self.pegs[to_peg][-1]:
            return False
        self.pegs[to_peg].append(self.pegs[from_peg].pop())
        return True
    
    def solve(self, n=None, source=0, auxiliary=1, target=2):
        """再帰的にハノイの塔を解く"""
        if n is None:
            n = self.n_disks
        
        if n == 1:
            self.move_disk(source, target)
            self.display()
            return
        
        # n-1個の円盤を補助の棒に移動
        self.solve(n-1, source, target, auxiliary)
        # 最大の円盤を目標の棒に移動
        self.move_disk(source, target)
        self.display()
        # n-1個の円盤を目標の棒に移動
        self.solve(n-1, auxiliary, source, target)
    
    def display(self):
        """現在の状態を表示"""
        print("\n現在の状態:")
        max_height = max(len(peg) for peg in self.pegs)
        
        for level in range(max_height - 1, -1, -1):
            for peg in self.pegs:
                if level < len(peg):
                    print(f"[{peg[level]}]", end=" ")
                else:
                    print("| ", end=" ")
            print()
        print("-" * 20)

# 使用例
if __name__ == "__main__":
    # 3枚の円盤でハノイの塔を作成
    tower = HanoiTower(3)
    print("初期状態:")
    tower.display()
    print("\n解を求めています...")
    tower.solve() 