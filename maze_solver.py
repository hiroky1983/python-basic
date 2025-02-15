from collections import deque
from typing import List, Tuple, Set

class Maze:
    def __init__(self, grid: List[List[str]]):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])
        self.start = self._find_position('S')
        self.goal = self._find_position('G')
        
        # 壁: '#', 通路: '.', スタート: 'S', ゴール: 'G'
        self.wall = '#'
        self.path = '.'
        
    def _find_position(self, char: str) -> Tuple[int, int]:
        """指定された文字の位置を探す"""
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j] == char:
                    return (i, j)
        return None
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """現在位置から移動可能な隣接マスを返す"""
        row, col = pos
        # 上下左右の移動
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        
        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            if (0 <= new_row < self.height and 
                0 <= new_col < self.width and 
                self.grid[new_row][new_col] != self.wall):
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def bfs_solve(self) -> List[Tuple[int, int]]:
        """幅優先探索で最短経路を見つける"""
        queue = deque([(self.start, [self.start])])
        visited = {self.start}
        
        while queue:
            current, path = queue.popleft()
            
            if current == self.goal:
                return path
            
            for next_pos in self.get_neighbors(current):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))
        
        return None
    
    def dfs_solve(self) -> List[Tuple[int, int]]:
        """深さ優先探索で経路を見つける"""
        stack = [(self.start, [self.start])]
        visited = {self.start}
        
        while stack:
            current, path = stack.pop()
            
            if current == self.goal:
                return path
            
            for next_pos in self.get_neighbors(current):
                if next_pos not in visited:
                    visited.add(next_pos)
                    stack.append((next_pos, path + [next_pos]))
        
        return None
    
    def display_solution(self, path: List[Tuple[int, int]]) -> None:
        """解の経路を表示する"""
        if not path:
            print("解が見つかりませんでした。")
            return
        
        # 解の経路をコピーした迷路に表示
        solution_grid = [row[:] for row in self.grid]
        for row, col in path:
            if solution_grid[row][col] not in ['S', 'G']:
                solution_grid[row][col] = '○'  # 経路を○で表示
        
        print("\n迷路の解:")
        for row in solution_grid:
            print(''.join(row))
        print(f"経路の長さ: {len(path)}")

# 使用例
if __name__ == "__main__":
    # 迷路の定義
    maze_grid = [
        ['S', '.', '#', '#', '.', '.'],
        ['.', '.', '.', '#', '.', '#'],
        ['#', '.', '.', '.', '.', '.'],
        ['#', '#', '#', '.', '#', '.'],
        ['.', '.', '.', '.', '#', 'G']
    ]
    
    maze = Maze(maze_grid)
    
    print("=== 元の迷路 ===")
    for row in maze_grid:
        print(''.join(row))
    
    print("\n=== BFSによる解法 ===")
    bfs_path = maze.bfs_solve()
    maze.display_solution(bfs_path)
    
    print("\n=== DFSによる解法 ===")
    dfs_path = maze.dfs_solve()
    maze.display_solution(dfs_path) 