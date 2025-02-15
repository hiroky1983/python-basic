from collections import deque
from copy import deepcopy

class HanoiState:
    def __init__(self, pegs):
        self.pegs = pegs
    
    def __eq__(self, other):
        return str(self.pegs) == str(other.pegs)
    
    def __hash__(self):
        return hash(str(self.pegs))
    
    def is_valid_move(self, from_peg, to_peg):
        if not self.pegs[from_peg]:
            return False
        if self.pegs[to_peg] and self.pegs[from_peg][-1] > self.pegs[to_peg][-1]:
            return False
        return True
    
    def move_disk(self, from_peg, to_peg):
        new_state = HanoiState(deepcopy(self.pegs))
        new_state.pegs[to_peg].append(new_state.pegs[from_peg].pop())
        return new_state
    
    def display(self):
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

class HanoiSearchSolver:
    def __init__(self, n_disks):
        self.n_disks = n_disks
        initial_pegs = [list(range(n_disks, 0, -1)), [], []]
        self.initial_state = HanoiState(initial_pegs)
        self.goal_state = HanoiState([[], [], list(range(n_disks, 0, -1))])
    
    def get_next_states(self, state):
        next_states = []
        for from_peg in range(3):
            for to_peg in range(3):
                if from_peg != to_peg and state.is_valid_move(from_peg, to_peg):
                    next_states.append(state.move_disk(from_peg, to_peg))
        return next_states
    
    def bfs_solve(self):
        """幅優先探索による解法"""
        queue = deque([(self.initial_state, [])])
        visited = {self.initial_state}
        
        while queue:
            current_state, path = queue.popleft()
            
            if current_state == self.goal_state:
                return path
            
            for next_state in self.get_next_states(current_state):
                if next_state not in visited:
                    visited.add(next_state)
                    new_path = path + [next_state]
                    queue.append((next_state, new_path))
        
        return None
    
    def dfs_solve(self):
        """深さ優先探索による解法"""
        stack = [(self.initial_state, [])]
        visited = {self.initial_state}
        
        while stack:
            current_state, path = stack.pop()
            
            if current_state == self.goal_state:
                return path
            
            for next_state in self.get_next_states(current_state):
                if next_state not in visited:
                    visited.add(next_state)
                    new_path = path + [next_state]
                    stack.append((next_state, new_path))
        
        return None

def display_solution(path):
    print("解法のステップ:")
    for state in path:
        state.display()

if __name__ == "__main__":
    n_disks = 3
    solver = HanoiSearchSolver(n_disks)
    
    print("=== BFSによる解法 ===")
    bfs_path = solver.bfs_solve()
    if bfs_path:
        print(f"BFSで見つかった解の手順数: {len(bfs_path)}")
        display_solution(bfs_path)
    
    print("\n=== DFSによる解法 ===")
    dfs_path = solver.dfs_solve()
    if dfs_path:
        print(f"DFSで見つかった解の手順数: {len(dfs_path)}")
        display_solution(dfs_path) 