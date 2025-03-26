import numpy as np
from enum import Enum

class Player(Enum):
    EMPTY = 0
    X = 1
    O = -1

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = Player.X
        self.game_over = False
        self.winner = None
    
    def make_move(self, row, col):
        """指定された位置に手を打つ"""
        if self.board[row][col] == Player.EMPTY.value:
            self.board[row][col] = self.current_player.value
            if self.check_winner():
                self.game_over = True
                self.winner = self.current_player
            elif self.is_board_full():
                self.game_over = True
            else:
                self.current_player = Player.O if self.current_player == Player.X else Player.X
            return True
        return False
    
    def check_winner(self):
        """勝者がいるかチェック"""
        # 横のチェック
        for i in range(3):
            if abs(sum(self.board[i])) == 3:
                return True
        
        # 縦のチェック
        for j in range(3):
            if abs(sum(self.board[:, j])) == 3:
                return True
        
        # 斜めのチェック
        if abs(sum(np.diag(self.board))) == 3:
            return True
        if abs(sum(np.diag(np.fliplr(self.board)))) == 3:
            return True
        
        return False
    
    def is_board_full(self):
        """盤面が埋まっているかチェック"""
        return not any(Player.EMPTY.value in row for row in self.board)
    
    def get_available_moves(self):
        """利用可能な手を返す"""
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == Player.EMPTY.value:
                    moves.append((i, j))
        return moves

class TicTacToeAI:
    def __init__(self, player):
        self.player = player  # X（先手）またはO（後手）
    
    def evaluate(self, board):
        """盤面の評価"""
        # 横のチェック
        for i in range(3):
            score = sum(board[i])
            if abs(score) == 3:
                return score * self.player.value
        
        # 縦のチェック
        for j in range(3):
            score = sum(board[:, j])
            if abs(score) == 3:
                return score * self.player.value
        
        # 斜めのチェック
        score = sum(np.diag(board))
        if abs(score) == 3:
            return score * self.player.value
        score = sum(np.diag(np.fliplr(board)))
        if abs(score) == 3:
            return score * self.player.value
        
        return 0
    
    def minimax(self, board, depth, maximizing_player):
        """ミニマックス法による最善手の探索"""
        score = self.evaluate(board)
        
        # 終端条件
        if score != 0:
            return score, None
        if depth == 0 or len(self.get_available_moves(board)) == 0:
            return 0, None
        
        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in self.get_available_moves(board):
                new_board = board.copy()
                new_board[move[0]][move[1]] = Player.X.value
                eval, _ = self.minimax(new_board, depth - 1, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in self.get_available_moves(board):
                new_board = board.copy()
                new_board[move[0]][move[1]] = Player.O.value
                eval, _ = self.minimax(new_board, depth - 1, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
            return min_eval, best_move
    
    def get_available_moves(self, board):
        """利用可能な手を返す"""
        moves = []
        for i in range(3):
            for j in range(3):
                if board[i][j] == Player.EMPTY.value:
                    moves.append((i, j))
        return moves
    
    def get_best_move(self, board):
        """最善手を返す"""
        _, best_move = self.minimax(board, 9, self.player == Player.X)
        return best_move

def print_board(board):
    """盤面を表示"""
    print("\n現在の盤面:")
    print("    横0  横1  横2")
    print("  +---+---+---+")
    for i in range(3):
        row = []
        for j in range(3):
            if board[i][j] == Player.EMPTY.value:
                row.append(" ")
            elif board[i][j] == Player.X.value:
                row.append("X")
            else:
                row.append("O")
        print(f"{i} | {' | '.join(row)} |")
        if i < 2:
            print("  +---+---+---+")
    print("  +---+---+---+")

def main():
    game = TicTacToe()
    ai_x = TicTacToeAI(Player.X)  # 先手AI
    ai_o = TicTacToeAI(Player.O)  # 後手AI
    
    print("3目並べ CPU同士の戦闘を開始します！")
    print("X（先手）vs O（後手）")
    print("評価値の説明：")
    print("  +3: 勝利確定")
    print("  +2: 有利な手")
    print("  +1: やや有利な手")
    print("   0: 中立な手")
    print("  -1: やや不利な手")
    print("  -2: 不利な手")
    print("  -3: 敗北確定")
    
    while not game.game_over:
        print_board(game.board)
        
        if game.current_player == Player.X:
            # X（先手）の手番
            print("\nX（先手）の手番です...")
            best_move = ai_x.get_best_move(game.board)
            game.make_move(best_move[0], best_move[1])
            eval_score = ai_x.evaluate(game.board)
            print(f"Xは (縦{best_move[0]}, 横{best_move[1]}) に置きました。")
            print(f"Xの評価値: {eval_score}")
        else:
            # O（後手）の手番
            print("\nO（後手）の手番です...")
            best_move = ai_o.get_best_move(game.board)
            game.make_move(best_move[0], best_move[1])
            eval_score = ai_o.evaluate(game.board)
            print(f"Oは (縦{best_move[0]}, 横{best_move[1]}) に置きました。")
            print(f"Oの評価値: {eval_score}")
    
    print_board(game.board)
    if game.winner:
        if game.winner == Player.X:
            print("\nX（先手）の勝ちです！")
        else:
            print("\nO（後手）の勝ちです！")
    else:
        print("\n引き分けです！")

if __name__ == '__main__':
    main() 