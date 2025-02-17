import numpy as np
from enum import Enum
import copy

class Piece(Enum):
    EMPTY = 0
    PAWN = 1      # 歩
    LANCE = 2     # 香車
    KNIGHT = 3    # 桂馬
    SILVER = 4    # 銀
    GOLD = 5      # 金
    BISHOP = 6    # 角
    ROOK = 7      # 飛車
    KING = 8      # 玉

class Player(Enum):
    BLACK = 1    # 後手（下側）
    WHITE = -1   # 先手（上側）

class ShogiBoard:
    def __init__(self):
        # 9x9の盤面を初期化
        self.board = np.zeros((9, 9), dtype=int)
        self.current_player = Player.BLACK
        self.initialize_board()
    
    def initialize_board(self):
        """盤面の初期配置"""
        # 歩の配置
        self.board[2, :] = Piece.PAWN.value * Player.WHITE.value  # 先手の歩
        self.board[6, :] = Piece.PAWN.value * Player.BLACK.value  # 後手の歩
        
        # その他の駒の配置
        initial_position = [
            Piece.LANCE.value,
            Piece.KNIGHT.value,
            Piece.SILVER.value,
            Piece.GOLD.value,
            Piece.KING.value,
            Piece.GOLD.value,
            Piece.SILVER.value,
            Piece.KNIGHT.value,
            Piece.LANCE.value
        ]
        
        # 先手（上側）と後手（下側）の配置
        self.board[0] = [x * Player.WHITE.value for x in initial_position]
        self.board[8] = [x * Player.BLACK.value for x in initial_position]
        
        # 飛車と角の配置
        self.board[1][1] = Piece.BISHOP.value * Player.WHITE.value
        self.board[1][7] = Piece.ROOK.value * Player.WHITE.value
        self.board[7][1] = Piece.BISHOP.value * Player.BLACK.value
        self.board[7][7] = Piece.ROOK.value * Player.BLACK.value

class ShogiMinimax:
    def __init__(self):
        # 駒の価値
        self.piece_values = {
            Piece.PAWN.value: 100,
            Piece.LANCE.value: 300,
            Piece.KNIGHT.value: 300,
            Piece.SILVER.value: 500,
            Piece.GOLD.value: 600,
            Piece.BISHOP.value: 800,
            Piece.ROOK.value: 1000,
            Piece.KING.value: 15000
        }
    
    def evaluate_position(self, board):
        """盤面の評価関数"""
        score = 0
        for i in range(9):
            for j in range(9):
                piece = board.board[i][j]
                if piece != 0:
                    # 駒の価値を加算（先手は負、後手は正）
                    abs_piece = abs(piece)
                    score += self.piece_values[abs_piece] * np.sign(piece)
        
        return score if board.current_player == Player.BLACK else -score
    
    def get_legal_moves(self, board):
        """合法手の生成"""
        legal_moves = []
        player = board.current_player.value
        
        for i in range(9):
            for j in range(9):
                piece = board.board[i][j]
                if piece * player > 0:  # 自分の駒
                    moves = self._get_piece_moves(board, i, j)
                    legal_moves.extend(moves)
        
        return legal_moves
    
    def _get_piece_moves(self, board, i, j):
        """各駒の移動可能な位置を返す"""
        moves = []
        piece = abs(board.board[i][j])
        player = np.sign(board.board[i][j])
        
        # 歩の移動
        if piece == Piece.PAWN.value:
            if player > 0 and i > 0:  # 後手の歩は上に移動
                moves.append((i, j, i-1, j))
            elif player < 0 and i < 8:  # 先手の歩は下に移動
                moves.append((i, j, i+1, j))
        
        # 玉の移動
        elif piece == Piece.KING.value:
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    new_i, new_j = i + di, j + dj
                    if 0 <= new_i < 9 and 0 <= new_j < 9:
                        if board.board[new_i][new_j] * player <= 0:
                            moves.append((i, j, new_i, new_j))
        
        return moves
    
    def make_move(self, board, move):
        """指定された手を実行"""
        new_board = copy.deepcopy(board)
        from_i, from_j, to_i, to_j = move
        new_board.board[to_i][to_j] = new_board.board[from_i][from_j]
        new_board.board[from_i][from_j] = 0
        new_board.current_player = Player.WHITE if board.current_player == Player.BLACK else Player.BLACK
        return new_board
    
    def minimax(self, board, depth, maximizing_player):
        """ミニマックス法による最善手の探索"""
        # 深さ0または終端ノードの場合、評価値を返す
        if depth == 0:
            return self.evaluate_position(board), None
        
        # 合法手を取得
        legal_moves = self.get_legal_moves(board)
        if not legal_moves:
            return -float('inf') if maximizing_player else float('inf'), None
        
        best_move = None
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                new_board = self.make_move(board, move)
                eval, _ = self.minimax(new_board, depth - 1, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in legal_moves:
                new_board = self.make_move(board, move)
                eval, _ = self.minimax(new_board, depth - 1, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
            return min_eval, best_move

def main():
    # ゲームの初期化
    board = ShogiBoard()
    ai = ShogiMinimax()
    
    def print_board(board):
        """盤面を表示"""
        print("\n現在の盤面:")
        for i in range(9):
            row = []
            for j in range(9):
                piece = board.board[i][j]
                row.append(f"{piece:3}")
            print(row)
    
    # テスト用の対局
    print("対局開始")
    print_board(board)
    
    # ミニマックス法で最善手を計算
    depth = 3  # 探索の深さ
    value, best_move = ai.minimax(board, depth, True)
    
    if best_move:
        from_i, from_j, to_i, to_j = best_move
        print(f"\n選択された移動:")
        print(f"移動元: ({from_i}, {from_j}) の駒: {board.board[from_i][from_j]}")
        print(f"移動先: ({to_i}, {to_j})")
        print(f"評価値: {value}")
        
        # 最善手を実行
        board = ai.make_move(board, best_move)
        print_board(board)
    
    print("\n対局終了")

if __name__ == '__main__':
    main() 