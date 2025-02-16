import numpy as np
from enum import Enum
import copy

# 将棋AI

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
    PROMOTED_PAWN = 9     # と金
    PROMOTED_LANCE = 10   # 成香
    PROMOTED_KNIGHT = 11  # 成桂
    PROMOTED_SILVER = 12  # 成銀
    PROMOTED_BISHOP = 13  # 馬
    PROMOTED_ROOK = 14    # 龍

class Player(Enum):
    BLACK = 1
    WHITE = -1

class ShogiBoard:
    def __init__(self):
        # 9x9の盤面を初期化
        self.board = np.zeros((9, 9), dtype=int)
        self.current_player = Player.BLACK
        self.initialize_board()
    
    def initialize_board(self):
        """盤面の初期配置"""
        # 歩の配置（先手は負の値、後手は正の値）
        self.board[2, :] = Piece.PAWN.value * Player.WHITE.value  # 先手（上側）の歩
        self.board[6, :] = Piece.PAWN.value * Player.BLACK.value  # 後手（下側）の歩
        
        # その他の駒の配置（簡略化のため主要な駒のみ）
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
        
        # 先手（上側、負の値）と後手（下側、正の値）の配置
        self.board[0] = [x * Player.WHITE.value for x in initial_position]  # 先手
        self.board[8] = [x * Player.BLACK.value for x in initial_position]  # 後手
        
        # 飛車と角の配置
        self.board[1][1] = Piece.BISHOP.value * Player.WHITE.value  # 先手の角
        self.board[1][7] = Piece.ROOK.value * Player.WHITE.value   # 先手の飛車
        self.board[7][1] = Piece.BISHOP.value * Player.BLACK.value  # 後手の角
        self.board[7][7] = Piece.ROOK.value * Player.BLACK.value   # 後手の飛車

class ShogiAI:
    def __init__(self):
        # 駒の価値（評価関数で使用）
        self.piece_values = {
            Piece.PAWN.value: 100,
            Piece.LANCE.value: 300,
            Piece.KNIGHT.value: 300,
            Piece.SILVER.value: 500,
            Piece.GOLD.value: 600,
            Piece.BISHOP.value: 800,
            Piece.ROOK.value: 1000,
            Piece.KING.value: 15000,
            Piece.PROMOTED_PAWN.value: 600,
            Piece.PROMOTED_LANCE.value: 600,
            Piece.PROMOTED_KNIGHT.value: 600,
            Piece.PROMOTED_SILVER.value: 600,
            Piece.PROMOTED_BISHOP.value: 1100,
            Piece.PROMOTED_ROOK.value: 1300
        }
        # トランスポジションテーブルの初期化
        self.transposition_table = {}
    
    def evaluate_position(self, board):
        """改善された盤面の評価関数"""
        score = 0
        
        # 1. 駒の価値による基本評価
        material_score = 0
        # 2. 玉の安全性評価
        king_safety_score = 0
        # 3. 駒の利きの評価
        control_score = 0
        
        for i in range(9):
            for j in range(9):
                piece = board.board[i][j]
                if piece != 0:
                    # 駒の基本価値
                    abs_piece = abs(piece)
                    multiplier = 1 if piece > 0 else -1
                    material_score += self.piece_values[abs_piece] * multiplier
                    
                    # 玉の安全性評価
                    if abs_piece == Piece.KING.value:
                        king_safety_score += self._evaluate_king_safety(board, i, j) * multiplier
                    
                    # 駒の利きの評価
                    control_score += self._evaluate_piece_control(board, i, j)
        
        # 各要素の重み付け
        score = (
            material_score * 1.0 +
            king_safety_score * 1.5 +
            control_score * 0.8
        )
        
        return score if board.current_player == Player.BLACK else -score
    
    def _evaluate_king_safety(self, board, king_i, king_j):
        """玉の安全性を評価"""
        safety_score = 0
        player = np.sign(board.board[king_i][king_j])
        
        # 玉の周囲8マスの評価
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                new_i, new_j = king_i + di, king_j + dj
                if 0 <= new_i < 9 and 0 <= new_j < 9:
                    piece = board.board[new_i][new_j]
                    if piece * player > 0:  # 味方の駒
                        safety_score += 50
        
        return safety_score
    
    def _evaluate_piece_control(self, board, i, j):
        """駒の利きを評価"""
        control_score = 0
        piece = board.board[i][j]
        if piece != 0:
            moves = self._get_piece_moves(board, i, j)
            control_score += len(moves) * 10  # 利きの数に応じてスコアを加算
        return control_score
    
    def get_legal_moves(self, board):
        """合法手の生成（簡略化）"""
        legal_moves = []
        player = board.current_player.value
        
        print(f"\n現在のプレイヤー: {'黒' if player == Player.BLACK.value else '白'}")
        for i in range(9):
            for j in range(9):
                piece = board.board[i][j]
                if piece * player > 0:  # 自分の駒
                    # 各駒の移動可能な位置を追加（簡略化）
                    moves = self._get_piece_moves(board, i, j)
                    if moves:
                        print(f"駒の位置 ({i}, {j}) の移動可能な手: {moves}")
                    legal_moves.extend(moves)
        
        return legal_moves
    
    def _get_piece_moves(self, board, i, j):
        """各駒の移動可能な位置を返す（簡略化）"""
        moves = []
        piece = abs(board.board[i][j])
        player = np.sign(board.board[i][j])  # プレイヤーは駒の符号から判断
        
        # 各駒の移動パターン（簡略化）
        if piece == Piece.PAWN.value:
            # 先手（負の値）は下に、後手（正の値）は上に移動
            if player > 0 and i > 0:  # 後手の歩は上に移動
                moves.append((i, j, i-1, j))
            elif player < 0 and i < 8:  # 先手の歩は下に移動
                moves.append((i, j, i+1, j))
        
        elif piece == Piece.KING.value:
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    new_i, new_j = i + di, j + dj
                    if 0 <= new_i < 9 and 0 <= new_j < 9:
                        if board.board[new_i][new_j] * player <= 0:  # 自分の駒でない場所に移動可能
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
    
    def alpha_beta_search(self, board, depth, alpha=-float('inf'), beta=float('inf'), maximizing=True):
        """アルファベータ探索による最善手の探索"""
        if depth == 0:
            return self.evaluate_position(board), None
        
        legal_moves = self.get_legal_moves(board)
        if not legal_moves:
            return -float('inf') if maximizing else float('inf'), None
        
        best_move = None
        
        if maximizing:
            value = -float('inf')
            for move in legal_moves:
                new_board = self.make_move(board, move)
                new_value, _ = self.alpha_beta_search(new_board, depth-1, alpha, beta, False)
                if new_value > value:
                    value = new_value
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, best_move
        else:
            value = float('inf')
            for move in legal_moves:
                new_board = self.make_move(board, move)
                new_value, _ = self.alpha_beta_search(new_board, depth-1, alpha, beta, True)
                if new_value < value:
                    value = new_value
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, best_move
    
    def iterative_deepening_search(self, board, max_depth, time_limit=5.0):
        """反復深化探索"""
        import time
        start_time = time.time()
        best_move = None
        best_value = -float('inf')
        
        for depth in range(1, max_depth + 1):
            if time.time() - start_time > time_limit:
                break
                
            value, move = self.alpha_beta_search(
                board, 
                depth,
                -float('inf'),
                float('inf'),
                True
            )
            
            if move:
                best_move = move
                best_value = value
        
        return best_value, best_move

def main():
    # ゲームの初期化
    board = ShogiBoard()
    ai = ShogiAI()
    
    def print_board(board):
        """盤面を表示"""
        print("\n現在の盤面:")
        for i in range(9):
            row = []
            for j in range(9):
                piece = board.board[i][j]
                row.append(f"{piece:3}")
            print(row)
    
    # 自動対局のメインループ
    max_moves = 100  # 最大手数
    move_count = 0
    
    print("対局開始")
    print_board(board)
    
    while move_count < max_moves:
        current_player = "黒" if board.current_player == Player.BLACK else "白"
        print(f"\n手番: {current_player}")
        
        # AIに最善手を計算させる
        value, best_move = ai.iterative_deepening_search(board, max_depth=3, time_limit=3.0)
        
        if best_move:
            from_i, from_j, to_i, to_j = best_move
            print(f"選択された移動:")
            print(f"移動元: ({from_i}, {from_j}) の駒: {board.board[from_i][from_j]}")
            print(f"移動先: ({to_i}, {to_j})")
            print(f"評価値: {value}")
            
            # 最善手を実行
            board = ai.make_move(board, best_move)
            print_board(board)
            
            move_count += 1
        else:
            print(f"{current_player}番、合法手なし")
            break
        
        # 王が取られたかチェック
        king_exists = {Player.BLACK: False, Player.WHITE: False}
        for i in range(9):
            for j in range(9):
                piece = board.board[i][j]
                if abs(piece) == Piece.KING.value:
                    player = Player.BLACK if piece > 0 else Player.WHITE
                    king_exists[player] = True
        
        if not king_exists[Player.BLACK]:
            print("\n白の勝ち")
            break
        elif not king_exists[Player.WHITE]:
            print("\n黒の勝ち")
            break
    
    if move_count >= max_moves:
        print("\n最大手数に到達しました")
    
    print("対局終了")

if __name__ == '__main__':
    main() 