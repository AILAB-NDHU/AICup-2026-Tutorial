import os
from collections import deque

import numpy as np
import pandas as pd
from sgfmill import sgf, boards

class SGFParseRankPrediction:
    """
    Class to parse SGF files and extract features for rank prediction.
    Each feature array has shape (history_length*2, 19, 19)
    2 channels per board state: one for black stones, one for white stones.
    """

    def __init__(self, history_length: int = 250):
        # number of previous board states to include in history
        self.history_length = history_length

    def _prepare_board(self,board:boards.Board):
        board_np = np.zeros((19,19), np.int8)
        for x in range(19):
            for y in range(19):
                # sgfmill keeps rows from bottom to top, columns from left to right
                player = board.get(x,y)
                if player == 'b':
                    board_np[y,x] = 1 # label black stones as 1 on the board
                if player == 'w':
                    board_np[y,x] = -1 # label white stones as -1 on the board
        return board_np

    def _extract_game_states(self, game):
        board = boards.Board(19) # Empty board
        board_states = [] # Board state after every move
        
        for node in game.get_main_sequence():
            move = node.get_move()
            if move[0] is not None: # If there's a move (could be stone placement or pass)
                color = move[0]
                # Check if it's a pass move
                # If move[1] is None, it's a pass, board state doesn't change
                if move[1] is not None:
                    # Place the stone on the board
                    row = move[1][0]
                    column = move[1][1]
                    board.play(row, column, color)
                # Record the board state after the move
                board_states.append(self._prepare_board(board))
            else: 
                # If the board is empty (root node), player black starts first
                board_states.append(self._prepare_board(board))
        return board_states

    def _extract_features(self, board_states):
        features = []
        while len(board_states) < self.history_length: # Pad with empty boards if board states are less than history_length
            board_states.insert(0, np.zeros((19, 19), dtype=np.int8))
        history_states = board_states[-self.history_length:] # Choose the last N states 
        # N*2 features
        for t in history_states:
            features.append((t == 1).astype(np.float32)) # black moves
            features.append((t == -1).astype(np.float32)) # white moves

        features = np.array(features)
        return features

    def features_from_sgf_content(self, game_content: str):
        """Parse SGF content and return a list of feature arrays for every move in the game.
        Args:
            game_content: str -- SGF file content as a string
        Returns:
            features_list: List[np.ndarray] -- each element is shape (1, history_length*2, 19, 19)
            error: Optional[Exception] -- exception if parsing failed, else None
        """
        try:
            game = sgf.Sgf_game.from_string(game_content)
        except Exception as e:
            return None, e

        board_states = self._extract_game_states(game)
        # Extract features within the history length
        features_list = self._extract_features(board_states)
        # Convert to uint8 to save space
        # Later in training, convert back to float32
        return np.array(features_list, dtype=np.uint8), None
    

class SGFParsePlayerIdentification:
    def __init__(self, history_length: int = 8):
        self.history_length = history_length  # Number of previous board states to consider

    def _generate_features(self, history_buffer, player_color):
        """
        Generates the (1+history_buffer*2, 19, 19) feature tensor.
        history_buffer: A deque of (black_stones, white_stones) numpy arrays.
        player_color: 'b' or 'w'
        """
        if player_color == 'b':
            current_player_planes = [s[0] for s in history_buffer]
            opponent_planes = [s[1] for s in history_buffer]
            color_plane = np.ones((19, 19), dtype=np.uint8)
        else: # player_color == 'w'
            current_player_planes = [s[1] for s in history_buffer]
            opponent_planes = [s[0] for s in history_buffer]
            color_plane = np.zeros((19, 19), dtype=np.uint8)

        # Pad with empty planes if history is not full
        while len(current_player_planes) < self.history_length:
            empty_plane = np.zeros((19, 19), dtype=np.uint8)
            current_player_planes.insert(0, empty_plane)
            opponent_planes.insert(0, empty_plane)

        # Stack all planes
        # (8 current, 8 opponent, 1 color)
        feature_stack = current_player_planes + opponent_planes + [color_plane]
        features = np.stack(feature_stack, axis=0)
        return features

    def generate_training_features(self, data:pd.DataFrame, player_to_game_indices:dict, player_name_to_id:dict, features_dir:str="./features"):
        target_players = list(player_name_to_id.keys())
        player_to_game_to_move_count = {player_name_to_id[player]: {} for player in target_players}

        for player_name, game_indices in player_to_game_indices.items():
            player_id = player_name_to_id[player_name]
            print(f"Processing player {player_name} (ID: {player_id}) with {len(game_indices)} games.")
            os.makedirs(f"{features_dir}/{player_id}/", exist_ok=True)
            
            for game_idx in game_indices:
                game_idx = int(game_idx)
                row = data.iloc[game_idx]
                sgf_content = row['sgf_content']
                target_color = row['color'].lower()  # 'b' or 'w'
                feature_filename_prefix = f"{features_dir}/{player_id}/{game_idx}"

                move_number = self.extract_features(
                    sgf_content=sgf_content,
                    target_color=target_color,
                    mode='training',
                    filename=feature_filename_prefix
                )

                # Record total move count for this game for the player
                player_to_game_to_move_count[player_id][game_idx] = move_number
        return player_to_game_to_move_count

    def extract_features(self, sgf_content:str, target_color:str, mode:str='inference', filename:str=None):
        """
        Extract features from SGF content for the specified target player color.
        Args:
            sgf_content: str -- SGF file content as a string
            target_color: str -- 'b' for black or 'w' for white
            mode: str -- 'training' or 'inference'
            filename: str -- base filename to save features if in training mode
        Returns:
            If mode is 'inference':
                List of feature arrays for each move made by the target player.
            If mode is 'training':
                Number of moves processed for the target player.
        """
        game_features = []
        sgf_game = sgf.Sgf_game.from_string(sgf_content)
        board = boards.Board(19)
        history_buffer = deque(maxlen=self.history_length)

        # Replay the game
        moves = sgf_game.get_main_sequence()[1:]  # Skip root node
        move_number = 0
        for node in moves:
            color, move = node.get_move()
            if move is None:  # Pass move
                continue

            # Get current board state
            black_stones = np.zeros((19, 19), dtype=np.uint8)
            white_stones = np.zeros((19, 19), dtype=np.uint8)
            for occupied_point in board.list_occupied_points():
                c, (_row, _col) = occupied_point
                if c == 'b':
                    black_stones[_row, _col] = 1
                elif c == 'w':
                    white_stones[_row, _col] = 1
            history_buffer.append((black_stones, white_stones))

            # If it's a target player's turn, generate features
            if color == target_color and len(history_buffer) >= self.history_length:
                features = self._generate_features(history_buffer, color)
                game_features.append(features)
                if mode == 'training' and filename is not None:
                    # Convert to uint8 for efficient storage
                    features = features.astype(np.uint8)
                    # Save features to .npy file
                    feature_filename = f"{filename}_{move_number}.npy"
                    np.save(feature_filename, features)
                    move_number += 1

            # Make the move on the board
            board.play(move[0], move[1], color)
        
        if mode == 'training':
            return move_number
        if mode == 'inference':
            return game_features
        return None