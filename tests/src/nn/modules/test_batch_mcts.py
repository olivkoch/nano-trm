import pytest
import torch
from src.nn.modules.batch_mcts import BatchMCTS
from src.nn.environments.vectorized_c4_env import VectorizedConnectFour
from src.nn.utils.constants import C4_PLAYER1_CELL, C4_PLAYER2_CELL

"""Unit test for BatchMCTS"""
print("Testing BatchMCTS")
print("=" * 60)

# Create a mock TRM model for testing
class MockTRMModel:
    """Mock TRM model for testing MCTS"""
    def __init__(self, device="cpu"):
        self.device = device
        
        # Create mock base TRM
        self.base_trm = self
        self.puzzle_emb_len = 0
        
        # Simple policy and value heads
        import torch.nn as nn
        self.policy_head = nn.Linear(10, 7).to(device)
        self.value_head = nn.Linear(10, 1).to(device)
    
    def initial_carry(self, batch):
        """Mock initial carry"""
        class MockCarry:
            def __init__(self, batch_size, device):
                self.inner_carry = self
                self.z_H = torch.randn(batch_size, 1, 10, device=device)
        return MockCarry(batch['input'].shape[0], self.device)
    
    def __call__(self, carry, batch):
        """Mock forward pass"""
        return carry, {}

class TestBatchMCTS:
    """Test suite for BatchMCTS"""

    def test_mcts_basic_functionality(self):
        # Test 1: Basic initialization
        print("\nTest 1: Initialization")
        print("-" * 40)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mock_model = MockTRMModel(device=device)

        mcts = BatchMCTS(
            trm_model=mock_model,
            c_puct=1.5,
            num_simulations=10,
            batch_size=4,
            device=device,
            temperature=1.0,
            use_compile=False
        )

        print(f"✓ MCTS initialized with device: {device}")
        print(f"  - Simulations: {mcts.num_simulations}")
        print(f"  - Batch size: {mcts.batch_size}")
        print(f"  - C_PUCT: {mcts.c_puct}")
        assert mcts.num_simulations == 10, "Number of simulations should be 10"
        assert mcts.batch_size == 4, "Batch size should be 4"
        assert mcts.c_puct == 1.5, "C_PUCT should be 1.5"
        print("✓ Initialization parameters are correct")

    def init_mcts(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mock_model = MockTRMModel(device=device)
        mcts = BatchMCTS(
            trm_model=mock_model,
            c_puct=1.5,
            num_simulations=10,
            batch_size=4,
            device=device,
            temperature=1.0,
            use_compile=False
        )
        return mcts, mock_model, device
    
    def test_empty_board_evaluation(self):
        # Test 2: Empty board evaluation
        print("\nTest 2: Empty Board Evaluation")
        print("-" * 40)

        mcts, _, device = self.init_mcts()

        vec_env = VectorizedConnectFour(n_envs=2, device=device)
        states = vec_env.reset()

        probs = mcts.get_action_probabilities(states, temperature=1.0)
        print(f"✓ Action probabilities shape: {probs.shape}")
        print(f"  Environment 0 probs: {probs[0].cpu().numpy()}")
        print(f"  Environment 1 probs: {probs[1].cpu().numpy()}")

        # Verify probabilities sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(2, device=device)), "Probabilities must sum to 1"
        print("✓ Probabilities sum to 1")

    def test_deterministic_action_selection(self):
        # Test 3: Deterministic action selection
        print("\nTest 3: Deterministic Action Selection")
        print("-" * 40)
        mcts, _, device = self.init_mcts()
        vec_env = VectorizedConnectFour(n_envs=2, device=device)
        states = vec_env.reset()

        action = mcts.select_action(states, deterministic=True)
        print(f"✓ Selected action (deterministic): {action}")
        assert 0 <= action <= 6, "Action must be valid column"

    def test_varied_board_states(self):
        # Test 4: Partially filled board
        print("\nTest 4: Partially Filled Board")
        print("-" * 40)
        mcts, _, device = self.init_mcts()
        # Create single environment for this test
        vec_env_single = VectorizedConnectFour(n_envs=1, device=device)
        vec_env_single.reset()

        # Make some moves
        vec_env_single.boards[0, 5, 3] = C4_PLAYER1_CELL
        vec_env_single.boards[0, 5, 4] = C4_PLAYER2_CELL
        vec_env_single.boards[0, 4, 3] = C4_PLAYER2_CELL
        vec_env_single.current_players[0] = 1

        states_single = vec_env_single.get_state()
        probs = mcts.get_action_probabilities(states_single, temperature=0.5)
        print(f"✓ Probabilities for partially filled board: {probs.cpu().numpy()}")

    def test_win_detection(self):
        # Test 5: Win detection
        print("\nTest 5: Near-Win Position")
        print("-" * 40)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Set up a position where player 1 can win
        vec_env_win = VectorizedConnectFour(n_envs=1, device=device)
        vec_env_win.reset()
        vec_env_win.boards[0, 5, 0] = C4_PLAYER1_CELL
        vec_env_win.boards[0, 5, 1] = C4_PLAYER1_CELL
        vec_env_win.boards[0, 5, 2] = C4_PLAYER1_CELL
        vec_env_win.boards[0, 5, 4] = C4_PLAYER2_CELL
        vec_env_win.boards[0, 5, 5] = C4_PLAYER2_CELL
        vec_env_win.boards[0, 5, 6] = C4_PLAYER2_CELL
        vec_env_win.current_players[0] = 1

        states_win = vec_env_win.get_state()

        # Render the board
        print("Board state:")
        print(vec_env_win.render(0))

        _, mock_model, device = self.init_mcts()

        # Get action with high simulation count for better accuracy
        high_sim_mcts = BatchMCTS(
            trm_model=mock_model,
            c_puct=1.5,
            num_simulations=50,
            batch_size=8,
            device=device,
            temperature=0.1,
            use_compile=False
        )

        probs = high_sim_mcts.get_action_probabilities(states_win, temperature=0)
        winning_move = torch.argmax(probs).item()
        print(f"✓ MCTS suggests move: {winning_move}")
        print(f"  Probabilities: {probs.cpu().numpy()}")
        assert winning_move == 3, "MCTS should suggest winning move in column 3"

    def test_illegal_move_handling(self):
        # Test 6: Illegal move handling
        print("\nTest 6: Illegal Move Handling")
        print("-" * 40)

        # Fill a column
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vec_env_illegal = VectorizedConnectFour(n_envs=1, device=device)
        vec_env_illegal.reset()
        for row in range(6):
            vec_env_illegal.boards[0, row, 2] = C4_PLAYER1_CELL if row % 2 == 0 else C4_PLAYER2_CELL

        states_illegal = vec_env_illegal.get_state()
        print(f"Legal moves: {states_illegal.legal_moves[0].cpu().numpy()}")

        mcts, _, _ = self.init_mcts()

        probs = mcts.get_action_probabilities(states_illegal, temperature=1.0)
        print(f"✓ Probabilities with filled column: {probs.cpu().numpy()}")

        # Verify illegal move has 0 probability
        assert abs(probs[2].item()) < 1e-6, "Filled column should have 0 probability"
        print("✓ Illegal move has 0 probability")

    def test_parallel_game_evaluation(self):
        # Test 7: Multiple parallel games
        print("\nTest 7: Parallel Game Evaluation")
        print("-" * 40)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vec_env_parallel = VectorizedConnectFour(n_envs=4, device=device)
        states_parallel = vec_env_parallel.reset()

        # Make different moves in each environment
        vec_env_parallel.boards[0, 5, 0] = C4_PLAYER1_CELL
        vec_env_parallel.boards[1, 5, 3] = C4_PLAYER1_CELL
        vec_env_parallel.boards[2, 5, 6] = C4_PLAYER1_CELL
        vec_env_parallel.boards[3, 5, 2] = C4_PLAYER1_CELL

        states_parallel = vec_env_parallel.get_state()

        mcts, _, _ = self.init_mcts()
        
        # Evaluate all 4 games in parallel
        all_probs = mcts.get_action_probabilities(states_parallel, temperature=1.0)
        print(f"✓ Evaluated {all_probs.shape[0]} games in parallel")

        for i in range(4):
            print(f"  Game {i}: max prob action = {torch.argmax(all_probs[i]).item()}")
        assert all_probs.shape[0] == 4, "Should evaluate 4 games in parallel"

    def test_terminal_state_handling(self):
        # Test 8: Terminal state handling
        print("\nTest 8: Terminal State Handling")
        print("-" * 40)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vec_env_terminal = VectorizedConnectFour(n_envs=1, device=device)
        vec_env_terminal.reset()

        # Create a won position
        for i in range(4):
            vec_env_terminal.boards[0, 5, i] = C4_PLAYER1_CELL
        vec_env_terminal._check_winners_batch()
        vec_env_terminal.is_terminal[0] = True
        vec_env_terminal.winners[0] = 1

        states_terminal = vec_env_terminal.get_state()
        print(f"Is terminal: {states_terminal.is_terminal[0]}")
        print(f"Winner: {states_terminal.winners[0]}")

        mcts, _, _ = self.init_mcts()

        # MCTS should still return some probabilities even for terminal state
        probs = mcts.get_action_probabilities(states_terminal, temperature=1.0)
        print(f"✓ Probabilities for terminal state: {probs.cpu().numpy()}")
        assert torch.allclose(probs.sum(), torch.tensor(1.0)), "Probabilities must sum to 1 even in terminal state"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])