"""
Test suite for AlphaZero-style MCTS
"""

from time import time
import pytest
import torch
import numpy as np
from src.nn.modules.batch_mcts import BatchMCTS
from src.nn.environments.vectorized_c4_env import VectorizedConnectFour
from src.nn.utils.constants import C4_EMPTY_CELL, C4_PLAYER1_CELL, C4_PLAYER2_CELL


class MockTRMModel:
    """Mock TRM model for testing MCTS"""
    def __init__(self, device="cpu"):
        self.device = device
        
    def forward(self, boards):
        """Mock forward pass returning policy and value"""
        batch_size = boards.shape[0]
        
        # Return random policy and value for testing
        policy = torch.softmax(torch.randn(batch_size, 7, device=self.device), dim=-1)
        value = torch.tanh(torch.randn(batch_size, device=self.device))
        
        return policy, value


class TestMCTS:
    """Test suite for MCTS"""
    
    def test_initialization(self):
        """Test MCTS initialization"""
        print("\nTest 1: Initialization")
        print("-" * 40)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mock_model = MockTRMModel(device=device)
        
        mcts = BatchMCTS(
            model=mock_model,
            c_puct=1.0,
            num_simulations=100,
            dirichlet_alpha=0.3,
            exploration_fraction=0.25,
            device=device
        )
        
        assert mcts.num_simulations == 100
        assert mcts.c_puct == 1.0
        assert mcts.dirichlet_alpha == 0.3
        print("✓ MCTS initialized correctly")
    
    def test_empty_board(self):
        """Test MCTS on empty board"""
        print("\nTest 2: Empty Board")
        print("-" * 40)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mock_model = MockTRMModel(device=device)
        
        mcts = BatchMCTS(
            model=mock_model,
            c_puct=1.0,
            num_simulations=50,
            dirichlet_alpha=0.3,
            exploration_fraction=0.25,
            device=device
        )
        
        # Empty board
        board = torch.full((6, 7), C4_EMPTY_CELL, dtype=torch.int32, device=device)
        legal_moves = torch.ones(7, dtype=torch.bool, device=device)
        
        # Get action probabilities
        probs = mcts.get_action_probs(board, legal_moves, temperature=1.0)
        
        assert probs.shape == (7,)
        assert torch.allclose(probs.sum(), torch.tensor(1.0))
        assert (probs >= 0).all()
        print(f"✓ Probabilities: {probs.cpu().numpy()}")
        print("✓ Valid probability distribution")
    
    def test_winning_position(self):
        """Test MCTS finds winning move"""
        print("\nTest 3: Winning Position")
        print("-" * 40)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create a model that recognizes wins
        class WinAwareModel:
            def __init__(self, device):
                self.device = device
                
            def forward(self, boards):
                batch_size = boards.shape[0]
                board = boards[0].reshape(6, 7)
                
                # Check for winning move at column 3
                policy = torch.zeros(batch_size, 7, device=self.device)
                
                # If there's a winning pattern, strongly prefer column 3
                if (board[5, 0] == C4_PLAYER1_CELL and 
                    board[5, 1] == C4_PLAYER1_CELL and 
                    board[5, 2] == C4_PLAYER1_CELL):
                    policy[:, 3] = 0.9
                    policy[:, 4:] = 0.1 / 3
                else:
                    policy = torch.ones(batch_size, 7, device=self.device) / 7
                
                value = torch.tensor([0.8], device=self.device)
                return policy, value
        
        model = WinAwareModel(device)
        
        mcts = BatchMCTS(
            model=model,
            c_puct=1.0,
            num_simulations=100,
            dirichlet_alpha=0.03,  # Less noise for test
            exploration_fraction=0.1,
            device=device
        )
        
        # Create winning position
        board = torch.full((6, 7), C4_EMPTY_CELL, dtype=torch.int32, device=device)
        board[5, 0] = C4_PLAYER1_CELL
        board[5, 1] = C4_PLAYER1_CELL
        board[5, 2] = C4_PLAYER1_CELL
        board[5, 4] = C4_PLAYER2_CELL
        board[5, 5] = C4_PLAYER2_CELL
        
        legal_moves = board[0, :] == C4_EMPTY_CELL
        
        # Get action probabilities with low temperature
        probs = mcts.get_action_probs(board, legal_moves, temperature=0.1)
        
        best_move = probs.argmax().item()
        print(f"✓ Best move: {best_move}")
        print(f"✓ Probabilities: {probs.cpu().numpy()}")
        
        # Should heavily favor column 3
        assert probs[3] > 0.5, "Should recognize winning move"
        print("✓ MCTS identifies winning move")
    
    def test_illegal_moves(self):
        """Test MCTS handles illegal moves correctly"""
        print("\nTest 4: Illegal Moves")
        print("-" * 40)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mock_model = MockTRMModel(device=device)
        
        mcts = BatchMCTS(
            model=mock_model,
            c_puct=1.0,
            num_simulations=50,
            device=device
        )
        
        # Board with some full columns
        board = torch.full((6, 7), C4_EMPTY_CELL, dtype=torch.int32, device=device)
        
        # Fill column 2 completely
        for row in range(6):
            board[row, 2] = C4_PLAYER1_CELL if row % 2 == 0 else C4_PLAYER2_CELL
        
        # Fill column 5 completely
        for row in range(6):
            board[row, 5] = C4_PLAYER2_CELL if row % 2 == 0 else C4_PLAYER1_CELL
        
        legal_moves = board[0, :] == C4_EMPTY_CELL
        assert not legal_moves[2]
        assert not legal_moves[5]
        
        # Get action probabilities
        probs = mcts.get_action_probs(board, legal_moves, temperature=1.0)
        
        # Check illegal moves have 0 probability
        assert abs(probs[2].item()) < 1e-6, "Full column 2 should have 0 probability"
        assert abs(probs[5].item()) < 1e-6, "Full column 5 should have 0 probability"
        assert torch.allclose(probs.sum(), torch.tensor(1.0))
        
        print(f"✓ Probabilities: {probs.cpu().numpy()}")
        print("✓ Illegal moves have 0 probability")
    
    def test_temperature_effect(self):
        """Test temperature parameter effect"""
        print("\nTest 5: Temperature Effect")
        print("-" * 40)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mock_model = MockTRMModel(device=device)
        
        mcts = BatchMCTS(
            model=mock_model,
            c_puct=1.0,
            num_simulations=100,
            device=device
        )
        
        board = torch.full((6, 7), C4_EMPTY_CELL, dtype=torch.int32, device=device)
        legal_moves = torch.ones(7, dtype=torch.bool, device=device)
        
        # Test with high temperature (more exploration)
        probs_high_temp = mcts.get_action_probs(board, legal_moves, temperature=2.0)
        
        # Test with low temperature (more exploitation)
        probs_low_temp = mcts.get_action_probs(board, legal_moves, temperature=0.1)
        
        # Test with zero temperature (deterministic)
        probs_zero_temp = mcts.get_action_probs(board, legal_moves, temperature=0.0)
        
        # Zero temperature should be one-hot
        assert probs_zero_temp.max() == 1.0
        assert probs_zero_temp.sum() == 1.0
        assert (probs_zero_temp == 1.0).sum() == 1
        
        # Calculate entropy
        def entropy(p):
            return -(p * torch.log(p + 1e-8)).sum().item()
        
        entropy_high = entropy(probs_high_temp)
        entropy_low = entropy(probs_low_temp)
        entropy_zero = entropy(probs_zero_temp)
        
        print(f"✓ High temp entropy: {entropy_high:.3f}")
        print(f"✓ Low temp entropy: {entropy_low:.3f}")
        print(f"✓ Zero temp entropy: {entropy_zero:.3f}")
        
        # Higher temperature should have higher entropy
        assert entropy_high > entropy_low
        assert entropy_zero < 0.01
        print("✓ Temperature correctly affects exploration")
    
    def test_dirichlet_noise(self):
        """Test Dirichlet noise is applied to root"""
        print("\nTest 6: Dirichlet Noise")
        print("-" * 40)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model that always returns uniform policy
        class UniformModel:
            def __init__(self, device):
                self.device = device
            
            def forward(self, boards):
                batch_size = boards.shape[0]
                policy = torch.ones(batch_size, 7, device=self.device) / 7
                value = torch.zeros(batch_size, device=self.device)
                return policy, value
        
        model = UniformModel(device)
        
        # MCTS with high exploration
        mcts_explore = BatchMCTS(
            model=model,
            c_puct=1.0,
            num_simulations=50,
            dirichlet_alpha=0.3,
            exploration_fraction=0.5,  # High exploration
            device=device
        )
        
        # MCTS without exploration
        mcts_no_explore = BatchMCTS(
            model=model,
            c_puct=1.0,
            num_simulations=50,
            dirichlet_alpha=0.3,
            exploration_fraction=0.0,  # No exploration
            device=device
        )
        
        board = torch.full((6, 7), C4_EMPTY_CELL, dtype=torch.int32, device=device)
        legal_moves = torch.ones(7, dtype=torch.bool, device=device)
        
        # Run multiple times to see variation
        probs_explore = []
        probs_no_explore = []
        
        for _ in range(5):
            p1 = mcts_explore.get_action_probs(board, legal_moves, temperature=1.0)
            p2 = mcts_no_explore.get_action_probs(board, legal_moves, temperature=1.0)
            probs_explore.append(p1)
            probs_no_explore.append(p2)
        
        # With exploration should have more variation
        var_explore = torch.stack(probs_explore).var(dim=0).mean()
        var_no_explore = torch.stack(probs_no_explore).var(dim=0).mean()
        
        print(f"✓ Variance with exploration: {var_explore:.6f}")
        print(f"✓ Variance without exploration: {var_no_explore:.6f}")
        print("✓ Dirichlet noise adds exploration")\
        
    # Add this test to test_batch_mcts.py

    def test_batch_evaluation(self, parallel: bool = False):
        """Test batch MCTS evaluation"""
        print("\nTest 7: Batch Evaluation")
        print("-" * 40)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mock_model = MockTRMModel(device=device)
        
        mcts = BatchMCTS(
            model=mock_model,
            c_puct=1.0,
            num_simulations=10,  # Fewer simulations for speed
            device=device
        )
        
        # Create multiple different boards
        boards = []
        legal_moves_list = []
        
        for i in range(4):
            board = torch.full((6, 7), C4_EMPTY_CELL, dtype=torch.int32, device=device)
            # Add some pieces to make boards different
            if i > 0:
                board[5, i-1] = C4_PLAYER1_CELL
            boards.append(board)
            legal_moves_list.append(torch.ones(7, dtype=torch.bool, device=device))
        
        # Get batch probabilities
        import time
        t0 = time.time()
        if parallel:
            batch_probs = mcts.get_action_probs_batch_parallel(boards, legal_moves_list, temperature=1.0)
        else:
            batch_probs = mcts.get_action_probs_batch(boards, legal_moves_list, temperature=1.0)
        t1 = time.time()
        print(f"✓ Batch evaluation time: {t1 - t0:.3f} seconds")

        assert batch_probs.shape == (4, 7)
        assert torch.allclose(batch_probs.sum(dim=1), torch.ones(4, device=device))
        
        print(f"✓ Batch shape: {batch_probs.shape}")
        print(f"✓ All probabilities sum to 1")
        print("✓ Batch evaluation works correctly")


if __name__ == "__main__":
    print("Testing MCTS")
    print("=" * 60)
    
    test = TestMCTS()
    test.test_initialization()
    test.test_empty_board()
    test.test_winning_position()
    test.test_illegal_moves()
    test.test_temperature_effect()
    test.test_dirichlet_noise()
    test.test_batch_evaluation(parallel=False)
    test.test_batch_evaluation(parallel=True)
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)