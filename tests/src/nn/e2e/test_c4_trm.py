#!/usr/bin/env python3
"""
Test Script for TRM Connect Four Implementation
"""

import torch
import sys
from pathlib import Path


def test_environment():
    """Test Connect Four environment"""
    print("Testing Connect Four Environment...")
    
    from connectfour_env import ConnectFourEnv
    
    env = ConnectFourEnv()
    state = env.reset()
    
    print(f"Initial board shape: {state.board.shape}")
    print(f"Legal moves: {state.legal_moves}")
    print(env.render())
    
    # Test moves
    moves = [3, 3, 4, 4, 5, 5, 6]
    for move in moves[:4]:
        state = env.make_move(move)
        print(f"\nAfter move {move}:")
        print(env.render())
        if state.is_terminal:
            print(f"Game over! Winner: {state.winner}")
            break
    
    # Test state conversion
    trm_input = state.to_trm_input()
    print(f"\nTRM input shape: {trm_input.shape}")
    print(f"TRM input values: {trm_input[:10]}...")
    
    print("✅ Environment test passed!\n")


def test_trm_wrapper():
    """Test TRM wrapper"""
    print("Testing TRM Wrapper...")
    
    from trm_c4_wrapper import TRMConnectFourWrapper
    from connectfour_env import ConnectFourEnv
    
    # Create mock TRM
    class MockTRM:
        class Hparams:
            def __init__(self):
                self.hidden_size = 128
                self.seq_len = 42
                self.puzzle_emb_len = 4
        
        def __init__(self):
            self.hparams = self.Hparams()
        
        def initial_carry(self, batch):
            from dataclasses import dataclass
            
            @dataclass
            class InnerCarry:
                z_H: torch.Tensor
                z_L: torch.Tensor
            
            @dataclass
            class Carry:
                inner_carry: InnerCarry
                steps: torch.Tensor
                halted: torch.Tensor
                current_data: dict
            
            batch_size = batch['input'].shape[0]
            return Carry(
                inner_carry=InnerCarry(
                    z_H=torch.randn(batch_size, 46, 128),
                    z_L=torch.randn(batch_size, 46, 128)
                ),
                steps=torch.zeros(batch_size),
                halted=torch.ones(batch_size, dtype=torch.bool),
                current_data=batch
            )
        
        def forward(self, carry, batch):
            outputs = {
                'logits': torch.randn(1, 42, 4),
                'q_halt_logits': torch.randn(1, 1)
            }
            return carry, outputs
    
    mock_trm = MockTRM()
    wrapper = TRMConnectFourWrapper(mock_trm)
    
    # Test policy/value extraction
    env = ConnectFourEnv()
    state = env.reset()
    
    policy, value = wrapper.get_policy_value(
        state.to_trm_input(),
        state.legal_moves
    )
    
    print(f"Policy shape: {policy.shape}")
    print(f"Policy sum: {policy.sum():.3f}")
    print(f"Value: {value:.3f}")
    
    print("✅ TRM Wrapper test passed!\n")


def test_mcts():
    """Test MCTS with TRM"""
    print("Testing MCTS...")
    
    from trm_mcts import TRM_MCTS
    from trm_c4_wrapper import TRMConnectFourWrapper
    from connectfour_env import ConnectFourEnv
    
    # Create mock model
    class MockTRM:
        class Hparams:
            def __init__(self):
                self.hidden_size = 128
                self.seq_len = 42
                self.puzzle_emb_len = 4
        
        def __init__(self):
            self.hparams = self.Hparams()
        
        def initial_carry(self, batch):
            from dataclasses import dataclass
            
            @dataclass
            class InnerCarry:
                z_H: torch.Tensor
            
            @dataclass
            class Carry:
                inner_carry: InnerCarry
                halted: torch.Tensor
            
            return Carry(
                inner_carry=InnerCarry(
                    z_H=torch.randn(1, 46, 128)
                ),
                halted=torch.ones(1, dtype=torch.bool)
            )
        
        def forward(self, carry, batch):
            return carry, {}
    
    mock_trm = MockTRM()
    wrapper = TRMConnectFourWrapper(mock_trm)
    
    mcts = TRM_MCTS(
        wrapper,
        num_simulations=10,
        num_trm_iterations=1
    )
    
    env = ConnectFourEnv()
    state = env.reset()
    
    # Test search
    visits = mcts.search(state)
    print(f"MCTS visits: {visits}")
    
    # Test action selection
    action = mcts.select_action(state)
    print(f"Selected action: {action}")
    
    print("✅ MCTS test passed!\n")


def test_self_play():
    """Test self-play data collection"""
    print("Testing Self-Play...")
    
    from trm_selfplay_trainer import TRMSelfPlayTrainer
    
    # Create minimal mock TRM
    class MockTRM:
        class Hparams:
            def __init__(self):
                self.hidden_size = 64
                self.seq_len = 42
                self.puzzle_emb_len = 2
        
        def __init__(self):
            self.hparams = self.Hparams()
        
        def initial_carry(self, batch):
            from dataclasses import dataclass
            
            @dataclass
            class InnerCarry:
                z_H: torch.Tensor
                z_L: torch.Tensor
            
            @dataclass
            class Carry:
                inner_carry: InnerCarry
                steps: torch.Tensor
                halted: torch.Tensor
                current_data: dict
            
            batch_size = batch['input'].shape[0]
            return Carry(
                inner_carry=InnerCarry(
                    z_H=torch.randn(batch_size, 44, 64),
                    z_L=torch.randn(batch_size, 44, 64)
                ),
                steps=torch.zeros(batch_size),
                halted=torch.ones(batch_size, dtype=torch.bool),
                current_data=batch
            )
        
        def forward(self, carry, batch):
            outputs = {
                'logits': torch.randn(batch['input'].shape[0], 42, 4),
                'q_halt_logits': torch.randn(batch['input'].shape[0], 1)
            }
            carry.halted = torch.ones(batch['input'].shape[0], dtype=torch.bool)
            return carry, outputs
        
        def compute_loss_and_metrics(self, carry, batch):
            return carry, torch.tensor(0.5), {}, True
        
        def parameters(self):
            return []
    
    mock_trm = MockTRM()
    
    config = {
        'buffer_size': 1000,
        'batch_size': 4,
        'mcts_simulations': 5,
        'trm_iterations': 1,
        'device': 'cpu'
    }
    
    trainer = TRMSelfPlayTrainer(mock_trm, config)
    
    # Collect one game
    print("Playing one self-play game...")
    samples = trainer.self_play_game(exploration_temp=1.0)
    print(f"Generated {len(samples)} positions")
    
    if samples:
        print(f"Sample state shape: {samples[0].state.shape}")
        print(f"Sample policy shape: {samples[0].policy_target.shape}")
        print(f"Sample value: {samples[0].value_target}")
    
    print("✅ Self-play test passed!\n")


def main():
    print("="*50)
    print("Testing TRM Connect Four Implementation")
    print("="*50)
    print()
    
    try:
        test_environment()
        test_trm_wrapper()
        test_mcts()
        test_self_play()
        
        print("="*50)
        print("All tests passed! ✅")
        print("="*50)
        print("\nYou can now train with:")
        print("  python train_c4_trm.py --use_mock --iterations 2")
        print("\nOr with your actual TRM:")
        print("  python train_c4_trm.py --iterations 100")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()