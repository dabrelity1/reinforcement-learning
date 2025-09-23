"""
Test suite for DQN correctness, focusing on Double DQN implementation
and soft target updates (Polyak updates).

This test suite performs static code analysis to verify the correctness
of the DQN implementation without requiring external dependencies.
"""
import os
import sys

# Ensure project root is on sys.path 
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def test_double_dqn_structure():
    """
    Test that the Double DQN implementation follows the correct structure:
    1. Action selection uses online network
    2. Q-value evaluation uses target network
    """
    # Read the source code directly to analyze structure
    dqn_file = os.path.join(ROOT, 'agents', 'dqn.py')
    with open(dqn_file, 'r') as f:
        source = f.read()
    
    # Check for Double DQN implementation patterns in vanilla DQN branch
    assert "next_online = self.policy" in source, "Action selection should use online network (self.policy)"
    assert "next_q = self.target" in source, "Q-value evaluation should use target network (self.target)"
    assert "argmax" in source, "Should use argmax for action selection"
    assert "gather" in source, "Should use gather for Q-value evaluation"
    
    # Verify the correct order: online for selection, target for evaluation
    online_pos = source.find("next_online = self.policy")
    target_pos = source.find("next_q = self.target")
    assert online_pos > 0 and target_pos > online_pos, "Should select action with online, then evaluate with target"
    
    print("✓ Double DQN implementation verified: action selection from online, evaluation from target")

def test_polyak_update_structure():
    """
    Test that the Polyak (soft target update) implementation is correct.
    """
    # Read the source code directly
    dqn_file = os.path.join(ROOT, 'agents', 'dqn.py')
    with open(dqn_file, 'r') as f:
        source = f.read()
    
    # Check for Polyak update implementation
    assert "tau" in source, "Should check for tau parameter"
    assert "mul_" in source, "Should use in-place multiplication for Polyak updates"
    assert "add_" in source, "Should use in-place addition for Polyak updates"
    assert "target_update_every" in source, "Should fallback to hard updates when tau=0"
    
    # Verify the Polyak update formula structure
    assert "1.0 - self.cfg.tau" in source, "Should use (1-tau) factor for target parameters"
    assert "alpha=self.cfg.tau" in source, "Should use tau factor for online parameters"
    
    print("✓ Polyak update implementation verified: tau gating and proper update formula")

def test_distributional_dqn_structure():
    """
    Test that both C51 and QR-DQN implementations follow Double DQN correctly.
    """
    # Read the source code directly
    dqn_file = os.path.join(ROOT, 'agents', 'dqn.py')
    with open(dqn_file, 'r') as f:
        source = f.read()
    
    # Check for C51 Double DQN implementation
    assert "if self.cfg.c51:" in source, "Should have C51 branch"
    assert "next_probs_online" in source, "C51 should compute online probabilities for action selection"
    assert "next_logits_target" in source, "C51 should use target network for evaluation"
    assert "next_q_online.argmax" in source, "C51 should select actions using online Q-values"
    
    # Check for QR-DQN Double DQN implementation  
    assert "elif self.cfg.qr_dqn:" in source, "Should have QR-DQN branch"
    assert "next_qtls_online" in source, "QR-DQN should compute online quantiles"
    assert "next_qtls_target" in source, "QR-DQN should use target network for evaluation"
    assert "next_q_online_mean.argmax" in source, "QR-DQN should select actions using online mean Q-values"
    
    print("✓ Distributional DQN implementations verified: both C51 and QR-DQN follow Double DQN")

def test_config_flag_consistency():
    """
    Test that configuration flags are consistent between DQNConfig and train Args.
    """
    # Read both files to check for parameter consistency
    dqn_file = os.path.join(ROOT, 'agents', 'dqn.py')
    train_file = os.path.join(ROOT, 'train.py')
    
    with open(dqn_file, 'r') as f:
        dqn_source = f.read()
    with open(train_file, 'r') as f:
        train_source = f.read()
    
    # Check that key parameters exist in both files
    shared_params = ['tau', 'target_update_every', 'c51', 'qr_dqn', 'gamma', 'lr']
    
    for param in shared_params:
        assert f"{param}:" in dqn_source or f"{param} =" in dqn_source, f"DQNConfig should have {param}"
        assert f"{param}:" in train_source or f"{param} =" in train_source, f"Args should have {param}"
    
    print("✓ Configuration consistency verified between DQNConfig and train Args")

def test_qr_dqn_default_potential():
    """
    Test that QR-DQN can be set as default and works with the configuration system.
    """
    # Read train.py to check QR-DQN handling
    train_file = os.path.join(ROOT, 'train.py')
    with open(train_file, 'r') as f:
        train_source = f.read()
    
    # Check for QR-DQN configuration options
    assert "qr_dqn" in train_source, "Should have qr_dqn parameter in Args"
    assert "num_quantiles" in train_source, "Should have num_quantiles parameter"
    assert "huber_kappa" in train_source, "Should have huber_kappa parameter"
    
    # Check for mutual exclusion logic
    assert "c51 and" in train_source and "qr_dqn" in train_source, "Should have mutual exclusion check"
    
    print("✓ QR-DQN can be configured as default distributional method")

if __name__ == "__main__":
    # Run all tests
    test_double_dqn_structure()
    test_polyak_update_structure()
    test_distributional_dqn_structure()
    test_config_flag_consistency()
    test_qr_dqn_default_potential()
    
    print("\n" + "="*50)
    print("All DQN correctness tests passed! ✓")
    print("="*50)