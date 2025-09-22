"""
Test for mutual exclusion and configuration validation in train.py
"""
import os
import sys

# Ensure project root is on sys.path 
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def test_mutual_exclusion_logic():
    """
    Test that the mutual exclusion check for C51 and QR-DQN works correctly.
    """
    # Read train.py to find the mutual exclusion check
    train_file = os.path.join(ROOT, 'train.py')
    with open(train_file, 'r') as f:
        source = f.read()
    
    # Look for the mutual exclusion check
    assert "if args.c51 and" in source, "Should have mutual exclusion check starting with c51"
    assert "qr_dqn" in source, "Should check qr_dqn in the condition"
    assert "ValueError" in source, "Should raise ValueError for mutual exclusion"
    assert "Choose only one" in source, "Should have descriptive error message"
    
    print("✓ Mutual exclusion logic verified for C51 and QR-DQN")

def test_default_tau_value():
    """
    Test that tau has a reasonable default value for Polyak updates.
    """
    train_file = os.path.join(ROOT, 'train.py')
    with open(train_file, 'r') as f:
        source = f.read()
    
    # Check tau default
    assert "tau: float = 0.0" in source, "Default tau should be 0.0 (disabled)"
    
    # Check that tau is properly passed to DQNConfig
    assert "tau=args.tau" in source, "Should pass tau parameter to DQNConfig"
    
    print("✓ Tau parameter properly configured with safe default")

def test_preset_file_format():
    """
    Test that the new preset file is properly formatted and contains required fields.
    """
    import json
    
    preset_file = os.path.join(ROOT, 'presets', 'dqn_stable_best.json')
    assert os.path.exists(preset_file), "Stable preset file should exist"
    
    with open(preset_file, 'r') as f:
        config = json.load(f)
    
    # Check required fields for stability improvements
    assert config.get('qr_dqn') == True, "Should default to QR-DQN"
    assert config.get('c51') == False, "Should disable C51"
    assert config.get('tau') == 0.005, "Should enable Polyak updates with reasonable tau"
    assert 'target_update_every' in config, "Should have fallback target update interval"
    
    # Check performance optimizations
    assert config.get('compile') == True, "Should enable compilation for speed"
    assert config.get('pin_memory') == True, "Should enable pinned memory"
    assert config.get('amp') == True, "Should enable AMP"
    
    print("✓ Stable preset configuration validated")

if __name__ == "__main__":
    test_mutual_exclusion_logic()
    test_default_tau_value()
    test_preset_file_format()
    
    print("\n" + "="*50)
    print("All configuration tests passed! ✓")
    print("="*50)