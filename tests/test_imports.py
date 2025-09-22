def test_imports():
    import importlib
    mods = [
        'agents.dqn', 'agents.replay_buffer',
        'env.chet_sim_env', 'env.sim_env',
        'utils.preprocessing', 'train', 'evaluate'
    ]
    for m in mods:
        importlib.import_module(m)
