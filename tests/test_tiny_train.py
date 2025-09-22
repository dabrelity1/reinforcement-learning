import types


def test_tiny_train_chetsim():
    # Run a super-short training loop on CPU to catch basic runtime errors
    import train as tr
    args = tr.Args(
        env='chet-sim', total_steps=300, start_learning=50, buffer_size=1000,
        batch_size=32, eval_every=0, save_every=0, render_every=0,
        num_envs=1, amp=False, prioritized=False, n_step=1,
    )
    # Force deterministic off and async off for simplicity
    args.async_learner = False
    # Run
    tr.main(args)
