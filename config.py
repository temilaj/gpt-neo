def get_config():
    return {
        # hyperparameters
        "batch_size": 64, # how many independent sequences will we process in parallel?
        "block_size": 256, # what is the maximum context length for predictions?
        "max_iters": 5000,
        "eval_interval": 500,
        "learning_rate": 3e-4,
        "eval_iters": 100,
        "n_embd": 384,
        "n_head": 6,
        "n_layer": 6,
        "dropout": 0.25,
        # ------------
    }