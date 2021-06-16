import wandb


def setup_wandb(config):

    args = {}

    # 1. Start a new run
    wandb.init(project='GNN', entity='classy_geometric',config=config)

    # 2. Save model inputs and hyperparameters
    config = wandb.config
    config.learning_rate = 0.01

    # 3. Log gradients and model parameters
    wandb.watch(model)
