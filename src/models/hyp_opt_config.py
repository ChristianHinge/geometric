
sweep_config = {
  "name" : "second-sweep",
  "method" : "bayes",
  "metric": {           # We want to maximize val_acc
      "name": "validation/accuracy",
      "goal": "maximize"
  },
  "early_terminate": {
        "type": "hyperband",
        "min_iter": 5,
        "eta": 2
    },
  "parameters" : {
    "lr" :{
    # log uniform distribution between exp(min) and exp(max)
     "distribution": "log_uniform",
     "min": -9.21,   # exp(-9.21) = 1e-4
     "max": -4.61    # exp(-4.61) = 1e-2
    },
    "epochs" : {
      "value" : 10
    },
    "batch_size" :{
      "values" : [32, 64, 128]
    },
    "layers" :{
      "values" : [[64, 64, 64],[64, 32, 16]]
    },
    "GPU" :{
      "value": False
    },
    "p" :{
      "distribution": "uniform",
      "min": 0,
      "max": 0.5
    }
  }
}
