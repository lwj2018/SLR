from datetime import datetime
class Arguments:
    def __init__(self):
        # Basic setting
        self.dataset = 'phoenix'

        # Path setting
        self.model_path = "./checkpoint"
        self.log_path = "log/transformer_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
        self.eval_log_path = "log/eval_transformer_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
        self.sum_path = "runs/slr_transformer_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
        self.eval_checkpoint = "./checkpoint/slr_transformer_epoch001.pth"

        # Hyperparams
        self.num_classes = 512
        self.epochs = 1000
        self.batch_size = 1
        self.learning_rate = 1e-6
        self.sample_size = 128
        self.clip_length = 2
        self.drop_p = 0.0
        self.smoothing = 0.1
        self.clip_length = 16

        # options
        self.device_list = '1'
