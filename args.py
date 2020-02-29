from datetime import datetime
class Arguments:
    def __init__(self):
        # Path setting
        self.train_frame_root = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train"
        self.train_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/train.corpus.csv"

        self.dev_frame_root = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/dev"
        self.dev_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/dev.corpus.csv"

        self.model_path = "./checkpoint"
        self.log_path = "log/transformer_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
        self.eval_log_path = "log/eval_transformer_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
        self.sum_path = "runs/slr_transformer_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
        self.resume_model = None
        # self.resume_model = "/home/liweijie/projects/SLR/checkpoint/3dres+transformer_phoenix_best.pth.tar"
        self.eval_checkpoint = "./checkpoint/slr_transformer_epoch001.pth"

        # Hyperparams
        self.num_classes = 500
        self.epochs = 100
        self.batch_size = 1
        self.learning_rate = 1e-5
        self.sample_size = 128
        self.clip_length = 16
        self.drop_p = 0.0
        self.smoothing = 0.1

        # options
        self.dataset = 'phoenix'
        self.model_type = '3dres+transformer'
        self.store_name = '_'.join([self.model_type,self.dataset])
        self.device_list = '0,2'
        self.log_interval = 100
