from datetime import datetime
class Arguments:
    def __init__(self):
        # Basic setting
        self.dataset = 'phoenix'
        self.modal = 'skeleton'        
        # Path setting
        if self.modal=='rgb':
            self.train_frame_root = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train"
            self.train_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/train.corpus.csv"

            self.dev_frame_root = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/dev"
            self.dev_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/dev.corpus.csv"
        elif self.modal=='skeleton':
            self.train_skeleton_root = "/mnt/data/haodong/openpose_output/train"
            self.train_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv"

            self.dev_skeleton_root = "/mnt/data/haodong/openpose_output/dev"
            self.dev_annotation_file = "/mnt/data/public/datasets/phoenix2014-release/phoenix-2014-signerindependent-SI5/annotations/manual/dev.SI5.corpus.csv"

        self.model_path = "./checkpoint"
        self.log_path = "log/transformer_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
        self.eval_log_path = "log/eval_transformer_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
        self.sum_path = "runs/slr_transformer_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
        self.resume_model = None
        # self.resume_model = "/home/liweijie/projects/SLR/checkpoint/202003022212_transformer_skeleton_ckpt.pth.tar"
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
        self.stride = 8

        # options
        self.store_name = '_'.join(['transformer',self.modal,self.dataset])
        self.device_list = '1,3'
        self.log_interval = 100
