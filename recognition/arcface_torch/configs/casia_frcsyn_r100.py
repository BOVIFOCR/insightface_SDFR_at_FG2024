from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r100"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1
config.verbose = 2000
# config.verbose = 10
config.dali = False

# config.rec = "/train_tmp/faces_emore"
config.rec = '/datasets2/frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/faces_webface_112x112'

# config.num_classes = 85742
config.num_classes = 10572

# config.num_image = 5822653
config.num_image = 490623

# config.num_epoch = 20
config.num_epoch = 30
config.warmup_epoch = 0

# config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
# config.val_targets = ['']
config.val_targets = ['bupt']
config.val_dataset_dir = ['/datasets2/frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112']
config.val_protocol_path = ['/datasets2/frcsyn_wacv2024/comparison_files/comparison_files/sub-tasks_1.1_1.2/bupt_comparison.txt']
