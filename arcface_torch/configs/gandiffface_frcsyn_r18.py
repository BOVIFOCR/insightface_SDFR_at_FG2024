from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r18"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
# config.batch_size = 128
config.batch_size = 64
config.lr = 0.1
config.verbose = 2000
# config.verbose = 10
config.dali = False

# config.rec = "/train_tmp/faces_emore"
# config.rec = '/datasets2/frcsyn_wacv2024/datasets/synthetic/GANDiffFace/images_crops_112x112'          # duo
config.rec = '/nobackup/unico/frcsyn_wacv2024/datasets/synthetic/GANDiffFace/images_crops_112x112'   # diolkos
# config.rec = '/home/bjgbiesseck/datasets/synthetic/GANDiffFace/images_crops_112x112'                 # daugman

# config.num_classes = 85742
config.num_classes = 9305

# config.num_image = 5822653
config.num_image = 501172

config.num_epoch = 20
# config.num_epoch = 30
config.warmup_epoch = 0

# config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
# config.val_targets = ['']
config.val_targets = ['bupt']

# duo
# config.val_dataset_dir = ['/datasets2/frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112']
# config.val_protocol_path = ['/datasets2/frcsyn_wacv2024/comparison_files/comparison_files/sub-tasks_1.1_1.2/bupt_comparison.txt']

# # diolkos
config.val_dataset_dir = ['/nobackup/unico/frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112']
config.val_protocol_path = ['/nobackup/unico/frcsyn_wacv2024/comparison_files/comparison_files/sub-tasks_1.1_1.2/bupt_comparison.txt']

# # daugman
# config.val_dataset_dir = ['/home/bjgbiesseck/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112']
# config.val_protocol_path = ['/home/daugman_groups/unico/frcsyn_wacv2024/comparison_files/comparison_files/sub-tasks_1.1_1.2/bupt_comparison.txt']



# WandB Logger
# config.wandb_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
config.wandb_key = "d7664714c72bd594a957381812da450279f80f66"

config.suffix_run_name = None

# config.using_wandb = False
config.using_wandb = True

# config.wandb_entity = "entity"
config.wandb_entity = "bovifocr"

config.wandb_project = "R18_GANDiffFace_9305classes_whole_dataset"
config.wandb_log_all = True

# config.save_artifacts = False
config.save_artifacts = True

config.wandb_resume = False # resume wandb run: Only if the you wand t resume the last run that it was interrupted

config.notes = ''
