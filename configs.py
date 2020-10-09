import datetime
import argparse as ap
import time
import random
import os
import shutil

thread_num = 8
torch.set_num_threads(thread_num)

if 'JUPYTER' in os.environ:
    from collections import namedtuple
    args = namedtuple(
        'Args', 'mode ckpt tgpu b l seed'
    )(
        mode='train',
        ckpt=None,
        tgpu=False,
        b=False,
        l=False,
        seed=None
    )
else:
    arg_parser = ap.ArgumentParser()
    arg_parser.add_argument('-m', '--mode', default='train')
    arg_parser.add_argument('-c', '--ckpt', default=None)
    arg_parser.add_argument('-tgpu', action='store_true', default=False)
    # arg_parser.add_argument('-d', '--device', type=int, default=0)
    arg_parser.add_argument('-b', action='store_true', default=False)
    arg_parser.add_argument('-l', action='store_true', default=False)
    arg_parser.add_argument('-s', '--seed', type=int, default=None)
    # arg_parser.add_argument('-wd', '--weight_decay', type=float, default=0)

    args = arg_parser.parse_args()

mode = args.mode
training = args.mode == 'train'
validating = args.mode == 'dev'
testing = args.mode == 'test'
debugging = args.mode == 'debug'
testing_gpu = args.tgpu
# ckpt.best is trained with data_part_num = 3
ckpt_id = args.ckpt
# device_id = args.device
loads_best_ckpt = args.b
loads_ckpt = args.l

# training
next_logging_pct = 5.
next_evaluating_pct = 20.

# optimizer
reset_optim = "none"
bert_learning_rate = 1e-5
task_learning_rate = 2e-4
decay_method = "linear"
decay_method_bert = "linear"
max_grad_norm = 1.0
l2_weight_decay = 1e-2
epoch_num = 12

seed = args.seed if args.seed is not None else int(time.time())
# print(f'seed={seed}')
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# training = True
# ckpt_id = None
# loads_best_ckpt = False
# loads_ckpt = False


timestamp = datetime.datetime.now().strftime('%m%d-%H%M%S')


class Dir:
    def __init__(self, name):
        self.name = name

        if not os.path.exists(name):
            os.mkdir(name)

    def __str__(self):
        return self.name


logs_dir = Dir('logs')
ckpts_dir = Dir('ckpts')
data_dir = Dir('data')
configs_pys_dir = Dir('configs_pys')

shutil.copyfile('configs.py', f'{configs_pys_dir}/configs.{timestamp}.py')

# Model hyperparameters.
max_sent_num = 11
feature_size = 20
span_embedding_dim = 2048
max_span_width = 30
ffnn_hidden_size = 1000 
top_span_ratio = .4
max_top_antecedents = 50
coref_depth = 2
dropout_prob = .3
