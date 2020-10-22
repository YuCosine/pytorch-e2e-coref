#!/usr/bin/env python3
import os
import os.path as osp
import sys
import argparse
import itertools
import glog as log
import json
from glob import glob
import tempfile
from collections import OrderedDict
from multiprocessing import Pool, Manager
from functools import partial
import copy
import numpy as np


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='deploy multiple tasks on multiple gpus')
    parser.add_argument('--gpus', nargs='+',
                        help='a list of device ids for available gpus')
    parser.add_argument('--num-task-each-gpu', type=int, default=1,
                        help='number of tasks in each gpu')
    parser.add_argument('--start', type=int, default=0,
                        help='split start')
    parser.add_argument('--end', type=int, default=0,
                        help='split end, 0 for all')
    parser.add_argument('--dry-run', action='store_true',
                        help='do not run actual experiments, just print tasks')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or predict')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    args.gpus = list(map(lambda e: int(e), args.gpus))
    return args


def run_one_exp_with_lock(lock, param):
    global available_gpus
    script = param['script']

    # get available gpu
    lock.acquire()
    assigned_exp_id = copy.deepcopy(exp_id.value)
    exp_id.value += 1
    log.info('Exp {}, current available gpus: {}'.format(assigned_exp_id, available_gpus))
    gpu_id = available_gpus.pop(0)
    log.info('Exp {}, assign gpu {}'.format(assigned_exp_id, gpu_id))
    lock.release()

    shell_cmd = 'GPU={} python3.7 {} '.format(gpu_id, script)
    for key, val in param.items():
        if key in ['script']:
            continue
        elif key in ['model', 'mode']:
            shell_cmd += '{} '.format(val)
        elif isinstance(val, bool):
            if val:
                shell_cmd += '--{} '.format(key)
        elif isinstance(val, str):
            if len(val) > 0:
                shell_cmd += '--{} {} '.format(key, val)
        else:
            shell_cmd += '--{} {} '.format(key, str(val))

    log.info('Exp {}: shell command is {}'.format(assigned_exp_id, shell_cmd))
    if not args.dry_run:
        os.system(shell_cmd)

    # release gpu
    lock.acquire()
    log.info('Exp {} done, release gpu {}'.format(assigned_exp_id, gpu_id))
    available_gpus.insert(0, gpu_id)
    log.info('Exp {}, available gpus after release: {}'.format(assigned_exp_id, available_gpus))
    lock.release()


if __name__ == '__main__':
    args = parse_args()
    available_gpus = args.gpus
    num_task_each_gpu = args.num_task_each_gpu
    params = list()

    # for exp in itertools.product(['ep10', 'ep8', 'ep6'], ['blr5', 'blr2', 'blr1'], ['dp3', 'dp4', 'dp5', 'dp6']):
    #     exp = '_'.join(exp)
    for exp in ['fdp5', 'fdp7', 'sasd2_fdp5', 'sasd2_fdp7', 'cased', 'cased_sasd2', 'cased_fdp5', 'cased_fdp7', 'cased_sasd2_fdp5', 'cased_sasd2_fdp7']:
        param = OrderedDict()
        param['script'] = 'runner.py'
        param['model'] = exp
        param['mode'] = 'train'
        params.append(param)

    if args.mode == 'predict':
        for i in range(len(params)):
            params[i]['output_dir'] = osp.realpath('output')
            # params[i]['split'] = 'val'

    # only run params[start:end] tasks on current machine
    start = args.start
    end = args.end
    if args.end == 0:
        end = len(params)
    assert end > start
    num_task = len(params)
    params = params[start:end]
    log.info('Run {} tasks (id range: [{}, {})) among all {} tasks'.format(len(params), start, end, num_task))

    # make temp directory to maintain consistency for long-time tasks
    with tempfile.TemporaryDirectory() as temp_dirname:
        log.info('Creat temp directory {}'.format(temp_dirname))
        os.system('cp -r *.conf *.py {}'.format(temp_dirname))
        os.system('ln -s {}/data {}/data'.format(osp.realpath(osp.dirname(__file__)), temp_dirname))
        os.system('ln -s {}/logs {}/logs'.format(osp.realpath(osp.dirname(__file__)), temp_dirname))
        log.info('All files copied into {} (data/ soft linked into it)'.format(temp_dirname))
        os.chdir(temp_dirname)
        log.info('Working directory changed into temp directory')

        # assign all gpus
        log.info('Paralleling {} tasks using mp.Manager()'.format(len(params)))
        with Manager() as manager:
            available_gpus = manager.list(available_gpus * num_task_each_gpu)
            exp_id = manager.Value('i', 0)
            l = manager.Lock()
            run_one_exp = partial(run_one_exp_with_lock, l)
            with Pool(processes=len(available_gpus)) as pool:
                pool.map(run_one_exp, params, chunksize=1)
                pool.close()
        log.info('All done')
