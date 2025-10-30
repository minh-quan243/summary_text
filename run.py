# scripts/run.py
import argparse
import subprocess
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bart','pegasus','seq2seq'], required=True)
    parser.add_argument('--mode', choices=['train','eval'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=False)
    parser.add_argument('--num_samples', type=int, default=200)
    args = parser.parse_args()

    if args.model == 'bart':
        if args.mode == 'train':
            subprocess.run([sys.executable, 'scripts/train_bart.py', '--config', args.config])
        else:
            cmd = [sys.executable, 'scripts/eval.py', '--config', args.config, '--model', 'bart', '--num_samples', str(args.num_samples)]
            if args.ckpt:
                cmd += ['--ckpt', args.ckpt]
            subprocess.run(cmd)
    elif args.model == 'pegasus':
        if args.mode == 'train':
            subprocess.run([sys.executable, 'scripts/train_pegasus.py', '--config', args.config])
        else:
            cmd = [sys.executable, 'scripts/eval.py', '--config', args.config, '--model', 'pegasus', '--num_samples', str(args.num_samples)]
            if args.ckpt:
                cmd += ['--ckpt', args.ckpt]
            subprocess.run(cmd)
    else:
        if args.mode == 'train':
            subprocess.run([sys.executable, 'scripts/train.py', '--config', args.config])
        else:
            cmd = [sys.executable, 'scripts/eval.py', '--config', args.config, '--model', 'seq2seq', '--num_samples', str(args.num_samples)]
            if args.ckpt:
                cmd += ['--ckpt', args.ckpt]
            subprocess.run(cmd)
