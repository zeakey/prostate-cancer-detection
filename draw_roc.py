import matplotlib, argparse, os
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os.path as osp
from warnings import warn

parser=argparse.ArgumentParser(description='Detection Project')
parser.add_argument('--evals', nargs='+', required=True)
parser.add_argument('--savefig', type=str, default='/tmp/eval.pdf')
args = parser.parse_args()

colors = np.random.uniform(size=(len(args.evals), 3))
colors[0, :] = 0

args.savefig = osp.realpath(args.savefig)

fig, axes = plt.subplots(1, len(args.evals), figsize=(5*len(args.evals)*1.5, 5))
legends = []

baseline = loadmat(args.evals[0])
base_x = np.squeeze(baseline['b_avg_fp'])
base_y = np.squeeze(baseline['sensitivity'])

for i, ev in enumerate(args.evals):
    legends.append(os.sep.join(ev.split(os.sep)[-3:]))
    if osp.isfile(ev):
        mat = loadmat(ev)
        x = np.squeeze(mat['b_avg_fp'])
        y = np.squeeze(mat['sensitivity'])
        axes[0].plot(x, y, color=colors[i])
    else:
        warn(f"{ev} not found.")
    #
    if (i >= 1):
        axes[i].plot(base_x, base_y, color=colors[0])
        axes[i].plot(x, y, color=colors[i])
    axes[i].set_xlim(0, min(x.max(), 10))
    axes[i].legend([os.sep.join(args.evals[0].split(os.sep)[-3:]), os.sep.join(args.evals[i].split(os.sep)[-3:])])

axes[0].set_xlim(0, min(x.max(), 10))
axes[0].legend(legends)
plt.tight_layout()
print(f"Figure saved to {args.savefig}")
plt.savefig(args.savefig)
