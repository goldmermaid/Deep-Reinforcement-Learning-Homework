import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas


parser = argparse.ArgumentParser()
parser.add_argument('--exps',  nargs='+', type=str)
parser.add_argument('--save', type=str, default=None)
parser.add_argument('--plotrandom', action='store_true', default=False)
args = parser.parse_args()

f, ax = plt.subplots(1, 1)

# if len(args.exps) == 1:
# 	exp = args.exps[0]
# 	log_fname = os.path.join('data', exp, 'log.csv')
#     csv = pandas.read_csv(log_fname)

#     for i in [0,1]:
# 	    color = cm.viridis(i / 2.0)
# 	    ax.plot(csv.index, csv['ReturnAvg'], color=color, label=exp)
# 	    ax.fill_between(csv['Itr'], csv['ReturnAvg'] - csv['ReturnStd'], csv['ReturnAvg'] + csv['ReturnStd'],
# 	                    color=color, alpha=0.2)
# else:
for i, exp in enumerate(args.exps):
    log_fname = os.path.join('data', exp, 'log.csv')
    csv = pandas.read_csv(log_fname)

    color = cm.viridis(i / float(len(args.exps)))
    # ax.plot(csv['Itr'], csv['ReturnAvg'], color=color, label=exp)
    ax.plot(csv.index, csv['ReturnAvg'], color=color, label=exp)
    ax.fill_between(csv.index, csv['ReturnAvg'] - csv['ReturnStd'], csv['ReturnAvg'] + csv['ReturnStd'],
                    color=color, alpha=0.2)
    	

ax.legend()
ax.set_xlabel('Iteration')
ax.set_ylabel('Return')

if args.save:
    os.makedirs('plots', exist_ok=True)
    f.savefig(os.path.join('plots', args.save + '.jpg'))
else:
    plt.show()
