import json, glob, itertools, collections, sys
D, BR = 8, 4
def load(f):
    j = json.load(open(f)); t = j['truth']; d = j['drafts']; n = len(d) // (BR * D)
    return t, [[d[(p * BR + b) * D:(p * BR + b + 1) * D] for b in range(BR)] for p in range(n)]
def acc(br, truth, i, shape):
    # shape: lengths per branch (branch 0 = the MTP chain); level-1 tokens differ across branches
    for b, L in enumerate(shape):
        if L == 0 or br[b][0] < 0 or br[b][0] != truth[i + 1]: continue
        a = 1
        while a < L and i + 1 + a < len(truth) and br[b][a] == truth[i + 1 + a]: a += 1
        return a
    return 0
def walk(truth, drafts, shape):
    i, rounds, toks = 0, 0, 0
    while i < len(drafts) - D:
        a = acc(drafts[i], truth, i, shape); rounds += 1; toks += a + 1; i += a + 1
    return toks, rounds
files = sorted(glob.glob('/home/jschmied/kolibri-drafter/treeacc/*.json'))
data = {f.split('/')[-1][:-5]: load(f) for f in files}
def mean_tpr(shape, keys):
    T = R = 0
    for k in keys:
        t, r = walk(*data[k], shape); T += t; R += r
    return T / R
groups = {'all': list(data), 'code': [k for k in data if k.startswith('code')], 'prose+chat': [k for k in data if k[:4] in ('pros', 'chat')], 'tool': [k for k in data if k.startswith('tool')]}
res = {}
for N in range(1, 16):  # draft nodes; window rows = N + 1
    chain = (min(N, D), 0, 0, 0)
    best = max((s for s in itertools.product(range(D + 1), repeat=BR) if sum(s) == N and s[0] >= 1 and all(s[b] <= s[0] for b in range(1, BR))), key=lambda s: mean_tpr(s, groups['all']))
    res[N] = {'chain': chain, 'tree': best, **{g + '_chain': mean_tpr(chain, ks) for g, ks in groups.items()}, **{g + '_tree': mean_tpr(best, ks) for g, ks in groups.items()}}
    r = res[N]
    print('nodes %2d rows %2d  chain tok/round all %.2f code %.2f prose %.2f | best tree %s all %.2f code %.2f prose %.2f' % (N, N + 1, r['all_chain'], r['code_chain'], r['prose+chat_chain'], r['tree'], r['all_tree'], r['code_tree'], r['prose+chat_tree']))
json.dump(res, open('/home/jschmied/kolibri-drafter/treeacc/sim.json', 'w'))
