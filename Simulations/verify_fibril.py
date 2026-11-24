#!/usr/bin/env python3
"""
Check that a tleap-ready insulin fibril (PDB + _bonds.leap) is fit to start
Amber MD (minimization with the cryo-EM backbone restrained).

  python verify_fibril.py insulin_fibril_3L.pdb [n_layers] [8SBD.pdb]

HARD FAILURES (things minimization / MD cannot repair):
  - D-amino acids, inverted C-beta (allo-Thr / allo-Ile), cis peptide bonds
  - a bond threaded through an aromatic or proline ring (entangled chains)
  - atoms of different chains closer than 1.5 A (overlap)
  - broken geometry: heavy-atom bond > 2.0 A, bond angle < 75 or > 165 deg
  - missing / wrong disulfides or bond commands, wrong sequence or caps,
    numbering not 1..N, TER missing, CRYST1 present, hydrogens present
  - with 8SBD.pdb given: any cryo-EM backbone atom (N, CA, C, O, CB) not at
    its exact file coordinates
  - a proline whose phi / ring is far outside what the ring allows
    (phi outside -120..-30 deg or ring angles outside 85-125 deg)
REPORTED (strain that minimization relaxes): bond angles outside 95-135 deg,
  bond lengths outside 1.18-1.62 A, twisted (but trans) peptides, bent
  carbonyls, mild proline strain, inter-chain contacts 1.5-2.2 A.
Items lying entirely within the cryo-EM model are listed as 'as deposited'.
Exit code 0 = MD-ready.
"""
import os, sys, math
import numpy as np
import gemmi
from scipy.spatial import cKDTree

HUMAN = {'A': "GIVEQCCTSICSLYQLENYCN", 'B': "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"}
ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'CYX': 'C', 'GLN': 'Q',
       'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'HID': 'H', 'HIE': 'H', 'HIP': 'H', 'ILE': 'I',
       'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T',
       'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
RINGS = {'PHE': ('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'), 'TYR': ('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'),
         'HIS': ('CG', 'ND1', 'CD2', 'CE1', 'NE2'), 'HID': ('CG', 'ND1', 'CD2', 'CE1', 'NE2'),
         'HIE': ('CG', 'ND1', 'CD2', 'CE1', 'NE2'), 'HIP': ('CG', 'ND1', 'CD2', 'CE1', 'NE2'),
         'PRO': ('N', 'CA', 'CB', 'CG', 'CD')}
BBX = ('N', 'CA', 'C', 'O', 'CB')


def dih(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - b0.dot(b1) * b1, b2 - b2.dot(b1) * b1
    return math.degrees(math.atan2(np.cross(b1, v).dot(w), v.dot(w)))


def ang(a, b, c):
    u, v = a - b, c - b
    return math.degrees(math.acos(np.clip(u.dot(v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))


def P(r, n):
    a = r.find_atom(n, '*')
    return None if a is None else np.array(a.pos.tolist())


path = sys.argv[1]
n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else None
dep_file = sys.argv[3] if len(sys.argv) > 3 else None
st = gemmi.read_structure(path)
m = st[0]
fails, strain, info = [], [], []
lines = open(path).read().splitlines()

# ---- cryo-EM atoms (by exact coordinate match)
dep_atom = set()
if dep_file:
    d = gemmi.read_structure(dep_file); d.remove_hydrogens()
    out_chains = {ch.name for ch in m}
    dep_keys = {(c.name, r.seqid.num, a.name): np.array(a.pos.tolist())
                for c in d[0] if c.name in out_chains for r in c for a in r if a.name != 'OXT'}
    dtree = cKDTree(np.array([a.pos.tolist() for c in d[0] for r in c for a in r if a.name != 'OXT']))
    for ch in m:
        for r in ch:
            for a in r:
                if dtree.query(np.array(a.pos.tolist()))[0] < 0.005:
                    dep_atom.add((ch.name, r.seqid.num, a.name))
    out_tree = cKDTree(np.array([a.pos.tolist() for ch in m for r in ch for a in r]))
    exp_bb = [k for k in dep_keys if k[2] in BBX]
    bb_missing = sum(1 for k in exp_bb if out_tree.query(dep_keys[k])[0] > 0.005)
    sc_all = [k for k in dep_keys if k[2] not in BBX]
    sc_moved = sum(1 for k in sc_all if out_tree.query(dep_keys[k])[0] > 0.005)
    info.append(f'cryo-EM backbone (N, CA, C, O, CB) exact: {len(exp_bb) - bb_missing}/{len(exp_bb)}; '
                f'cryo-EM side-chain atoms moved: {sc_moved}/{len(sc_all)}')
    if bb_missing:
        fails.append(f'{bb_missing} cryo-EM backbone atoms are not at their file coordinates')
res_dep = {(ch.name, r.seqid.num): bool(dep_atom) and all((ch.name, r.seqid.num, x) in dep_atom
                                                         for x in ('N', 'CA', 'C'))
           for ch in m for r in ch}


def report(keys, msg, hard):
    """keys: residues involved; entirely cryo-EM -> info; else hard failure or strain"""
    if keys and all(res_dep.get(k, False) for k in keys):
        info.append(f'as deposited: {msg}')
    elif hard:
        fails.append(msg)
    else:
        strain.append(msg)


# ---- file structure, sequence, caps
if any(l.startswith('CRYST1') for l in lines):
    fails.append('CRYST1 present')
if sum(l.startswith('TER') for l in lines) != len(m):
    fails.append('TER count != chain count')
nums = [r.seqid.num for ch in m for r in ch]
if nums != list(range(1, len(nums) + 1)):
    fails.append('residue numbering not 1..N')
if n_layers and len(m) != 4 * n_layers:
    fails.append(f'{len(m)} chains, expected {4 * n_layers}')
if any(a.is_hydrogen() for ch in m for r in ch for a in r):
    fails.append('hydrogens present (tleap should build them)')
kind = []
for ch in m:
    body = ''.join(ONE.get(r.name, '?') for r in ch if r.name not in ('ACE', 'NME'))
    k = 'A' if body == HUMAN['A'] else 'B' if body == HUMAN['B'] else None
    if k is None:
        fails.append(f'chain {ch.name}: sequence is not human insulin')
    if ch[0].name != 'ACE' or ch[len(ch) - 1].name != 'NME':
        fails.append(f'chain {ch.name}: ACE/NME caps missing')
    elif {a.name for a in ch[len(ch) - 1]} != {'N', 'CH3'}:
        fails.append(f'chain {ch.name}: NME atom names are not Amber (N, CH3)')
    kind.append(k)
if kind != ['A', 'B'] * (len(m) // 2):
    fails.append('chains not in A, B, A, B, ... order')

# ---- stereochemistry
for ch in m:
    rs = list(ch)
    for i, r in enumerate(rs):
        me = [(ch.name, r.seqid.num)]
        N, CA, C, CB = (P(r, x) for x in ('N', 'CA', 'C', 'CB'))
        if all(x is not None for x in (N, CA, C, CB)) and dih(N, C, CA, CB) < 0:
            report(me, f'D-amino acid {ch.name}{r.seqid.num}{r.name}', True)
        if r.name in ('THR', 'ILE'):
            X_, G_ = P(r, 'OG1' if r.name == 'THR' else 'CG1'), P(r, 'CG2')
            if X_ is not None and G_ is not None and dih(CA, X_, CB, G_) < 0:
                report(me, f'allo-{r.name} {ch.name}{r.seqid.num}', True)
        if r.name == 'PRO' and i > 0:
            phi = dih(P(rs[i - 1], 'C'), N, CA, C)
            ring = [P(r, x) for x in ('N', 'CA', 'CB', 'CG', 'CD')]
            ra = [ang(ring[k - 1], ring[k], ring[(k + 1) % 5]) for k in range(5)]
            if not (-110 <= phi <= -40) or min(ra) < 95 or max(ra) > 118:
                gross = not (-120 <= phi <= -30) or min(ra) < 85 or max(ra) > 125
                report(me, f'proline {ch.name}{r.seqid.num}: phi {phi:.0f}, ring angles '
                       f'{min(ra):.0f}-{max(ra):.0f}' + (' (ring cannot close properly)' if gross else ''),
                       gross)
    for r1, r2 in zip(rs, rs[1:]):
        pair = [(ch.name, r1.seqid.num), (ch.name, r2.seqid.num)]
        a = P(r1, 'CH3') if r1.name == 'ACE' else P(r1, 'CA')
        b = P(r2, 'CH3') if r2.name == 'NME' else P(r2, 'CA')
        c, o, n = P(r1, 'C'), P(r1, 'O'), P(r2, 'N')
        om, ocn, imp = dih(a, c, n, b), ang(o, c, n), abs(dih(a, n, c, o))
        if abs(om) < 90:
            report(pair, f'cis peptide {ch.name}{r1.seqid.num}-{r2.seqid.num} (omega {om:.0f})', True)
        elif abs(om) < 150:
            report(pair, f'twisted peptide {ch.name}{r1.seqid.num}-{r2.seqid.num} (omega {om:.0f})', False)
        if ocn < 110 or imp < 160:
            report(pair, f'bent carbonyl {ch.name}{r1.seqid.num}: O-C-N {ocn:.0f}, '
                   f'{180 - imp:.0f} deg out of plane', False)

# ---- heavy-atom bonds and angles (bonds by distance within a chain)
atoms = [(ch.name, r.seqid.num, r.name, a.name, a.element.name, np.array(a.pos.tolist()))
         for ch in m for r in ch for a in r]
X = np.array([a[5] for a in atoms])
tree = cKDTree(X)
bonds = {i: set() for i in range(len(atoms))}
for i, j in tree.query_pairs(2.3):
    if atoms[i][0] != atoms[j][0]:
        continue
    dd = np.linalg.norm(X[i] - X[j])
    s_ = 'S' in (atoms[i][4], atoms[j][4])
    # a long distance counts as a (stretched) bond only for genuinely bonded backbone pairs
    p_, q_ = (i, j) if atoms[i][1] <= atoms[j][1] else (j, i)
    same = atoms[p_][1] == atoms[q_][1]
    real = ((same and {atoms[p_][3], atoms[q_][3]} in ({'N', 'CA'}, {'CA', 'C'}, {'CA', 'CB'}, {'C', 'O'}))
            or (atoms[q_][1] == atoms[p_][1] + 1 and atoms[p_][3] == 'C' and atoms[q_][3] == 'N'))
    if dd < (2.3 if s_ else 1.75) or (real and dd < 2.3):
        bonds[i].add(j); bonds[j].add(i)
        keys = [atoms[i][:2], atoms[j][:2]]
        if dd > (2.2 if s_ else 2.0):
            report(keys, f'broken bond {atoms[i][0]}{atoms[i][1]}:{atoms[i][3]}-{atoms[j][3]} {dd:.2f} A', True)
        elif not (1.18 <= dd <= (2.10 if s_ else 1.62)):
            report(keys, f'bond {atoms[i][0]}{atoms[i][1]}:{atoms[i][3]}-{atoms[j][3]} {dd:.2f} A', False)
for j, nb in bonds.items():
    nb = sorted(nb)
    for x in range(len(nb)):
        for y in range(x + 1, len(nb)):
            t = ang(X[nb[x]], X[j], X[nb[y]])
            if not (95 <= t <= 135):
                keys = [atoms[q][:2] for q in (nb[x], j, nb[y])]
                report(keys, f'angle {atoms[j][0]}{atoms[j][1]}{atoms[j][2]}:{atoms[nb[x]][3]}-'
                       f'{atoms[j][3]}-{atoms[nb[y]][3]} {t:.0f} deg', not (75 <= t <= 165))

# ---- ring piercing (a bond threaded through a ring)
bond_list = [(i, j) for i, nb in bonds.items() for j in nb if i < j]
BP = np.array([X[i] for i, j in bond_list]); BQ = np.array([X[j] for i, j in bond_list])
res_atoms = {}
for idx, at in enumerate(atoms):
    res_atoms.setdefault(at[:3], {})[at[3]] = idx
for (cname, num, rname), amap in res_atoms.items():
    if rname not in RINGS or not all(x in amap for x in RINGS[rname]):
        continue
    idx = [amap[x] for x in RINGS[rname]]
    R = X[idx]; c0 = R.mean(0)
    nrm = np.linalg.svd(R - c0)[2][2]
    rad = 0.8 * np.linalg.norm(R - c0, axis=1).mean()
    own = set(amap.values())
    near = np.linalg.norm((BP + BQ) / 2 - c0, axis=1) < 3.0
    s1, s2 = (BP - c0) @ nrm, (BQ - c0) @ nrm
    for bi in np.where(near & (s1 * s2 < 0))[0]:
        i, j = bond_list[bi]
        if i in own or j in own:
            continue
        tt = s1[bi] / (s1[bi] - s2[bi])
        if np.linalg.norm(BP[bi] + tt * (BQ[bi] - BP[bi]) - c0) < rad:
            report([(cname, num), atoms[i][:2]], f'ring of {cname}{num}{rname} pierced by a bond of '
                   f'{atoms[i][0]}{atoms[i][1]}{atoms[i][2]}', True)

# ---- inter-chain contacts
close = [(np.linalg.norm(X[i] - X[j]), i, j) for i, j in tree.query_pairs(2.2)
         if atoms[i][0] != atoms[j][0] and not (atoms[i][3] == atoms[j][3] == 'SG')]
for dd, i, j in sorted(close):
    report([atoms[i][:2], atoms[j][:2]], f'contact {atoms[i][0]}{atoms[i][1]}{atoms[i][2]}:{atoms[i][3]} - '
           f'{atoms[j][0]}{atoms[j][1]}{atoms[j][2]}:{atoms[j][3]} {dd:.2f} A', dd < 1.5)
shortest = min((np.linalg.norm(X[i] - X[j]) for i, j in tree.query_pairs(3.0)
                if atoms[i][0] != atoms[j][0] and not (atoms[i][3] == atoms[j][3] == 'SG')), default=9.9)

# ---- disulfides + bond commands
bfile = path[:-4] + '_bonds.leap'
cmds = ([l for l in open(bfile).read().splitlines() if l.startswith('bond')]
        if os.path.exists(bfile) else None)
res_of = {r.seqid.num: (ci, r) for ci, ch in enumerate(m) for r in ch}
cyx = [r for ch in m for r in ch if r.name == 'CYX']
n_ss = sum(1 for a_ in range(len(cyx)) for b_ in range(a_ + 1, len(cyx))
           if cyx[a_].find_atom('SG', '*').pos.dist(cyx[b_].find_atom('SG', '*').pos) < 2.5)
if n_layers and n_ss != 6 * n_layers:
    fails.append(f'{n_ss} disulfides in the file, expected {6 * n_layers}')
if cmds is None:
    info.append('no _bonds.leap file next to the PDB: bond-command checks skipped')
else:
    if len(cmds) != n_ss:
        fails.append(f'{len(cmds)} bond commands but {n_ss} disulfides')
    for l in cmds:
        t = l.split()
        a_, b_ = int(t[1].split('.')[1]), int(t[2].split('.')[1])
        (ca, ra), (cb, rb) = res_of[a_], res_of[b_]
        dd = (ra.find_atom('SG', '*').pos.dist(rb.find_atom('SG', '*').pos)
              if ra.name == rb.name == 'CYX' else 99)
        if not (1.9 < dd < 2.2) or abs(ca - cb) > 1 or (ca != cb and min(ca, cb) % 2 != 0):
            fails.append(f'bad bond command: {l}')

print(f"{path}: {len(m)} chains, {len(nums)} residues, {len(cmds) if cmds is not None else 0} "
      f"bond commands, shortest inter-chain contact {shortest:.2f} A -> "
      f"{'MD-READY' if not fails else 'FAIL'}" + (f" ({len(strain)} strain item(s) for "
                                                   f"minimization to relax)" if strain else ''))
for tag, lst in (('FAIL', fails), ('strain', strain), ('info', info)):
    for f in lst[:10]:
        print(f'    {tag}: {f}')
    if len(lst) > 10:
        print(f'    {tag}: ... {len(lst) - 10} more')
sys.exit(0 if not fails else 1)
