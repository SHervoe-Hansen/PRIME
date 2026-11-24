#!/usr/bin/env python3
"""
Build an N-layer human-insulin fibril from the original 8SBD file (PDB or
mmCIF) by GROWING the unresolved residues one at a time, and write a
tleap-ready PDB plus the tleap disulfide bond commands.

    python build_insulin_fibril.py 8SBD.pdb 3              # -> insulin_fibril_3L.pdb
    python build_insulin_fibril.py 8SBD.pdb 3 my_fibril    # -> my_fibril.pdb

Rules
  - Every cryo-EM backbone atom (N, CA, C, O, CB) and every cysteine is kept
    exactly at its file coordinates. Cryo-EM side chains within 4 A of modeled
    atoms may move (lightly anchored to the file); all moves are reported.
  - Only unresolved residues (and the ACE/NME caps) are built.

Steps
  1. Find insulin molecules (A chain + disulfide-linked B chain), the fibril
     axis, protofilaments and layers from the geometry; take N layers from the
     middle (if N exceeds the file: WARN and extrapolate with the fitted
     helical step; extra layers are model-built, not experimental).
  2. Start from the cryo-EM atoms only. Each growth round adds the next residue
     at every open chain end in every layer at once: backbone from ideal
     geometry (least-clashing phi/psi from the extended and turn regions; the
     modeled parts need not stay in the layer plane; Pro phi -65), side chain from
     the PDBFixer template with a clash-aware rotamer scan (correct L geometry
     by construction), caps last.
  3. After each round the structure is minimized in vacuum (amber14): the new
     residues enter through a soft-core potential ramped to the full force
     field, stereochemistry of modeled residues held by restraints (at junction
     peptides trans is held firmly; the first modeled N may tilt up to 25 deg
     out of the frozen peptide plane and O-C-N is kept >= 110 deg). New residues are audited; failures are re-grown with
     the next candidate (up to 3 attempts).
  4. Write the PDB ordered layer by layer: insulin 1 (A, B), insulin 2 (A, B)
     (each insulin one contiguous molecule), residues 1..N, Amber names (NME
     CH3, CYX, HID/HIE/HIP), heavy atoms only, TER per chain, no CRYST1; write
     the tleap bond commands (unit "mol") to <out>_bonds.leap and print the
     restraint mask for the cryo-EM backbone.
  Runs on the fastest OpenMM platform that works on the machine (CUDA, OpenCL,
  else CPU); force one with e.g.  OPENMM_PLATFORM=CPU python build_insulin_fibril.py ...
  If verify_fibril.py is in the same folder it is run automatically at the end
  (MD-READY / FAIL); it can also be run on its own:
      python verify_fibril.py <out>.pdb <N> 8SBD.pdb

Requires: pip install gemmi pdbfixer scipy   (OpenMM is installed with pdbfixer)
"""
import argparse, math, os, random, string, sys, tempfile, warnings
import numpy as np
import gemmi
from pdbfixer import PDBFixer
from pdbfixer.pdbfixer import Sequence
from openmm import CustomExternalForce, CustomTorsionForce, Context, LocalEnergyMinimizer, VerletIntegrator, unit
from openmm.app import CutoffNonPeriodic, ForceField, PDBxFile
import pdbfixer as _pdbfixer
from openmm import CustomAngleForce, CustomNonbondedForce, Platform
from scipy.spatial import cKDTree


HUMAN = {'A': "GIVEQCCTSICSLYQLENYCN",
         'B': "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"}
THREE = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',
         'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
         'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
         'Y': 'TYR', 'V': 'VAL'}
ONE = {v: k for k, v in THREE.items()}
BACKBONE = {'N', 'CA', 'C', 'O'}
BB_CB = BACKBONE | {'CB'}          # always kept exactly as deposited (and all Cys)
SS_CUT = 2.5          # A, SG-SG distance that counts as a disulfide


# ----------------------------------------------------------------- geometry
def dih(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - b0.dot(b1) * b1, b2 - b2.dot(b1) * b1
    return math.degrees(math.atan2(np.cross(b1, v).dot(w), v.dot(w)))


def angle(a, b, c):
    u, v = a - b, c - b
    return math.degrees(math.acos(u.dot(v) / np.linalg.norm(u) / np.linalg.norm(v)))


def P(res, name):
    a = res.find_atom(name, '*')
    return None if a is None else np.array(a.pos.tolist())


def audit(st):
    """Stereochemistry defects: list of (kind, chain, resnum, resname)."""
    out = []
    for ch in st[0]:
        rs = list(ch)
        for i_, r in enumerate(rs):
            N, CA, C, CB = (P(r, x) for x in ('N', 'CA', 'C', 'CB'))
            if i_ > 0 and N is not None and rs[i_ - 1].find_atom('C', '*') is not None:
                A_ = P(r, 'C') if r.name == 'NME' else CA          # NME methyl is 'C' here
                if A_ is not None:
                    t_ = angle(P(rs[i_ - 1], 'C'), N, A_)
                    if t_ > 135 or t_ < 105:
                        out.append(('bad-ang', ch.name, r.seqid.num, r.name))
            if all(x is not None for x in (N, CA, C, CB)) and dih(N, C, CA, CB) < 0:
                out.append(('D-CA', ch.name, r.seqid.num, r.name))
            if r.name in ('THR', 'ILE') and CA is not None and CB is not None:
                X, G = P(r, 'OG1' if r.name == 'THR' else 'CG1'), P(r, 'CG2')
                if X is not None and G is not None and dih(CA, X, CB, G) < 0:
                    out.append(('inv-CB', ch.name, r.seqid.num, r.name))
            if r.name == 'PRO' and i_ > 0 and rs[i_ - 1].find_atom('C', '*'):
                # the ring ties phi: L-Pro needs phi ~ -65 and ~112 deg at N
                phi = dih(P(rs[i_ - 1], 'C'), N, CA, C)
                CD = P(r, 'CD')
                if not (-110 <= phi <= -40) or (CD is not None and angle(CA, N, CD) < 100):
                    out.append(('bad-Pro', ch.name, r.seqid.num, r.name))
        for r1, r2 in zip(rs, rs[1:]):
            a = P(r1, 'CH3') if r1.name == 'ACE' else P(r1, 'CA')
            b = P(r2, 'C') if r2.name == 'NME' else P(r2, 'CA')
            c, o, n = P(r1, 'C'), P(r1, 'O'), P(r2, 'N')
            if any(x is None for x in (a, b, c, n)):
                continue
            om_ = abs(dih(a, c, n, b))
            if om_ < 150:                             # cis, or strongly twisted
                out.append(('cis' if om_ < 120 else 'twist!' if om_ < 135 else 'twist',
                            ch.name, r1.seqid.num, f'{r1.name}-{r2.name}'))
            if o is not None:
                ocn, imp = angle(o, c, n), abs(dih(a, n, c, o))
                if ocn < 100 or imp < 140:
                    out.append(('bad-CO!', ch.name, r1.seqid.num, r1.name))     # gross
                elif ocn < 110 or imp < 160:
                    out.append(('bad-CO', ch.name, r1.seqid.num, r1.name))
    return out


def disulfides(model):
    """[(chain1, res1, chain2, res2)] for SG-SG pairs closer than SS_CUT."""
    sg = [(ch.name, r) for ch in model for r in ch
          if r.name in ('CYS', 'CYX') and r.find_atom('SG', '*')]
    out = []
    for i in range(len(sg)):
        for j in range(i + 1, len(sg)):
            p = sg[i][1].find_atom('SG', '*').pos
            q = sg[j][1].find_atom('SG', '*').pos
            if p.dist(q) < SS_CUT:
                out.append((sg[i][0], sg[i][1], sg[j][0], sg[j][1]))
    return out


def kabsch(p, q):
    """R, t with R @ p_i + t ~= q_i"""
    pc, qc = p.mean(0), q.mean(0)
    U, S, Vt = np.linalg.svd((p - pc).T @ (q - qc))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def openmm_to_gemmi(topology, positions, path):
    with open(path, 'w') as f:
        PDBxFile.writeFile(topology, positions, f, keepIds=True)
    return gemmi.read_structure(path)


def write_cif(st, path):
    st.setup_entities()
    st.make_mmcif_document().write_file(path)


# ------------------------------------------------- repair helpers
def rotate(points, origin, axis, ang):
    k, c, s = axis, math.cos(ang), math.sin(ang)
    v = points - origin
    return origin + v * c + np.cross(k, v) * s + np.outer(v @ k, k) * (1 - c)


def category(d):
    """'modeled', 'junction' (peptide joining a cryo-EM and a modeled residue)
    or 'deposited' (entirely inside the cryo-EM model)"""
    k, c, num = d[0], d[1], d[2]
    res = [(c, num)] + ([(c, num + 1)] if k in ('cis', 'twist', 'twist!', 'bad-CO', 'bad-CO!') else [])
    dep = [x in deposited_res for x in res]
    return 'deposited' if all(dep) else 'junction' if any(dep) else 'modeled'


def fixable(defects):
    """what the builder repairs: everything in modeled parts, plus true cis bonds
    at junctions (by moving the modeled side). Junction twist / carbonyl bend
    against frozen cryo-EM atoms is reported, not forced."""
    return [d for d in defects if category(d) == 'modeled' or
            (category(d) == 'junction' and d[0] in ('cis', 'twist!', 'bad-CO!', 'D-CA',
                                                     'inv-CB', 'bad-Pro'))]


def summary(defects):
    c = {}
    for d in defects:
        c[d[0]] = c.get(d[0], 0) + 1
    return ', '.join(f'{k} {v}' for k, v in sorted(c.items())) or 'none'


# ------------------------------------------------------------------ inputs
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('structure', help='original 8SBD file (.pdb or .cif)')
ap.add_argument('layers', type=int, help='number of layers to build')
ap.add_argument('out', nargs='?', help='output base name (default insulin_fibril_<N>L)')
args = ap.parse_args()
if args.layers < 1:
    sys.exit('number of layers must be >= 1')
OUT = args.out or f'insulin_fibril_{args.layers}L'
tmp = tempfile.mkdtemp(prefix='fibril_')
random.seed(0); np.random.seed(0)            # stable PDBFixer builds

# ------------------------------------------------ 1) molecules and layers
st = gemmi.read_structure(args.structure)
st.remove_alternative_conformations()
st.remove_hydrogens()
st.remove_ligands_and_waters()
st.remove_empty_chains()
model = st[0]

kind = {}
for ch in model:
    body = {r.seqid.num: ONE[r.name] for r in ch if r.name in ONE}
    if not body:
        continue
    score = {k: sum(1 <= n <= len(ref) and ref[n - 1] == a for n, a in body.items())
             for k, ref in HUMAN.items()}
    k = max(score, key=score.get)
    bad = [f'{n}{a}' for n, a in body.items()
           if not (1 <= n <= len(HUMAN[k])) or HUMAN[k][n - 1] != a]
    if bad:
        sys.exit(f'chain {ch.name} differs from human insulin {k} chain at: {bad}')
    kind[ch.name] = k

partner = {}
for c1, r1, c2, r2 in disulfides(model):
    if c1 != c2 and {kind.get(c1), kind.get(c2)} == {'A', 'B'}:
        a, b = (c1, c2) if kind[c1] == 'A' else (c2, c1)
        if partner.get(a, b) != b:
            sys.exit(f'A chain {a} is disulfide-linked to two B chains')
        partner[a] = b
mols = []                                    # (A chain, B chain, centroid, normal)
for a in sorted(c for c in kind if kind[c] == 'A'):
    if a not in partner:
        warnings.warn(f'A chain {a} has no disulfide-linked B chain; skipped')
        continue
    ca = np.array([r['CA'][0].pos.tolist() for c in (a, partner[a])
                   for r in model[c] if r.find_atom('CA', '*')])
    c0 = ca.mean(0)
    normal = np.linalg.eigh(np.cov((ca - c0).T))[1][:, 0]  # flat sheet -> fibril axis
    mols.append({'A': a, 'B': partner[a], 'c': c0, 'n': normal})
if not mols:
    sys.exit('no insulin molecules (A + disulfide-linked B chain) found')

ref_n = mols[0]['n']
axis = sum(m['n'] * np.sign(m['n'] @ ref_n) for m in mols)
axis /= np.linalg.norm(axis)
for m in mols:
    m['z'] = m['c'] @ axis
    m['lat'] = m['c'] - m['z'] * axis

# two protofilaments = two lateral clusters (2-means)
groups = [list(range(len(mols)))]
if len(mols) > 1:
    L = np.array([m['lat'] for m in mols])
    D = np.linalg.norm(L[:, None] - L[None], axis=2)
    i, j = np.unravel_index(D.argmax(), D.shape)
    cen = [L[i], L[j]]
    for _ in range(20):
        lab = np.array([0 if np.linalg.norm(x - cen[0]) <= np.linalg.norm(x - cen[1])
                        else 1 for x in L])
        cen = [L[lab == g].mean(0) for g in (0, 1)]
    if np.linalg.norm(cen[0] - cen[1]) > 10.0:
        groups = [list(np.where(lab == g)[0]) for g in (0, 1)]
groups = [sorted(g, key=lambda i: mols[i]['z']) for g in groups]
groups.sort(key=lambda g: min(mols[i]['A'] for i in g))   # insulin 1 = pf with first A-chain ID

layers = []
if len(groups) == 1:
    layers = [[i] for i in groups[0]]
else:
    free = list(groups[1])
    for i in groups[0]:
        if free:
            j = min(free, key=lambda j: abs(mols[j]['z'] - mols[i]['z']))
            free.remove(j)
            layers.append([i, j])
        else:
            layers.append([i])
    layers += [[j] for j in free]
layers.sort(key=lambda lay: np.mean([mols[i]['z'] for i in lay]))

z0 = np.mean([mols[i]['z'] for i in layers[0]])
print(f'{args.structure}: {len(mols)} insulin molecules, {len(groups)} protofilament(s), '
      f'{len(layers)} layers')
for k, lay in enumerate(layers, 1):
    z = np.mean([mols[i]['z'] for i in lay])
    print(f'  layer {k:2d}  z = {z - z0:6.1f} A   ' +
          '   '.join(f"insulin {n}: {mols[i]['A']}+{mols[i]['B']}" for n, i in enumerate(lay, 1)))

def layer_points(la, lb):
    """matched deposited backbone (N, CA, C) coordinates of layers la -> lb"""
    p, q = [], []
    for si, di in zip(la, lb):
        for k in ('A', 'B'):
            dres = {r.seqid.num: r for r in model[mols[di][k]]}
            for r in model[mols[si][k]]:
                if r.seqid.num in dres:
                    for an in ('N', 'CA', 'C'):
                        a1, a2 = r.find_atom(an, '*'), dres[r.seqid.num].find_atom(an, '*')
                        if a1 and a2:
                            p.append(a1.pos.tolist()); q.append(a2.pos.tolist())
    return np.array(p), np.array(q)


# helical step (layer k -> k+1), fitted over every consecutive pair in the file
L_file = len(layers)
k_ref = (L_file - 1) // 2
step = np.eye(4)
if L_file > 1:
    pq = [layer_points(layers[k], layers[k + 1]) for k in range(L_file - 1)]
    p, q = np.vstack([x[0] for x in pq]), np.vstack([x[1] for x in pq])
    R1, t1 = kabsch(p, q)
    step[:3, :3], step[:3, 3] = R1, t1
    twist = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R1) - 1) / 2))))
    u = np.array([R1[2, 1] - R1[1, 2], R1[0, 2] - R1[2, 0], R1[1, 0] - R1[0, 1]])
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-9 else axis
    rise = float(t1 @ u)
    if rise < 0:
        rise, twist = -rise, -twist
    worst = 0.0                                # does the step reproduce the file?
    for k in range(L_file):
        M = np.linalg.matrix_power(step if k >= k_ref else np.linalg.inv(step), abs(k - k_ref))
        p, q = layer_points(layers[k_ref], layers[k])
        worst = max(worst, np.sqrt(((p @ M[:3, :3].T + M[:3, 3] - q) ** 2).sum(1).mean()))
    print(f'helical step: rise {rise:.2f} A, twist {twist:.2f} deg per layer; '
          f'reproduces all {L_file} deposited layers to <= {worst:.3f} A RMSD')

n = args.layers
if n * 110 > 9999:
    sys.exit(f'{n} layers = {n * 110} residues; PDB residue numbers stop at 9999 '
             f'(max {9999 // 110} layers)')
extra = max(0, n - L_file)
below, above = extra // 2, extra - extra // 2
start = 0 if extra else (L_file - n) // 2
chosen = layers[start:start + min(n, L_file)]
if extra:
    warnings.warn(f'{n} layers requested but {args.structure} contains only {L_file}: '
                  f'EXTRAPOLATING {extra} layer(s) ({below} below, {above} above) with the '
                  f"fibril's helical symmetry (rise {rise:.2f} A, twist {twist:.2f} deg). "
                  f'The extrapolated layers are model-built, not experimental.')
    taken = {ch.name for ch in model}         # keep 8SBD's style: A chain X, B chain x
    pairs = [(u, u.lower()) for u in string.ascii_uppercase
             if u not in taken and u.lower() not in taken]
    digits = [d for d in string.digits if d not in taken]
    pairs += list(zip(digits[0::2], digits[1::2]))
    vcount = 0
    def new_pair():
        global vcount
        vcount += 1
        return pairs.pop(0) if pairs else (f'V{vcount}A', f'V{vcount}B')
    virtual = []
    for pos_ in list(range(-below, 0)) + list(range(L_file, L_file + above)):
        lay = []
        for i in layers[k_ref]:
            a_, b_ = new_pair()
            kind[a_], kind[b_] = 'A', 'B'
            mols.append({'A': a_, 'B': b_, 'virtual': pos_})
            lay.append(len(mols) - 1)
        virtual.append((pos_, lay))
    chosen = ([lay for p_, lay in virtual if p_ < 0] + chosen +
              [lay for p_, lay in virtual if p_ >= L_file])
is_virtual = lambda lay: 'virtual' in mols[lay[0]]
order = [c for lay in chosen for i in lay for c in (mols[i]['A'], mols[i]['B'])]
print(f'building {len(chosen)} layer(s): file layers {start + 1}-{start + min(n, L_file)}'
      + (f' + {below} extrapolated below and {above} above' if extra else ''))

sel = gemmi.Structure()
sel.add_model(gemmi.Model(1))
for c in [c for c in order if c in kind and model.find_chain(c)]:
    ch = model[c].clone()
    for r in ch:
        for i, a in reversed(list(enumerate(r))):
            if a.name == 'OXT':
                del r[i]
        for a in r:
            a.occ = 1.0
    sel[0].add_chain(ch)
deposited_xyz = {(ch.name, r.seqid.num, a.name): np.array(a.pos.tolist()) / 10.0
                 for ch in model for r in ch for a in r if a.name != 'OXT'}   # nm
deposited = set(deposited_xyz)
ns = gemmi.NeighborSearch(sel[0], sel.cell, 5).populate()
clashes = set()
for ch in sel[0]:
    for r in ch:
        for a in r:
            for mk in ns.find_atoms(a.pos, '\0', radius=2.2):
                c = mk.to_cra(sel[0])
                if c.chain.name != ch.name and not (r.name == c.residue.name == 'CYS'
                                                    and a.name == c.atom.name == 'SG'):
                    clashes.add(tuple(sorted((f'{ch.name}:{r.name}{r.seqid.num}:{a.name}',
                                              f'{c.chain.name}:{c.residue.name}'
                                              f'{c.residue.seqid.num}:{c.atom.name}'))))
if clashes:
    warnings.warn(f'{len(clashes)} inter-chain contacts < 2.2 A between deposited atoms '
                  f'(kept frozen here; they relax in your MD): {sorted(clashes)[:6]}')
deposited_res = {(c, num) for c, num, _ in deposited}


# --------------------------------------------- 2) start: cryo-EM atoms of the chosen layers
def structure_of(chains):
    st_ = gemmi.Structure()
    st_.add_model(gemmi.Model(1))
    for ch in chains:
        st_[0].add_chain(ch)
    return st_



def clean_clone(c):
    ch = model[c].clone()
    for r in ch:
        for i, a in reversed(list(enumerate(r))):
            if a.name == 'OXT':
                del r[i]
        for a in r:
            a.occ = 1.0
    return ch


BOND = {'NCA': 1.458, 'CAC': 1.525, 'CN': 1.329, 'CO': 1.231, 'CACB': 1.530, 'ACE': 1.52}
ANGL = {'NCAC': 111.2, 'CACN': 116.2, 'CNCA': 121.7, 'CACO': 120.8, 'OCN': 123.0, 'CCACB': 110.1}
PHIS, PSIS = (-150, -125, -100, -75, -60), (-50, -30, 100, 125, 150, 175)   # extended + turn


def _place(a, b, c, bond, ang, tor):
    bc = (c - b) / np.linalg.norm(c - b)
    n = np.cross(b - a, bc); n /= np.linalg.norm(n)
    m_ = np.cross(n, bc)
    A_, T_ = math.radians(ang), math.radians(tor)
    return c + bond * (-math.cos(A_) * bc + math.sin(A_) * math.cos(T_) * m_ + math.sin(A_) * math.sin(T_) * n)


_t = _place(np.array([1.0, 0, 0]), np.zeros(3), np.array([0, 1.0, 0]), 1.5, 110, 60)
_SIGN = 1 if abs(dih(np.array([1.0, 0, 0]), np.zeros(3), np.array([0, 1.0, 0]), _t) - 60) < 1 else -1


def place(a, b, c, bond, ang, tor):
    """atom d with |cd| = bond, angle(b,c,d) = ang, dihedral(a,b,c,d) = tor"""
    return _place(a, b, c, bond, ang, _SIGN * tor)



inv_step = np.linalg.inv(step)
ref_lay = layers[k_ref]


def virtual_chains(lay):
    """cryo-EM atoms of the reference layer carried to an extrapolated layer"""
    p_ = mols[lay[0]]['virtual']
    M = np.linalg.matrix_power(step if p_ >= k_ref else inv_step, abs(p_ - k_ref))
    out = []
    for si, di in zip(ref_lay, lay):
        for k in ('A', 'B'):
            ch = clean_clone(mols[si][k])
            ch.name = mols[di][k]
            for r in ch:
                for a in r:
                    xyz = M[:3, :3] @ np.array(a.pos.tolist()) + M[:3, 3]
                    a.pos = gemmi.Position(*xyz)
                    key = (ch.name, r.seqid.num, a.name)
                    deposited_xyz[key] = xyz / 10.0
                    deposited.add(key); deposited_res.add(key[:2])
            out.append(ch)
    return out


chains0 = []
for lay in chosen:
    chains0 += (virtual_chains(lay) if is_virtual(lay) else
                [clean_clone(c) for i in lay for c in (mols[i]['A'], mols[i]['B'])])
order_idx = {c: i for i, c in enumerate(order)}
st = structure_of(sorted(chains0, key=lambda ch: order_idx[ch.name]))

plan = {}                                         # remaining residue numbers per chain end
for ch in st[0]:
    n_seq = len(HUMAN[kind[ch.name]])
    first, last = ch[0].seqid.num, ch[len(ch) - 1].seqid.num
    plan[ch.name] = {'N': list(range(first - 1, 0, -1)) + [0],            # ..., ACE
                     'C': list(range(last + 1, n_seq + 1)) + [n_seq + 1]}  # ..., NME

TPL = os.path.join(os.path.dirname(_pdbfixer.__file__), 'templates')
_tpl = {}


def template(name):
    if name not in _tpl:
        r = gemmi.read_structure(os.path.join(TPL, f'{name}.pdb'))[0][0][0]
        _tpl[name] = {a.name: np.array(a.pos.tolist()) for a in r if not a.is_hydrogen()}
    return _tpl[name]


def side_chain(name, bb):
    """side-chain heavy atoms from the PDBFixer template superposed on N, CA, C, CB"""
    if name in ('GLY', 'ALA', 'ACE', 'NME'):
        return {}
    t = template(name)
    keys = [k for k in ('N', 'CA', 'C', 'CB') if k in bb and k in t]
    R, tr = kabsch(np.array([t[k] for k in keys]), np.array([bb[k] for k in keys]))
    return {k: R @ v + tr for k, v in t.items() if k not in ('N', 'CA', 'C', 'O', 'OXT', 'CB')}


GAMMA = ('CG', 'CG1', 'OG', 'OG1', 'SG')


def best_rotamer(name, bb, sc, tree):
    """chi1/chi2 scan (30 deg) of a new side chain against everything around it"""
    if not sc or name == 'PRO':
        return sc
    names_ = list(sc)
    base = np.array([sc[k] for k in names_])
    CA, CB = bb['CA'], bb['CB']
    piv = next((j for j, k in enumerate(names_) if k in ('CG', 'CG1')
                and any(x[1:2] == 'D' for x in names_)), None)
    mov2 = [j for j, k in enumerate(names_)
            if piv is not None and k not in GAMMA and not (name == 'ILE' and k == 'CG2')]
    ax1 = (CB - CA) / np.linalg.norm(CB - CA)
    best = None
    for c1 in np.radians(np.arange(0, 360, 30)):
        x1 = rotate(base, CB, ax1, c1)
        for c2 in (np.radians(np.arange(0, 360, 30)) if mov2 else [0.0]):
            x = x1.copy()
            if mov2 and c2:
                ax2 = (x1[piv] - CB) / np.linalg.norm(x1[piv] - CB)
                x[mov2] = rotate(x1[mov2], x1[piv], ax2, c2)
            d, _ = tree.query(x, distance_upper_bound=3.2)
            pen = float(np.sum((3.2 - d[np.isfinite(d)]) ** 2))
            if best is None or pen < best[0]:
                best = (pen, x)
    return {k: best[1][j] for j, k in enumerate(names_)}


def heavy_tree(st_, skip):
    return cKDTree(np.array([a.pos.tolist() for c in st_[0] for r in c for a in r
                             if (c.name, r.seqid.num) not in skip]))


def clash(new, tree):
    d, _ = tree.query(np.array(new), distance_upper_bound=3.0)
    return float(np.sum((3.0 - d[np.isfinite(d)]) ** 2))


def candidates_c(ch, name, tree):
    """backbone candidates for the next residue at the C-terminal end"""
    e = ch[len(ch) - 1]
    N0, CA0, C0 = P(e, 'N'), P(e, 'CA'), P(e, 'C')
    firsts = []                                   # (N_new, re-placed O of e, penalty)
    if (ch.name, e.seqid.num) in deposited_res:  # O of e is cryo-EM: N follows it (+ mild tilt)
        O0 = P(e, 'O')
        Nn = place(N0, CA0, C0, BOND['CN'], ANGL['CACN'], dih(N0, CA0, C0, O0) + 180)
        nrm = np.cross(CA0 - C0, O0 - C0); nrm /= np.linalg.norm(nrm)
        tax = np.cross((Nn - C0) / np.linalg.norm(Nn - C0), nrm); tax /= np.linalg.norm(tax)
        for tilt in (0, 10, -10, 20, -20, 30, -30):
            firsts.append((rotate(Nn[None], C0, tax, math.radians(tilt))[0], None, 0.05 * abs(tilt) / 10))
    else:
        for psi0 in PSIS:
            firsts.append((place(N0, CA0, C0, BOND['CN'], ANGL['CACN'], psi0),
                           place(N0, CA0, C0, BOND['CO'], ANGL['CACO'], psi0 + 180), 0.0))
    out = []
    for Nn, Oe, p0 in firsts:
        extra_ = [Oe] if Oe is not None else []
        if name == 'NME':
            CH3 = place(CA0, C0, Nn, BOND['NCA'], ANGL['CNCA'], 180)
            out.append((p0 + clash([Nn, CH3] + extra_, tree), {'N': Nn, 'C': CH3}, Oe))
            continue
        CA = place(CA0, C0, Nn, BOND['NCA'], ANGL['CNCA'], 180)
        for phi in ((-65,) if name == 'PRO' else PHIS):
            C = place(C0, Nn, CA, BOND['CAC'], ANGL['NCAC'], phi)
            CB = None if name == 'GLY' else place(Nn, C, CA, BOND['CACB'], ANGL['CCACB'], 122.7)
            for psi in PSIS:
                O = place(Nn, CA, C, BOND['CO'], ANGL['CACO'], psi + 180)
                at = {'N': Nn, 'CA': CA, 'C': C, 'O': O}
                if CB is not None:
                    at['CB'] = CB
                ring = list(side_chain('PRO', at).values()) if name == 'PRO' else []
                out.append((p0 + clash(list(at.values()) + extra_ + ring, tree), at, Oe))
    return sorted(out, key=lambda x: x[0])


def candidates_n(ch, name, tree):
    """backbone candidates for the next residue at the N-terminal end"""
    s0 = ch[0]
    N0, CA0, C0 = P(s0, 'N'), P(s0, 'CA'), P(s0, 'C')
    out = []
    for phi in ((-65,) if s0.name == 'PRO' else PHIS):       # phi of the current first residue
        C = place(C0, CA0, N0, BOND['CN'], ANGL['CNCA'], phi)
        O = place(CA0, N0, C, BOND['CO'], ANGL['OCN'], 0)
        if name == 'ACE':
            at = {'C': C, 'O': O, 'CH3': place(CA0, N0, C, BOND['ACE'], ANGL['CACN'], 180)}
            out.append((clash(list(at.values()), tree), at, None))
            continue
        CA = place(CA0, N0, C, BOND['CAC'], ANGL['CACN'], 180)
        for psi in PSIS:
            N = place(N0, C, CA, BOND['NCA'], ANGL['NCAC'], psi)
            at = {'N': N, 'CA': CA, 'C': C, 'O': O}
            if name != 'GLY':
                at['CB'] = place(N, C, CA, BOND['CACB'], ANGL['CCACB'], 122.7)
            ring = list(side_chain('PRO', at).values()) if name == 'PRO' else []
            out.append((clash(list(at.values()) + ring, tree), at, None))
    return sorted(out, key=lambda x: x[0])


def new_residue(name, num, atoms):
    r = gemmi.Residue()
    r.name, r.seqid, r.entity_type = name, gemmi.SeqId(num, ' '), gemmi.EntityType.Polymer
    for an, xyz in atoms.items():
        a = gemmi.Atom()
        a.name, a.element, a.pos, a.occ, a.b_iso = an, gemmi.Element(an[0]), gemmi.Position(*xyz), 1.0, 50.0
        r.add_atom(a)
    return r



_platform, _props = None, {}


def context(system):
    """OpenMM Context on the fastest platform that actually works on this machine
    (CUDA, then OpenCL, then CPU). Set OPENMM_PLATFORM=CPU (or CUDA, OpenCL) to force one."""
    global _platform, _props
    if _platform is not None:
        return Context(system, VerletIntegrator(0.001), _platform, _props)
    forced = os.environ.get('OPENMM_PLATFORM')
    names = [forced] if forced else ['CUDA', 'OpenCL', 'CPU', 'Reference']
    skipped = []
    for nm in names:
        try:
            plat = Platform.getPlatformByName(nm)
            props = {'Precision': 'mixed'} if nm in ('CUDA', 'OpenCL') else {}
            ctx = Context(system, VerletIntegrator(0.001), plat, props)   # test it for real
        except Exception as e:
            skipped.append(f'{nm} ({str(e).splitlines()[0]})')
            continue
        _platform, _props = plat, props
        print(f'OpenMM platform: {nm}' + (f'   [not usable here: {"; ".join(skipped)}]' if skipped else ''))
        return ctx
    sys.exit('no usable OpenMM platform: ' + '; '.join(skipped))


def minimize(st_, new_res):
    """add H, freeze the cryo-EM backbone (+CB, all Cys), anchor nearby cryo-EM side
    chains lightly, restrain modeled stereochemistry, let the residues in new_res
    enter through a soft-core potential, then minimize with the full force field"""
    path = os.path.join(tmp, 'grow_in.cif')
    write_cif(st_, path)
    fx = PDBFixer(filename=path)
    fx.topology.setPeriodicBoxVectors(None)
    sgs = [a for a in fx.topology.atoms() if a.name == 'SG' and a.residue.name in ('CYS', 'CYX')]
    Xs = np.array(fx.positions.value_in_unit(unit.angstrom))
    have = {frozenset((a.index, b.index)) for a, b in fx.topology.bonds()}
    for i in range(len(sgs)):                 # disulfides -> CYX (no thiol H added)
        for j in range(i + 1, len(sgs)):
            if (np.linalg.norm(Xs[sgs[i].index] - Xs[sgs[j].index]) < SS_CUT
                    and frozenset((sgs[i].index, sgs[j].index)) not in have):
                fx.topology.addBond(sgs[i], sgs[j])
    fx.addMissingHydrogens(7.0)
    top, pos = fx.topology, fx.positions
    ss_res = {a.residue for b in top.bonds() for a in b
              if b[0].name == 'SG' and b[1].name == 'SG'}
    system = ForceField('amber14-all.xml').createSystem(   # open (still growing) chain ends:
        top, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1.0 * unit.nanometer,
        ignoreExternalBonds=True,             # residues matched on their own atoms
        residueTemplates={r: 'CYX' for r in ss_res})
    nb = [f for f in system.getForces() if f.__class__.__name__ == 'NonbondedForce'][0]
    atoms_ = list(top.atoms())
    for a in atoms_:                          # polar-H guard (output is heavy atoms only)
        q, sig, eps = nb.getParticleParameters(a.index)
        if a.element.symbol == 'H' and eps.value_in_unit(unit.kilojoule_per_mole) == 0:
            nb.setParticleParameters(a.index, q, 0.04 * unit.nanometer, 0.046 * unit.kilocalorie_per_mole)
    akey = lambda a: (a.residue.chain.id, int(a.residue.id), a.name)
    X_ = np.array(pos.value_in_unit(unit.angstrom))
    modeled_i = [a.index for a in atoms_ if a.element.symbol != 'H' and akey(a) not in deposited]
    tm = cKDTree(X_[modeled_i]) if modeled_i else None
    flex = set()
    for a in atoms_:
        k_ = akey(a)
        if (tm is not None and a.element.symbol != 'H' and k_ in deposited and a.name not in BB_CB
                and a.residue.name not in ('CYS', 'CYX') and tm.query_ball_point(X_[a.index], 4.0)):
            flex.add(k_[:2])
    frozen = []
    anchor = CustomExternalForce('kp*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    for n in ('kp', 'x0', 'y0', 'z0'):
        anchor.addPerParticleParameter(n)
    for a in atoms_:
        k_ = akey(a)
        if a.element.symbol != 'H' and k_ in deposited:
            if a.name in BB_CB or a.residue.name in ('CYS', 'CYX') or k_[:2] not in flex:
                frozen.append(a.index)
                system.setParticleMass(a.index, 0.0)
            else:
                anchor.addParticle(a.index, [1.0 * 418.4, *deposited_xyz[k_]])
    anchor.setForceGroup(2)
    system.addForce(anchor)
    K = 200.0 * unit.kilocalorie_per_mole
    rest = CustomTorsionForce('scale*k*(1-cos(theta-theta0))')
    rest.addGlobalParameter('scale', 1.0)
    rest.addPerTorsionParameter('k'); rest.addPerTorsionParameter('theta0')
    flat = CustomTorsionForce('836.8*step(dt-w)*(dt-w)^2; dt=min(d,2*pi-d); '   # kJ/mol/rad^2
                              'd=abs(theta-theta0); w=0.4363; pi=3.141592653589793')
    flat.addPerTorsionParameter('theta0')     # 200 kcal/mol/rad^2 beyond 25 deg
    ocn = CustomAngleForce('836.8*step(t0-theta)*(t0-theta)^2; t0=1.9199')
    for ch in top.chains():
        rs = list(ch.residues())
        ix = [{a.name: a.index for a in r.atoms()} for r in rs]
        for (r1, d1), (r2, d2) in zip(zip(rs, ix), zip(rs[1:], ix[1:])):
            a = d1.get('CH3') if r1.name == 'ACE' else d1.get('CA')
            b = d2.get('C') if r2.name == 'NME' else d2.get('CA')
            if None in (a, b) or 'C' not in d1 or 'N' not in d2:
                continue
            dep1 = (r1.chain.id, int(r1.id)) in deposited_res
            dep2 = (r2.chain.id, int(r2.id)) in deposited_res
            if dep1 and dep2:
                continue
            rest.addTorsion(a, d1['C'], d2['N'], b, [K * (2.0 if (dep1 or dep2) else 1.0), math.pi])   # trans
            if 'O' in d1:
                if dep1 or dep2:              # junction: free tilt up to 25 deg, firm beyond
                    flat.addTorsion(d1['O'], d1['C'], d2['N'], b, [0.0])
                    flat.addTorsion(a, d2['N'], d1['C'], d1['O'], [math.pi])
                    ocn.addAngle(d1['O'], d1['C'], d2['N'], [])   # and O-C-N >= 110 deg
                else:
                    rest.addTorsion(d1['O'], d1['C'], d2['N'], b, [K, 0.0])
                    rest.addTorsion(a, d2['N'], d1['C'], d1['O'], [K, math.pi])
        for j_, (r, d) in enumerate(zip(rs, ix)):
            if r.name == 'PRO' and j_ > 0 and 'C' in ix[j_ - 1]:
                rest.addTorsion(ix[j_ - 1]['C'], d['N'], d['CA'], d['C'], [K, math.radians(-65.0)])
            if all(k in d for k in ('N', 'C', 'CA', 'CB')):
                rest.addTorsion(d['N'], d['C'], d['CA'], d['CB'], [K, math.radians(122.7)])
            if r.name in ('THR', 'ILE') and all(k in d for k in ('CA', 'CB', 'CG2')):
                x = d.get('OG1' if r.name == 'THR' else 'CG1')
                if x is not None:
                    rest.addTorsion(d['CA'], x, d['CB'], d['CG2'], [K, math.radians(122.7)])
    rest.setForceGroup(1)
    system.addForce(rest)
    flat.setForceGroup(1)
    system.addForce(flat)
    ocn.setForceGroup(1)
    system.addForce(ocn)
    new_idx = [a.index for a in atoms_ if (a.residue.chain.id, int(a.residue.id)) in new_res]
    saved = {}
    if new_idx:                               # soft-core entry of the new residues
        sc = CustomNonbondedForce('lam*4*eps*(x^2-x); x=1/(0.5*(1-lam)+(r/sig)^6);'
                                  'sig=0.5*(s1+s2); eps=sqrt(e1*e2)')
        sc.addGlobalParameter('lam', 0.0)
        sc.addPerParticleParameter('s'); sc.addPerParticleParameter('e')
        sc.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
        sc.setCutoffDistance(1.0 * unit.nanometer)
        for i in range(system.getNumParticles()):
            q, sig, eps = nb.getParticleParameters(i)
            sc.addParticle([sig, eps])
        for i in new_idx:
            saved[i] = nb.getParticleParameters(i)
            q, sig, eps = saved[i]
            nb.setParticleParameters(i, 0.0, sig, 0.0)
        for j in range(nb.getNumExceptions()):
            p1, p2 = nb.getExceptionParameters(j)[:2]
            sc.addExclusion(p1, p2)
        newset = set(new_idx)
        sc.addInteractionGroup(new_idx, [i for i in range(system.getNumParticles()) if i not in newset])
        sc.addInteractionGroup(new_idx, new_idx)
        sc.setForceGroup(3)
        system.addForce(sc)
    ctx = context(system)
    ctx.setPositions(pos)
    if saved:
        for lam in (0.05, 0.2, 0.5, 0.8, 1.0):
            ctx.setParameter('lam', lam)
            LocalEnergyMinimizer.minimize(ctx, 10.0, 300)
        for i, p_ in saved.items():           # hand the new atoms to the real force field
            nb.setParticleParameters(i, *p_)
        nb.updateParametersInContext(ctx)
        ctx.setParameter('lam', 0.0)
    LocalEnergyMinimizer.minimize(ctx, 10.0, 1000)
    ctx.setParameter('scale', 0.2)            # let bond angles settle; still no flips
    LocalEnergyMinimizer.minimize(ctx, 10.0, 500)
    return top, ctx.getState(getPositions=True, getForces=True, groups={0}), frozen, system


# --------------------------------------------- 3) grow, one residue per round
print('growing unresolved residues (one per chain end per round):')
rnd = 0
while any(plan[c]['N'] or plan[c]['C'] for c in plan):
    rnd += 1
    ends = [(c, side, plan[c][side][0]) for c in plan for side in ('N', 'C') if plan[c][side]]
    choice = {e_: 0 for e_ in ends}
    cands = {}
    snapshot = st.clone()
    for attempt in range(3):
        st = snapshot.clone()
        new_res = set()
        for (c, side, num) in ends:
            ch = st[0][c]
            n_seq = len(HUMAN[kind[c]])
            name = 'ACE' if num == 0 else 'NME' if num == n_seq + 1 else THREE[HUMAN[kind[c]][num - 1]]
            end_r = ch[0] if side == 'N' else ch[len(ch) - 1]
            tree = heavy_tree(st, {(c, end_r.seqid.num)})
            if (c, side, num) not in cands:
                cands[(c, side, num)] = (candidates_n if side == 'N' else candidates_c)(ch, name, tree)
            lst = cands[(c, side, num)]
            _, bb, Oe = lst[min(choice[(c, side, num)], len(lst) - 1)]
            atoms_new = dict(bb)
            if name not in ('ACE', 'NME'):
                atoms_new.update(best_rotamer(name, bb, side_chain(name, bb), tree))
            r_new = new_residue(name, num, atoms_new)
            if side == 'C':
                if Oe is not None:
                    end_r.find_atom('O', '*').pos = gemmi.Position(*Oe)
                ch.add_residue(r_new)
            else:
                ch.add_residue(r_new, 0)
            new_res.add((c, num))
        top, state, frozen, system = minimize(st, new_res)
        cur = openmm_to_gemmi(top, state.getPositions(), os.path.join(tmp, 'grow_out.cif'))
        bad = [d for d in fixable(audit(cur)) if (d[1], d[2]) in new_res or (d[1], d[2] + 1) in new_res]
        if not bad:
            break
        for e_ in ends:                        # re-grow failing ends with the next candidate
            if any(d[1] == e_[0] and (d[2] == e_[2] or d[2] + 1 == e_[2]) for d in bad):
                choice[e_] += 1
    print(f'  round {rnd}: added {len(ends)} residue(s)' +
          (f'; still flagged after 3 attempts: {summary(bad)}' if bad else ''))
    st = cur.clone()
    st.remove_hydrogens()
    for (c, side, num) in ends:
        plan[c][side].pop(0)

print('final minimization of the complete stack ...')
top, state, frozen, system = minimize(st, set())
cur = openmm_to_gemmi(top, state.getPositions(), os.path.join(tmp, 'final.cif'))

# --------------------------------------------- 5) final checks
X = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
F = np.linalg.norm(state.getForces(asNumpy=True)
                   .value_in_unit(unit.kilocalorie_per_mole / unit.angstrom), axis=1)
mobile = np.ones(len(F), bool); mobile[frozen] = False
bf = [f for f in system.getForces() if f.__class__.__name__ == 'HarmonicBondForce'][0]
worst_bond = max(abs(np.linalg.norm(X[p] - X[q]) - r0.value_in_unit(unit.angstrom))
                 for p, q, r0, k in (bf.getBondParameters(i) for i in range(bf.getNumBonds())))
atoms = list(top.atoms())
bonds = np.array([(b[0].index, b[1].index) for b in top.bonds()])
pierced = []                                  # a bond threaded through a ring
RINGS = {'PHE': ('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'),
         'TYR': ('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'),
         'HIS': ('CG', 'ND1', 'CD2', 'CE1', 'NE2'),
         'PRO': ('N', 'CA', 'CB', 'CG', 'CD')}
for res in top.residues():
    if res.name not in RINGS:
        continue
    idx = [a.index for a in res.atoms() if a.name in RINGS[res.name]]
    R = X[idx]; c = R.mean(0)
    nrm = np.linalg.svd(R - c)[2][2]
    rad = 0.8 * np.linalg.norm(R - c, axis=1).mean()
    own = {a.index for a in res.atoms()}
    p, q = X[bonds[:, 0]], X[bonds[:, 1]]
    near = np.linalg.norm((p + q) / 2 - c, axis=1) < 3.0
    s1, s2 = (p - c) @ nrm, (q - c) @ nrm
    for bi in np.where(near & (s1 * s2 < 0))[0]:
        if bonds[bi, 0] in own or bonds[bi, 1] in own:
            continue
        t = s1[bi] / (s1[bi] - s2[bi])
        if np.linalg.norm(p[bi] + t * (q[bi] - p[bi]) - c) < rad:
            a = atoms[bonds[bi, 0]].residue
            pierced.append(f'{res.chain.id}:{res.name}{res.id} ring pierced by '
                           f'{a.chain.id}:{a.name}{a.id}')

final = cur
all_left = audit(final)
left = fixable(all_left)
junction = [d for d in all_left if category(d) == 'junction' and d not in left]
as_deposited = [d for d in all_left if category(d) == 'deposited']
def _shift(c, r, a):
    return float(np.linalg.norm(np.array(a.pos.tolist()) - deposited_xyz[(c.name, r.seqid.num, a.name)] * 10))
shift = [_shift(c, r, a) for c in final[0] for r in c for a in r
         if (c.name, r.seqid.num, a.name) in deposited_xyz]
bb = [_shift(c, r, a) ** 2 for c in final[0] for r in c for a in r
      if a.name in BACKBONE and (c.name, r.seqid.num, a.name) in deposited_xyz]
bb_shift = [_shift(c, r, a) for c in final[0] for r in c for a in r
            if a.name in BB_CB and (c.name, r.seqid.num, a.name) in deposited_xyz]
moved = {}
for c in final[0]:
    for r in c:
        for a in r:
            k_ = (c.name, r.seqid.num, a.name)
            if k_ in deposited_xyz and a.name not in BB_CB and _shift(c, r, a) > 0.1:
                moved[(c.name, r.name, r.seqid.num)] = max(moved.get((c.name, r.name, r.seqid.num), 0), _shift(c, r, a))
print(f'cryo-EM backbone (N, CA, C, O, CB): largest shift {max(bb_shift):.4f} A (kept as deposited)')
print(f'cryo-EM side chains moved to make room for modeled residues: {len(moved)}' +
      (f' (largest {max(moved.values()):.2f} A; ' +
       ', '.join(f'{c}:{n}{i} {d:.1f}' for (c, n, i), d in sorted(moved.items())[:8]) +
       (' ...' if len(moved) > 8 else '') + ')' if moved else ''))
if junction:
    print(f'junction strain (modeled residue meeting frozen cryo-EM atoms; relaxes in your '
          f'restrained MD): {summary(junction)}')
if as_deposited:
    print(f'note: {len(as_deposited)} item(s) lie entirely in the cryo-EM model and were left as '
          f'deposited: {as_deposited[:4]}')
print(f'final: max |F| mobile {F[mobile].max():.1f}, all atoms {F.max():.1f} kcal/mol/A; '
      f'worst bond deviation {worst_bond:.2f} A; stereo defects {len(left)}; '
      f'pierced rings {len(pierced)}')
for msg, bad in (('stereochemistry defects remain', left),
                 ('ring piercing (entangled modeled chains)', pierced)):
    if bad:
        warnings.warn(f'{msg}: {bad} - inspect before use')
af = [f for f in system.getForces() if f.__class__.__name__ == 'HarmonicAngleForce'][0]
bad_angles = set()
worst_angle = 0.0
for i in range(af.getNumAngles()):
    p1, p2, p3, t0, k_ = af.getAngleParameters(i)
    if any(atoms[j].element.symbol == 'H' for j in (p1, p2, p3)):
        continue                              # output is heavy atoms; tleap rebuilds H
    dev = abs(angle(X[p1], X[p2], X[p3]) - math.degrees(t0.value_in_unit(unit.radian)))
    worst_angle = max(worst_angle, dev)
    if dev > 25:
        r_ = atoms[p2].residue
        bad_angles.add(f'{r_.chain.id}:{r_.name}{r_.id}:{atoms[p2].name}')
print(f'       worst heavy-atom bond-angle deviation {worst_angle:.1f} deg')
if bad_angles:
    warnings.warn(f'bond angles > 25 deg from ideal at {sorted(bad_angles)} - inspect before use')
if F.max() > 1000:
    warnings.warn(f'force-field force up to {F.max():.0f} kcal/mol/A - inspect before use')
if worst_bond > 0.2:
    warnings.warn(f'a bond is stretched by {worst_bond:.2f} A - inspect before use')

# --------------------------------------------- 6) tleap-ready PDB + bonds
m = final[0]
for ch in m:
    for r in ch:
        if r.name == 'HIS':
            d, e = r.find_atom('HD1', '*'), r.find_atom('HE2', '*')
            r.name = 'HIP' if (d and e) else 'HID' if d else 'HIE'
for c1, r1, c2, r2 in disulfides(m):
    r1.name = r2.name = 'CYX'
final.remove_hydrogens()
old_num, seq = {}, 0
for ch in m:
    for r in ch:
        seq += 1
        old_num[(ch.name, seq)] = r.seqid.num
        if r.name == 'NME':
            for a in r:
                if a.name == 'C':
                    a.name = 'CH3'
        r.het_flag = 'A'
        r.seqid = gemmi.SeqId(seq, ' ')
final.connections.clear()
ss = disulfides(m)                           # re-detect with the new numbering

# sanity (internal chain names): sequence, caps, contiguous molecules
names = [ch.name for ch in m]
for ch in m:
    body = ''.join(ONE.get({'CYX': 'CYS', 'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS'}
                           .get(r.name, r.name), '') for r in ch)
    assert ch[0].name == 'ACE' and ch[len(ch) - 1].name == 'NME', ch.name
    assert body == HUMAN[kind[ch.name]], f'chain {ch.name} sequence mismatch'
for c1, _, c2, _ in ss:
    if c1 != c2:
        assert abs(names.index(c1) - names.index(c2)) == 1, f'{c1}/{c2} not adjacent'

# one-character output chain IDs (file IDs kept; consecutive chains always differ)
ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits
out_id, used, prev, k = {}, set(), None, 0
for nm in names:
    if len(nm) == 1 and nm not in used:
        oid = nm
    else:
        while True:
            oid = ALPHABET[k % len(ALPHABET)]; k += 1
            if oid != prev and (oid not in used or len(used) >= len(ALPHABET)):
                break
    out_id[nm], prev = oid, oid
    used.add(oid)
reused = len(set(out_id.values())) < len(names)

where = {}                                   # chain -> layer label, insulin
for L_i, lay in enumerate(chosen, 1):
    tag = f'layer {L_i}' + (' (extrapolated)' if is_virtual(lay) else '')
    for I_i, i in enumerate(lay, 1):
        where[mols[i]['A']] = where[mols[i]['B']] = (tag, I_i)
cmds = []
for c1, r1, c2, r2 in sorted(ss, key=lambda s_: (s_[1].seqid.num, s_[3].seqid.num)):
    a, b = r1.seqid.num, r2.seqid.num
    tag, I_i = where[c1]
    cmds.append(f'bond mol.{a}.SG mol.{b}.SG    # {tag} insulin {I_i}: '
                f'{out_id[c1]} Cys{old_num[(c1, a)]} - {out_id[c2]} Cys{old_num[(c2, b)]}')

spans, cur_ = [], None                       # restraint mask: deposited backbone
for ch in m:
    for r in ch:
        on = (ch.name, old_num[(ch.name, r.seqid.num)]) in deposited_res
        if on and cur_ and cur_[1] == r.seqid.num - 1:
            cur_[1] = r.seqid.num
        elif on:
            cur_ = [r.seqid.num, r.seqid.num]; spans.append(cur_)
        else:
            cur_ = None
mask = ':' + ','.join(f'{a}-{b}' if a != b else f'{a}' for a, b in spans) + '@N,CA,C,O'

for ch in m:
    ch.name = out_id[ch.name]
lines, prev = [], None
for line in final.make_pdb_string().splitlines(True):
    if line.startswith(('ATOM', 'HETATM')):
        if prev is not None and line[21] != prev:
            lines.append('TER\n')
        prev = line[21]
        lines.append(line)
open(OUT + '.pdb', 'w').writelines(lines + ['TER\n', 'END\n'])
open(OUT + '_bonds.leap', 'w').write('\n'.join(cmds) + '\n')

chk = gemmi.read_structure(OUT + '.pdb')    # file-level sanity
assert [r.seqid.num for ch in chk[0] for r in ch] == list(range(1, seq + 1))
assert sum(l.startswith('TER') for l in open(OUT + '.pdb')) == len(names)

print(f'\nwrote {OUT}.pdb ({len(names)} chains, {seq} residues, '
      f'{chk[0].count_atom_sites()} heavy atoms) and {OUT}_bonds.leap')
if reused:
    print('note: more chains than one-character IDs, so some chain IDs repeat; '
          'tleap ignores chain IDs (chains are separated by TER records)')
print('\n'.join(cmds))
print(f"\nrestraint mask (deposited backbone{', incl. extrapolated cores' if extra else ''}): "
      f"restraintmask='{mask}'")

# --------------------------------------------- 7) independent check
vpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verify_fibril.py')
if os.path.exists(vpath):
    import subprocess
    print('\nindependent check (verify_fibril.py):')
    res = subprocess.run([sys.executable, vpath, OUT + '.pdb', str(len(chosen)), args.structure],
                         capture_output=True, text=True)
    print(res.stdout.rstrip() or res.stderr.rstrip())
else:
    print(f'\nverify_fibril.py not found next to this script; check the result with\n'
          f'  python verify_fibril.py {OUT}.pdb {len(chosen)} {args.structure}')
