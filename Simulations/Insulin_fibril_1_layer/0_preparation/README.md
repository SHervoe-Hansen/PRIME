# 0_preparation

This directory contains the steps required to prepare the insulin fibril system for constant pH molecular dynamics (CpHMD) simulations using Amber.

The preparation consists of:
1. Modifying residue names for titratable residues
2. Reorder fibril chains and identify disulfide bonds
3. Auditing system composition and CpHMD charge baseline
4. Generating topology and coordinate files
5. Creating the CpHMD input file (CPIN)

---

## Step 1 – Modify residue names of titratable residues

Modify the residue names of residues that should be allowed to titrate during the CpHMD simulations.

Note that Amber currently does **not allow titration of N-terminal residues or C-terminal residues**.

Example command:

```bash
perl -pi -e 's/HID/HIP/,s/ASP/AS4/,s/GLU/GL4/,s/CYS/CYX/' insulin_fibril_1L.pdb
```

---

## Step 2 – Reorder fibril chains and identify disulfide bonds

Reorder the insulin chains and identify all disulfide bonds using the provided script:

```bash
python process_insulin_fibril.py 8SBD_1_layer_capped_noH.pdb 8SBD_1_layer_capped_noH_reordered.pdb --disulfides
```

---

## Step 3 – Audit system composition and CpHMD charge baseline

Before feeding the PDB into `tleap`, run the homologous alignment and global tally script to verify the exact number of titratable sites and map out structural asymmetries or truncations within the fibril layers.

```bash
python cphmd_charge_audit.py 8SBD_1_layer_capped_noH_reordered.pdb
```

---

## Step 4 – Generate topology and coordinate files

Prepare the Amber topology (`prmtop` / `parm7`) and input coordinate (`inpcrd`) files using `tleap`:

```bash
tleap -f tleap.in
```

---

## Step 5 – Create the CPIN file

Generate the CpHMD protonation input file using `cpinutil.py`:

```bash
cpinutil.py -resnames cpinutil.py -resnames HIP AS4 GL4 LYS TYR -p Insulin_fibril_1.parm7 -o Insulin_fibril_1.cpin -op Insulin_fibril_1.explicit.parm7
```

This command generates the CPIN file required for constant pH simulations.

