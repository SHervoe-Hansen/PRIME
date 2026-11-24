#!/bin/bash
#
# run_cphremd.sh - prepare the next chunk of an AMBER pH-REMD (constant-pH replica
# exchange) production run. One script does BOTH the initial run and the restarts:
# just call it again after each job finishes.
#
#   run.001  starts from the equilibration restart
#   run.002  starts from run.001's restarts
#   run.003  starts from run.002's restarts   ... and so on until the target.
#
# Model: each chunk is sized to FINISH cleanly within walltime, so you never rely
# on recovering a half-finished job. (~100 ns/run measured ~46 h < 48 h.)
#
# Safety:
#   * Each call creates a NEW run.NNN/ directory and only READS the previous run's
#     restart files, so finished data can never be overwritten.
#   * It refuses to start the next chunk unless the previous one finished cleanly.
#
# pH seeding: pH-REMD makes the pH random-walk across replicas, so each folder is
# one continuous WALKER (its name is just its starting pH), not a fixed pH. Each
# restart resumes every walker at the pH it finished the previous chunk at (read
# from that chunk's cpout), so the chunks stitch into one continuous run with no
# seam. Use --nominal-ph to re-impose the folder pH instead.
#
# Usage:
#   ./run_cphremd.sh              prepare the next chunk (auto-detected); prints qsub
#   ./run_cphremd.sh --redo       re-prepare the highest chunk (overwrites its outputs)
#   ./run_cphremd.sh --nominal-ph seed each walker at its folder pH instead
#
# It does NOT submit anything - it prints the qsub command for you.

set -euo pipefail

#=====================================================================#
#  CONFIGURATION - edit to match your system                          #
#=====================================================================#
BASENAME="insulin"

# Static inputs that must already exist (relative to this directory):
PARM_REL="../0_preparation/${BASENAME}.explicit.parm7"
INIT_CPIN_REL="../0_preparation/${BASENAME}.cpin"
INIT_RST_REL="../3_equilibration/${BASENAME}.equil.rst7"

# Reference structure for -ref (positional restraints / targeted MD).
# This is held FIXED for every run - restraints are always measured against the
# ORIGINAL structure, never the previous chunk's endpoint (which would let the
# structure ratchet away chunk by chunk). Must contain ALL atoms of the solvated
# system, in the same order as the parm7.
# NOTE: -ref only does something if the mdin sets ntr=1 (with restraintmask and
# restraint_wt) or itgtmd; otherwise pmemd just ignores it.
REF_REL="${REF_REL:-${INIT_RST_REL}}"

# mdin template (must contain the solvph=PH placeholder):
TEMPLATE="md_pH_template_NVT.mdin"

# The pH ladder - identical for every run. (Override via env for testing.)
PH_VALUES="${PH_VALUES:-$(seq 1.8 0.2 14)}"

# Chunk size and finish line, in numexchg (exchange attempts).
#   ns per run = numexchg * nstlim * dt / 1000 = numexchg * 300 * 0.002 / 1000
#              = numexchg * 0.0006
#   ~100 ns/run  -> 166667   (measured ~46 h < 48 h walltime)
#    300 ns total -> 500000   -> exactly 3 runs (the last one auto-trims)
NUMEXCHG_PER_RUN="${NUMEXCHG_PER_RUN:-500000}"
TOTAL_NUMEXCHG="${TOTAL_NUMEXCHG:-500000}"

# pH seeding: 1 = seamless (resume each walker at its last pH), 0 = folder pH.
CONTINUE_PH="${CONTINUE_PH:-1}"
CPOUT_PH_LABEL="Solvent pH"          # label preceding the pH value in the cpout

# PBS / environment (baked into the generated submit script):
PBS_QUEUE="regular-g"
PBS_WALLTIME="48:00:00"
PBS_GROUP="hp260036"
CUDA_MODULE="cuda/12.8"
AMBERHOME_DIR="/home/u48000/pmemd24"
#=====================================================================#

# ---- flags ----
REDO=0
for arg in "$@"; do
  case "$arg" in
    --redo)        REDO=1 ;;
    --nominal-ph)  CONTINUE_PH=0 ;;
    --continue-ph) CONTINUE_PH=1 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

BASE_DIR="$(pwd)"

# ---- verify + resolve static inputs ----
for f in "$PARM_REL" "$INIT_CPIN_REL" "$INIT_RST_REL" "$REF_REL" "$TEMPLATE"; do
  [[ -e "$f" ]] || { echo "ERROR: required input not found: $f" >&2; exit 1; }
done
PARM="$(readlink -f "$PARM_REL")"
INIT_CPIN="$(readlink -f "$INIT_CPIN_REL")"
INIT_RST="$(readlink -f "$INIT_RST_REL")"
REF="$(readlink -f "$REF_REL")"

# -ref is inert unless the mdin turns on restraints (ntr>0) or targeted MD.
if ! grep -Eq '(^|[^a-zA-Z_])ntr[[:space:]]*=[[:space:]]*[1-9]|itgtmd[[:space:]]*=[[:space:]]*[1-9]' "$TEMPLATE"; then
  echo ">> NOTE: $TEMPLATE does not set ntr>0 (or itgtmd), so -ref will be supplied but unused."
  echo ">>       To restrain to the reference, add to the template: ntr=1, restraintmask='...', restraint_wt=..."
fi

# nstlim / dt from the template, only for reporting ns.
NSTLIM="$(sed -n 's/.*nstlim[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$TEMPLATE" | head -1)"
DT="$(sed -n 's/.*[^a-zA-Z]dt[[:space:]]*=[[:space:]]*\([0-9.][0-9.]*\).*/\1/p' "$TEMPLATE" | head -1)"
ns_of() { local ex="$1"; [[ -n "$NSTLIM" && -n "$DT" ]] || { echo ""; return; }
  awk -v e="$ex" -v n="$NSTLIM" -v d="$DT" 'BEGIN{printf " (%.1f ns)", e*n*d/1000.0}'; }

num_norm() { awk -v x="$1" 'BEGIN{printf "%g\n", x}'; }
ladder_token_for() { local g lv; g="$(num_norm "$1")"
  for lv in $PH_VALUES; do [[ "$(num_norm "$lv")" == "$g" ]] && { echo "$lv"; return 0; }; done
  return 1; }

# A run is CLEAN iff every replica has non-empty restart files AND a clean-exit
# marker in its mdout.
run_is_clean() {
  local n="$1" rn pH rst cpr mdo; rn="$(printf 'run.%03d' "$n")"
  [[ -d "$rn" ]] || return 1
  for pH in $PH_VALUES; do
    rst="${rn}/pH${pH}/${BASENAME}.pHREMD.rst7"
    cpr="${rn}/pH${pH}/${BASENAME}.pHREMD.cpin"
    mdo="${rn}/pH${pH}/${BASENAME}.pHREMD.mdout"
    [[ -s "$rst" && -s "$cpr" ]] || return 1
    grep -Eq 'Total wall time|Final Performance|wallclock|Master Total wall time' "$mdo" 2>/dev/null || return 1
  done
  return 0
}

read_numexchg() {   # numexchg configured in run N (0 if unknown)
  local n="$1" f; f="$(ls "$(printf 'run.%03d' "$n")"/pH*/*.mdin 2>/dev/null | head -1)"
  [[ -n "$f" ]] || { echo 0; return; }
  sed -n 's/.*numexchg[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$f" | head -1
}

banked_exchanges() {   # sum numexchg over CLEAN runs in [1..$1]
  local upto="$1" sum=0 n
  for (( n=1; n<=upto; n++ )); do run_is_clean "$n" && sum=$(( sum + $(read_numexchg "$n") )); done
  echo "$sum"
}

# ---- find the highest existing run and decide which to prepare ----
shopt -s nullglob; existing=( run.[0-9][0-9][0-9] ); shopt -u nullglob
last_run=0
for d in "${existing[@]}"; do n="$((10#${d#run.}))"; (( n > last_run )) && last_run="$n"; done

if (( REDO == 1 )); then
  (( last_run >= 1 )) || { echo "ERROR: nothing to --redo (no run.NNN found)." >&2; exit 1; }
  next_run="$last_run"; OVERWRITE="-O"
  echo ">> --redo: re-preparing run.$(printf '%03d' "$next_run")"
elif (( last_run == 0 )); then
  next_run=1; OVERWRITE=""
else
  if ! run_is_clean "$last_run"; then
    echo "ERROR: run.$(printf '%03d' "$last_run") has not finished cleanly." >&2
    echo "  (missing restart files, or no clean-exit marker in the mdouts)." >&2
    echo "  If the job is still running, wait. Otherwise resubmit it, or rebuild it with --redo." >&2
    exit 1
  fi
  next_run=$((last_run + 1)); OVERWRITE=""
fi

prev=$((next_run - 1))
prevdir="$(printf 'run.%03d' "$prev")"
RUNDIR="$(printf 'run.%03d' "$next_run")"

if [[ -e "$RUNDIR" && "$REDO" -eq 0 ]]; then
  echo "ERROR: $RUNDIR already exists. Refusing to overwrite (use --redo)." >&2; exit 1
fi

# ---- size this chunk to the remaining exchanges; stop if target met ----
banked_before="$(banked_exchanges "$prev")"
remaining=$(( TOTAL_NUMEXCHG - banked_before ))
if (( remaining <= 0 )); then
  echo ">> Target reached: ${banked_before} / ${TOTAL_NUMEXCHG} exchanges$(ns_of "$banked_before") in completed runs. Nothing to do."
  echo "   (Raise TOTAL_NUMEXCHG to extend, or use --redo to redo the last chunk.)"
  exit 0
fi
CHUNK=$(( NUMEXCHG_PER_RUN < remaining ? NUMEXCHG_PER_RUN : remaining ))
after=$(( banked_before + CHUNK ))
runs_left=$(( (TOTAL_NUMEXCHG - after + NUMEXCHG_PER_RUN - 1) / NUMEXCHG_PER_RUN ))

# ---- assign the pH each walker runs at this chunk ----
# FOLDERS[i] = walker (folder) pH; ASSIGNED[i] = solvph it runs at this chunk.
FOLDERS=(); ASSIGNED=(); SEED_MODE="nominal"
if (( next_run == 1 )); then
  SEED_MODE="initial"
elif (( CONTINUE_PH == 1 )); then
  # resume each walker at the pH it finished the previous (clean) chunk at
  norms=(); fail=""
  for fp in $PH_VALUES; do
    cpo="${prevdir}/pH${fp}/${BASENAME}.pHREMD.cpout"
    raw="$( [[ -s "$cpo" ]] && grep -a "$CPOUT_PH_LABEL" "$cpo" | tail -1 | awk '{print $NF}' )"
    [[ -n "$raw" ]] || { fail="no '${CPOUT_PH_LABEL}' in ${cpo}"; break; }
    tok="$(ladder_token_for "$raw")" || { fail="walker pH${fp} last solvph=${raw} not in ladder"; break; }
    FOLDERS+=("$fp"); ASSIGNED+=("$tok"); norms+=("$(num_norm "$raw")")
  done
  if [[ -z "$fail" ]]; then
    ls_sorted="$(for lv in $PH_VALUES; do num_norm "$lv"; done | sort -n | tr '\n' ' ')"
    en_sorted="$(printf '%s\n' "${norms[@]}" | sort -n | tr '\n' ' ')"
    [[ "$ls_sorted" == "$en_sorted" ]] || fail="recovered pHs are not a permutation of the ladder"
  fi
  if [[ -n "$fail" ]]; then
    echo ">> continue-ph: cannot reconstruct pH assignment ($fail); using nominal folder-pH seeding." >&2
    echo ">> (Set CPOUT_PH_LABEL if your cpout differs, or run with --nominal-ph.)" >&2
    FOLDERS=(); ASSIGNED=()
  else
    SEED_MODE="continue"
  fi
fi
if [[ "$SEED_MODE" != "continue" ]]; then
  FOLDERS=(); ASSIGNED=(); for fp in $PH_VALUES; do FOLDERS+=("$fp"); ASSIGNED+=("$fp"); done
fi

# emit order: sorted by assigned pH (correct for either exchange-neighbour rule)
PLAN="$( i=0; for fp in "${FOLDERS[@]}"; do
  printf '%s|%s|%s\n' "$(num_norm "${ASSIGNED[$i]}")" "$fp" "${ASSIGNED[$i]}"; i=$((i+1))
done | sort -t'|' -k1,1n )"

# ---- report ----
echo ">> Preparing $RUNDIR"
(( next_run == 1 )) && echo ">> Inputs: equilibration restart + initial cpin" \
                    || echo ">> Inputs: restarts from $prevdir"
echo ">> Reference (-ref, same for every run): ${REF_REL}"
case "$SEED_MODE" in
  initial)  echo ">> pH seeding: initial ladder assignment (run 1)." ;;
  continue) echo ">> pH seeding: seamless - each walker resumes at its last pH from $prevdir." ;;
  nominal)  echo ">> pH seeding: nominal - folder pH re-imposed." ;;
esac
echo ">> Progress: ${banked_before} / ${TOTAL_NUMEXCHG}$(ns_of "$banked_before") banked; this run adds ${CHUNK} -> ${after} / ${TOTAL_NUMEXCHG}$(ns_of "$after")."
(( after >= TOTAL_NUMEXCHG )) && echo ">> This is the FINAL run; it lands exactly on the target." \
                             || echo ">> Runs remaining after this one: ${runs_left}."
(( CHUNK < NUMEXCHG_PER_RUN )) && echo ">> (Final chunk trimmed ${NUMEXCHG_PER_RUN} -> ${CHUNK} to hit ${TOTAL_NUMEXCHG} exactly.)"

# ---- write the run ----
mkdir -p "$RUNDIR"
rm -f "${RUNDIR}"/pH*/"${BASENAME}".solvph_*.mdin 2>/dev/null || true   # clear stale mdins on --redo
GROUPFILE="${RUNDIR}/groupfile"; : > "$GROUPFILE"

Nrep=0
while IFS='|' read -r _ fp atok; do
  [[ -n "$fp" ]] || continue
  Nrep=$((Nrep + 1))
  pHdir="${RUNDIR}/pH${fp}"; mkdir -p "$pHdir"     # folder = walker; it never moves

  mdin="${pHdir}/${BASENAME}.solvph_${atok}.mdin"
  sed -E -e "s/(solvph[[:space:]]*=[[:space:]]*)PH/\1${atok}/" \
         -e "s/(numexchg[[:space:]]*=[[:space:]]*)[0-9]+/\1${CHUNK}/" \
         "$TEMPLATE" > "$mdin"

  if (( next_run == 1 )); then
    in_rst="$INIT_RST"; in_cpin="$INIT_CPIN"
  else
    in_rst="${BASE_DIR}/${prevdir}/pH${fp}/${BASENAME}.pHREMD.rst7"
    in_cpin="${BASE_DIR}/${prevdir}/pH${fp}/${BASENAME}.pHREMD.cpin"
  fi
  [[ -s "$in_rst"  ]] || { echo "ERROR: missing input restart for walker pH${fp}: $in_rst" >&2; exit 1; }
  [[ -s "$in_cpin" ]] || { echo "ERROR: missing input cpin for walker pH${fp}: $in_cpin" >&2; exit 1; }

  o_mdin="${BASE_DIR}/${mdin}"
  o_base="${BASE_DIR}/${pHdir}/${BASENAME}.pHREMD"
  printf '# walker pH%s @ solvph %s\n' "$fp" "$atok" >> "$GROUPFILE"
  printf '%s -i %s -p %s -c %s -ref %s -cpin %s -o %s -cpout %s -cprestrt %s -r %s -inf %s -x %s\n' \
    "$OVERWRITE" "$o_mdin" "$PARM" "$in_rst" "$REF" "$in_cpin" \
    "${o_base}.mdout" "${o_base}.cpout" "${o_base}.cpin" "${o_base}.rst7" \
    "${o_base}.mdinfo" "${o_base}.nc" >> "$GROUPFILE"
done <<< "$PLAN"
sed -i 's/^ -i /-i /' "$GROUPFILE"   # cosmetic: strip leading space when no -O

# ---- submit script ----
SUBMIT="${RUNDIR}/submit_CpHREMD.sh"
cat > "$SUBMIT" <<EOF
#!/bin/bash
#PBS -q ${PBS_QUEUE}
#PBS -l select=${Nrep}:mpiprocs=1
#PBS -l walltime=${PBS_WALLTIME}
#PBS -W group_list=${PBS_GROUP}
#PBS -j oe
# Run from inside the run directory so AMBER's per-replica logfile.XXX, the
# remlog, and any stray output land in ${RUNDIR}/ rather than the parent.
# (All paths inside the groupfile are absolute, so cwd does not affect them.)
cd ${BASE_DIR}/${RUNDIR}
export OMP_NUM_THREADS=1
source ~/.bashrc
source ~/.bash_profile
module load ${CUDA_MODULE}
module load cmake
export AMBERHOME=${AMBERHOME_DIR}
source ${AMBERHOME_DIR}/amber.sh
mpiexec -np ${Nrep} pmemd.cuda.MPI -ng ${Nrep} -groupfile groupfile -rem 4 -remlog pHremd.log
EOF
chmod u+x "$SUBMIT"

echo ">> Wrote $GROUPFILE ($Nrep replicas) and $SUBMIT"
echo ">> Submit with:  qsub $SUBMIT"
