#!/bin/bash
#$ -cwd
#$ -o /u/scratch/z/zhengton/multi_gxe/log/atoms.$JOB_ID.$TASK_ID
#$ -j y
#$ -l h_rt=23:00:00,h_data=16G
#$ -pe shared 8
#$ -N mgxe_atoms
#
# Stage B, one task per trait (array over atom_splits/traits_paper41.txt).
#
#   N=$(grep -cvE '^\s*(#|$)' atom_splits/traits_paper41.txt)
#   qsub -t 1-$N -tc 10 atom_splits/run_atom_pass.sh
#   qsub -t 18 -v SMOKE=1 atom_splits/run_atom_pass.sh      # bmi, 20k rows x 20k SNPs, minutes
#
#   -v TRAITS_FILE=/abs/path     another trait list
#   -v ATOMS=/abs/path.tsv       another atom file (default data/atom_splits/atoms_pool47.G12.tsv)
#   -v PERM=<seed>               V5 null replicate: permute E within atoms
#   -v EXTRA="--no-keep-sketch"  lower memory (recomputes X'Cin in pass 2)
#   -v BACKEND=engine              the C++ passes C++ passes (after build_engine_atoms.sh)
#
# Cost: two passes of ~1.1e15 flops each at G = 12; ~4-6 h on 8 cores (AVX2), less on AVX-512.
# IDEMPOTENT: a task whose tensor exists exits 0.
set -euo pipefail
source /u/local/Modules/default/init/modules.sh
module load anaconda3
conda activate test
module unload python gcc 2>/dev/null || true
ulimit -c 0

ROOT=${MULTI_GXE_ROOT:-$PWD}                       # analysis root: data/, results/
CODE=${MULTI_GXE_CODE:-${ROOT}/atom_splits}        # this directory
SCRATCH=${MULTI_GXE_SCRATCH:-/u/scratch/z/zhengton/multi_gxe}
TRAITS_FILE=${TRAITS_FILE:-${CODE}/traits_paper41.txt}
ATOMS=${ATOMS:-${ROOT}/data/atom_splits/atoms_pool47.G12.tsv}
G=$(awk -F'\t' 'NR>1{if($3>m)m=$3} END{print m+1}' "${ATOMS}")
mapfile -t TRAITS < <(grep -vE '^\s*(#|$)' "${TRAITS_FILE}")
idx=$(( SGE_TASK_ID - 1 ))
(( idx >= 0 && idx < ${#TRAITS[@]} )) || { echo "[ERR] task ${SGE_TASK_ID} out of 1-${#TRAITS[@]}" >&2; exit 1; }
TRAIT="${TRAITS[$idx]}"
export OMP_NUM_THREADS=${NSLOTS:-8} OPENBLAS_NUM_THREADS=${NSLOTS:-8}
mkdir -p "${SCRATCH}/log" "${SCRATCH}/atom_splits/${TRAIT}"

ARGS=(--trait "${TRAIT}" --atoms "${ATOMS}" --out "${SCRATCH}/atom_splits" --threads "${NSLOTS:-8}" --backend "${BACKEND:-numpy}")
if [[ "${BACKEND:-numpy}" == engine ]]; then conda activate engine; fi
TAG=""
if [[ -n "${PERM:-}" ]]; then ARGS+=(--permute-env "${PERM}"); TAG=".perm${PERM}"; fi
if [[ "${SMOKE:-0}" == 1 ]]; then ARGS+=(--max-rows 20000 --max-snps 20000 --jk 10); TAG="${TAG}.smoke20000x20000"; fi
OUT="${SCRATCH}/atom_splits/${TRAIT}/${TRAIT}.atoms.G${G}${TAG}.npz"
if [[ -s "${OUT}" ]]; then echo "[skip] ${OUT} exists"; exit 0; fi
echo "[atoms] ${TRAIT} G=${G} on $(hostname), $(grep -c avx512f /proc/cpuinfo) avx512 cores"
"${CONDA_PREFIX}/bin/python" "${CODE}/run_atom_pass.py" "${ARGS[@]}" ${EXTRA:-}
[[ -s "${OUT}" ]] || { echo "[ERR] no tensor written" >&2; exit 1; }
echo "[atoms] done ${TRAIT}"
