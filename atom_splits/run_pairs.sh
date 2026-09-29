#!/bin/bash
#$ -cwd
#$ -o /u/scratch/z/zhengton/multi_gxe/log/pairs.$JOB_ID.$TASK_ID
#$ -j y
#$ -l h_rt=4:00:00,h_data=4G
#$ -pe shared 4
#$ -N mgxe_pairs
# Stage D, one task per trait, after run_atom_pass.sh.
#   N=$(grep -cvE '^\s*(#|$)' atom_splits/traits_paper41.txt)
#   qsub -t 1-$N atom_splits/run_pairs.sh
set -euo pipefail
source /u/local/Modules/default/init/modules.sh
module load anaconda3
conda activate test
module unload python gcc 2>/dev/null || true
ROOT=${MULTI_GXE_ROOT:-$PWD}                       # analysis root: data/, results/
CODE=${MULTI_GXE_CODE:-${ROOT}/atom_splits}        # this directory
SCRATCH=${MULTI_GXE_SCRATCH:-/u/scratch/z/zhengton/multi_gxe}
TRAITS_FILE=${TRAITS_FILE:-${CODE}/traits_paper41.txt}
mapfile -t TRAITS < <(grep -vE '^\s*(#|$)' "${TRAITS_FILE}")
TRAIT="${TRAITS[$(( SGE_TASK_ID - 1 ))]}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TENSOR=${TENSOR:-${SCRATCH}/atom_splits/${TRAIT}/${TRAIT}.atoms.G12${TAG:-}.npz}
[[ -s "${TENSOR}" ]] || { echo "[ERR] missing ${TENSOR}" >&2; exit 1; }
# IDEMPOTENT: a task whose summary exists exits 0 (delete results/atom_splits/<trait>.* to redo).
if [[ -s "${ROOT}/results/atom_splits${TAG:-}/${TRAIT}.summary.tsv" ]]; then echo "[skip] ${TRAIT} pairs done"; exit 0; fi
"${CONDA_PREFIX}/bin/python" "${CODE}/run_pairs.py" --trait "${TRAIT}" --tensor "${TENSOR}" \
    --jobs "${NSLOTS:-4}" --out "${ROOT}/results/atom_splits${TAG:-}" ${EXTRA:-}
