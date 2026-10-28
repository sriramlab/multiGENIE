#!/bin/bash
#$ -cwd
#$ -o /u/scratch/z/zhengton/multi_gxe/log/atoms.$JOB_ID.$TASK_ID
#$ -j y
#$ -l h_rt=23:00:00,h_data=16G
#$ -pe shared 8
#$ -N mgxe_atoms
#
# Stage B, one task per trait (array over scripts/stability_selection/traits_paper41.txt).
#
#   N=$(grep -cvE '^\s*(#|$)' scripts/stability_selection/traits_paper41.txt)
#   qsub -t 1-$N -tc 10 scripts/atom_splits/run_atom_pass.sh
#   qsub -t 18 -v SMOKE=1 scripts/atom_splits/run_atom_pass.sh      # bmi, 20k rows x 20k SNPs, minutes
#
#   -v TRAITS_FILE=/abs/path     another trait list
#   -v ATOMS=/abs/path.tsv       another atom file (default data/atom_splits/atoms_pool47.G12.tsv)
#   -v PERM=<seed>               V5 null replicate: permute E within atoms
#   -v TAG=.seed2,SEED_SHIFT=1   second replicate: output <trait>.atoms.G12.seed2.npz with a shifted probe seed
#                                (pair with ATOMS=.../atoms_pool47.G12.seed2.tsv for a fresh atom partition)
#   -v EXTRA="--no-keep-sketch"  lower memory (recomputes X'Cin in pass 2)
#   -v ENVFILE=/abs/path.env,TAG=.pool84   another environment file (pool87 variant -> L=84 tensor)
#
# Cost: two passes of ~1.1e15 flops each at G = 12; ~4-6 h on 8 cores (AVX2), less on AVX-512.
# IDEMPOTENT: a task whose tensor exists exits 0.
set -euo pipefail
source /u/local/Modules/default/init/modules.sh
module load anaconda3
conda activate test
module unload python gcc 2>/dev/null || true
ulimit -c 0

ROOT=/u/home/z/zhengton/project-sriram/multi_gxe
SCRATCH=/u/scratch/z/zhengton/multi_gxe
TENSOR_DIR=${TENSOR_DIR:-${SCRATCH}/atom_splits}   # -v TENSOR_DIR=... writes the tensor elsewhere (scratch quota)
TRAITS_FILE=${TRAITS_FILE:-${ROOT}/scripts/stability_selection/traits_paper41.txt}
ATOMS=${ATOMS:-${ROOT}/data/atom_splits/atoms_pool47.G12.tsv}
G=$(awk -F'\t' 'NR>1{if($3>m)m=$3} END{print m+1}' "${ATOMS}")
mapfile -t TRAITS < <(grep -vE '^\s*(#|$)' "${TRAITS_FILE}")
idx=$(( SGE_TASK_ID - 1 ))
(( idx >= 0 && idx < ${#TRAITS[@]} )) || { echo "[ERR] task ${SGE_TASK_ID} out of 1-${#TRAITS[@]}" >&2; exit 1; }
TRAIT="${TRAITS[$idx]}"
export OMP_NUM_THREADS=${NSLOTS:-8} OPENBLAS_NUM_THREADS=${NSLOTS:-8}
mkdir -p "${SCRATCH}/log" "${TENSOR_DIR}/${TRAIT}"

ARGS=(--trait "${TRAIT}" --atoms "${ATOMS}" --out "${TENSOR_DIR}" --threads "${NSLOTS:-8}")
TAG="${TAG:-}"
if [[ "${ENVFILE:-}" == MANIFEST ]]; then ENVFILE=${ROOT}/data/pool87/$(awk -F"\t" -v t="${TRAIT}" 'NR>1 && $1==t {print $2}' ${ROOT}/data/pool87/manifest.tsv); fi   # per-trait pool87 variant
if [[ -n "${ENVFILE:-}" ]]; then ARGS+=(--env "${ENVFILE}"); fi   # another environment file (ENV itself is clobbered by the Modules init)
if [[ -n "${TAG}" ]]; then ARGS+=(--tag "${TAG}"); fi
if [[ -n "${SEED_SHIFT:-}" ]]; then ARGS+=(--seed-shift "${SEED_SHIFT}"); fi
if [[ -n "${PERM:-}" ]]; then ARGS+=(--permute-env "${PERM}"); TAG="${TAG}.perm${PERM}"; fi
if [[ "${SMOKE:-0}" == 1 ]]; then ARGS+=(--max-rows 20000 --max-snps 20000 --jk 10); TAG="${TAG}.smoke20000x20000"; fi
OUT="${TENSOR_DIR}/${TRAIT}/${TRAIT}.atoms.G${G}${TAG}.npz"
if [[ -s "${OUT}" ]]; then echo "[skip] ${OUT} exists"; exit 0; fi
echo "[atoms] ${TRAIT} G=${G} on $(hostname), $(grep -c avx512f /proc/cpuinfo) avx512 cores"
"${CONDA_PREFIX}/bin/python" "${ROOT}/scripts/atom_splits/run_atom_pass.py" "${ARGS[@]}" ${EXTRA:-}
[[ -s "${OUT}" ]] || { echo "[ERR] no tensor written" >&2; exit 1; }
echo "[atoms] done ${TRAIT}"
