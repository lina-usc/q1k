#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --account=def-emayada
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

source /project/def-emayada/q1k/experimental/q1k_env/bin/activate
python -m q1k.pylossless.pipeline "$@"
