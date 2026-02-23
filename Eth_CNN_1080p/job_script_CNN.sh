#!/bin/bash
#SBATCH --job-name=Eth-CNN
#SBATCH --output=outputs/test_output_%j.txt
#SBATCH --error=outputs/test_error_%j.txt
#SBATCH --partition=dgx1
#SBATCH --qos=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

echo "=== Fastervit Testing Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURMD_NODENAME"

# Working directory
cd /home/somdyutiai/Krishna_24AI60R38/Eth_CNN_1080p/

# 2. Create output directories to prevent errors
mkdir -p outputs logs saved_models checkpoints

# Activate virtual environment
source /home/somdyutiai/miniconda3/bin/activate
conda activate /raid/somdyutiai/conda_envs/pt__env

# Environment verification
echo "=== Environment Check ==="
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Virtual env: $VIRTUAL_ENV"

# 4. Verify GPU visibility
echo "=== GPU Check ==="
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Dataset verification (quick check)
echo "=== Dataset Verification ==="
ls -lh Data/
echo "Training data exists: $(test -f /raid/somdyutiai/Krishna_24AI60R38/PycharmProjects/HEVC_Intra_Models-ETH-CNN_Pt/Data/1080p_dataset/AI_Train_163200.dat_shuffled && echo 'YES' || echo 'NO')"
echo "Validation data exists: $(test -f /raid/somdyutiai/Krishna_24AI60R38/PycharmProjects/HEVC_Intra_Models-ETH-CNN_Pt/Data/1080p_dataset/AI_Valid_9600.dat_shuffled && echo 'YES' || echo 'NO')"
echo "Test data exists: $(test -f /raid/somdyutiai/Krishna_24AI60R38/PycharmProjects/HEVC_Intra_Models-ETH-CNN_Pt/Data/1080p_dataset/AI_Test_19200.dat_shuffled && echo 'YES' || echo 'NO')"

# Storage monitoring
echo "=== Storage Space Check ==="
df -h /home/somdyutiai/
echo "Available space: $(df -h /home/somdyutiai/ | awk 'NR==2{print $4}')"
echo "Project size: $(du -sh . | cut -f1)"

# 5. Run the Training
echo "=== Starting Training ==="
# Running with python -u (unbuffered) so logs update instantly
python -u ETH_CNN.py > logs/training_log_${SLURM_JOB_ID}.txt 2>&1

echo "=== Job Completed ==="