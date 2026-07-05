# Medical Imaging: Chest X-Ray Pneumonia Classification Pipeline

This repository contains a convolutional neural network (CNN) pipeline with a two-stage fine-tuning workflow for the binary classification of pneumonia from chest X-ray data. The codebase documents an engineering progression from a local, baseline architecture to an optimized, high-throughput training pipeline deployed on a local single-node Slurm workload manager. AI-assisted.

## Dataset source:
Available from Kaggle Repository: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data).

## Architecture Development

* **CNN-1.py (Test Accuracy: 73.08%)**: Baseline pipeline with 3-layer sequential CNN.
* **CNN-2.py (Test Accuracy: 81.25%)**: With pre-trained MobileNetV2 backbone frozen with ImageNet weights.
* **CNN-3.py (Test Accuracy: 87.50%)**: Two-stage fine-tuning workflow (unfreezing the last 40 layers of the backbone) with inverse-frequency class weighting to correct dataset class imbalance.
* **CNN-4.py / submit_job.sh (Slurm Deployment - Test Accuracy: 86.86%)**: Migrating the pipeline to a headless task manager profile, with explicit scheduler memory constraints (`--mem=4G`) and CPU thread mappings (`--cpus-per-task=4`) to automate workload allocation, and implementing non-interactive `ModelCheckpoint` for intermediate weight serialization.

## Infrastructure: Slurm and WSL2

* Local single-node **Slurm** cluster within a Windows Subsystem for Linux (**WSL2**) subsystem running Python 3.12.

## Environment Setup

Steps for native `Linux` or `WSL2` instance:

1. Clone the repository to the local filesystem.
2. Construct a Python 3.12 environment (required for TensorFlow version compliance):
   ```bash
   python3.12 -m venv linux_tf_env
   source linux_tf_env/bin/activate
   ```
3. Install the verified dependency matrix:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Execution and Monitoring

1. Ensure the local `munged` security service and Slurm daemons are active.
2. Match the `/etc/slurm/slurm.conf` resource metrics to the system hardware using `slurmd -C`.
3. Clear Windows carriage returns and dispatch the script to the scheduler queue from the project path:
   ```bash
   cat submit_job.sh | tr -d '\r' > clean.sh && mv clean.sh submit_job.sh
   sudo sbatch submit_job.sh
   ```
4. Audit the task queue and track the runtime logging stream:
   ```bash
   sudo squeue
   tail -f logs_training_*.out
   ```
