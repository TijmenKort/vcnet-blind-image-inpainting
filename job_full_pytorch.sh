#!/bin/bash
#Set job requirements
#SBATCH -t 72:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=gtx1080ti:4
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jlp.kort@gmail.com

module load 2019

pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

#Copy input data to scratch and create output directory
#cp -r $HOME/vcnet-blind-image-inpainting "$TMPDIR"
#mkdir "$TMPDIR"/output_dir

#Run program
python main.py --base_cfg config.yml


#Copy output data from scratch to home
#cp -r "$TMPDIR"/output_dir $HOME