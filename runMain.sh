#!/bin/bash
# Bash script to submit a SLURM job for training the RNN
# Inputs:
# $1 = call to main.py with input arguments
# $2 = current_run_name

## example command call
# runMain.sh "python3 main.py --current_run hello_world --layers 3" hello_world
# (main.py will parse all the input arguments)

echo $1
echo $2

### error check the inputs
# there must be at least 2 inputs
if (($#<2))
then
    echo 'not enough input variables'
    exit
fi


## run on rice servers
if [[ "$HOSTNAME" = *"rice"* ]]; then
sbatch <<SLURM
#!/bin/bash
## set SBATCH directives
#SBATCH --job-name="$2"
#SBATCH --output="$2"-%j.out
#SBATCH --error="$2"-%j.out
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem=4G

## setup environment
source venv/bin/activate

## run code
$1
SLURM
fi


# run on sherlock servers
if [[ "$HOSTNAME" = *"sh-ln"* ]]; then
sbatch <<SLURM
#!/bin/bash
## set SBATCH directives
#SBATCH --job-name="$2"
#SBATCH --output="$2"-%j.out
#SBATCH --error="$2"-%j.out
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem=4G

## setup environment
module purge
module load python/3.6.1
source venv/bin/activate

## run code
$1
SLURM
fi

