#!/bin/bash
# Bash script to submit a SLURM job for training the RNN

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    -c|--current_run)
        CURRENT_RUN="$2"
        shift 2     # positional parameters are shifted to the left by N. $# is reduced by N
        ;;          # syntax for case statement
    ## network architecture options
    -hu|--hidden_units)
        NUM_HIDDEN_UNITS=$2
        shift 1
        ;;
    -l|--layers)
        NUM_LAYERS=$2
        shift 2
        ;;
    -uni|--unidirectional)
        UNIDIRECTIONAL=$2
        shift 2
        ;;
    -ct|--cell_type)
        CELL_TYPE="$2"
        shift 2
        ;;
    ## input data options
    -sf|--sampling_frequency)
        SAMPLING_FREQUENCY=$2
        shift 2
        ;;
    -tw|--time_window_duration)
        TIME_WINDOW_DURATION=$2
        shift 2
        ;;
    -ed|--example_duration)
        EXAMPLE_DURATION=$2
        shift 2
        ;;
    ## training options
    -g|--gpus)
        GPUS=$2
        shift 2
        ;;
    -ld|--loss_domain)
        LOSS_DOMAIN=$2
        shift 2
        ;;
    -bs|--batch_size)
        BATCH_SIZE=$2
        shift 2
        ;;
    -lr|--learning_rate)
        LEARNING_RATE=$2
        shift 2
        ;;
    -e|--epochs)
        TOTAL_EPOCHS=$2
        shift 2
        ;;
    -ste|--starting_epoch)
        STARTING_EPOCH=$2
        shift 2
        ;;
    -esi|--epoch_save_interval)
        EPOCH_SAVE_INTERVAL=$2
        shift 2
        ;;
    -evi|--epoch_val_interval)
        EPOCH_VAL_INTERVAL=$2
        shift 2
        ;;
    -eei|--epoch_eval_interval)
        EPOCH_EVAL_INTERVAL=$2
        shift 2
        ;;
    ## other options
    -lm|--load_model)
        LOAD_MODEL="$2"
        shift 2
        ;;
    -ll|--load_last)
        LOAD_LAST=TRUE
        shift 1
        ;;
    -m|--mode)
        MODE="$2"
        shift 2
        ;;
    ## file system options
    --data_dir)
        DATA_DIR="$2"
        shift 2
        ;;
    --predict_data_dir)
        PREDICT_DATA_DIR="$2"
        shift 2
        ;;
    --runs_dir)
        RUNS_DIR="$2"
        shift 2
        ;;
    ## slurm options
    -t|--time)
        TIME="$2"
        shift 2
        ;;
    *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift 1 # shift past argument
        ;;
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo CURRENT_RUN             = "${CURRENT_RUN}"
echo
echo NUM_HIDDEN_UNITS        = ${NUM_HIDDEN_UNITS}
echo NUM_LAYERS              = ${NUM_LAYERS}
echo UNIDIRECTIONAL          = $UNIDIRECTIONAL
echo CELL_TYPE               = "${CELL_TYPE}"
echo
echo SAMPLING_FREQUENCY      = $SAMPLING_FREQUENCY
echo TIME_WINDOW_DURATION    = $TIME_WINDOW_DURATION
echo EXAMPLE_DURATION        = $EXAMPLE_DURATION
echo
echo LOSS_DOMAIN             = "${LOSS_DOMAIN}"
echo BATCH_SIZE              = ${BATCH_SIZE}
echo LEARNING_RATE           = ${LEARNING_RATE}
echo TOTAL_EPOCHS            = ${TOTAL_EPOCHS}
echo STARTING_EPOCH          = ${STARTING_EPOCH}
echo EPOCH_SAVE_INTERVAL     = ${EPOCH_SAVE_INTERVAL}
echo EPOCH_VALIDATE_INTERVAL = ${EPOCH_VALIDATE_INTERVAL}
echo EPOCH_EVALUATE_INTERVAL = ${EPOCH_EVALUATE_INTERVAL}
echo
echo LOAD_MODEL              = "${LOAD_MODEL}"
echo LOAD_LAST               = ${LOAD_LAST}
echo MODE                    = "${MODE}"
echo
echo DATA_DIR                = "${DATA_DIR}"
echo PREDICT_DATA_DIR        = "${PREDICT_DATA_DIR}"
echo RUNS_DIR                = "${RUNS_DIR}"
echo
echo TIME                   = "${TIME}"

sbatch << -SLURM
    #!/bin/bash
    ## set SBATCH directives
    #SBATCH --job-name=${CURRENT_RUN}
    #SBATCH --output=${CURRENT_RUN}-%j.out
    #SBATCH --error=${CURRENT_RUN}-%j.out
    #SBATCH --partition=normal
    #SBATCH --time=02:00:00
    #SBATCH --nodes=1
    #SBATCH --mem=4G

    ## setup environment
    module purge
    module load python/3.6.1

    ## run code
    python3 main.py --current_run $CURRENT_RUN
SLURM
