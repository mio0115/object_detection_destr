# !/bin/sh

IMAGE_NAME="project/face_det_torch"
IMAGE_TAG="train"
CONTAINER_NAME="face_det_container"
PATH_TO_PROJECT="/home/daniel/cv_project/face_detection_torch"
WORKDIR="/workspace"

LEARNING_RATE=1e-5
LEARNING_RATE_BACKBONE=1e-5
AUGMENT_FACTOR=5
EPOCHS=10
BATCH_SIZE=8
SET_COST_CLASS=0.2
SET_COST_CIOU=0.2
SET_COST_BBOX=0.5
MODEL_WEIGHT_NAME="model_weights.pth"
DEVICE="cuda"
SAVE_AS="model.weights.pth"
NUM_ENC=6
NUM_DEC=6
TOPK=300
CLS_NUM=2
HIDDEN_DIM=256
RESUME=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --learning_rate) LEARNING_RATE="$2"; shift ;;
        -lr) LEARNING_RATE="$2"; shift ;;
        --lr_backbone) LEARNING_RATE_BACKBONE="$2"; shift;;
        --augment_factor) AUGMENT_FACTOR="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        -bs) BATCH_SIZE="$2"; shift;;
        --resume) RESUME="--resume" ;;
        --set_cost_class) SET_COST_CLASS="$2"; shift;;
        --set_cost_ciou) SET_COST_CIOU="$2"; shift;;
        --set_cost_bbox) SET_COST_BBOX="$2"; shift;;
        --model_weight_name) MODEL_WEIGHT_NAME="$2"; shift;;
        --device) DEVICE="$2"; shift;;
        --save_as) SAVE_AS="$2"; shift;;
        --number_encoder_blocks) NUM_ENC="$2"; shift;;
        -num_enc) NUM_ENC="$2"; shift;;
        --number_decoder_blocks) NUM_DEC="$2"; shift;;
        -num_dec) NUM_DEC="$2"; shift;;
        --top_k) TOPK="$2"; shift;;
        -k) TOPK="$2"; shift;;
        --class_number) CLS_NUM="$2"; shift;;
        -cls) CLS_NUM="$2"; shift;;
        --hidden_dim) HIDDEN_DIM="$2"; shift;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift # Move to the next argument
done

echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile_train .
docker image prune -f

if [[ $? -ne 0 ]]; then
    echo "Docker build failed. Exiting."
    exit 1
fi

echo "Start Docker container..."

docker run --rm -it \
        --name ${CONTAINER_NAME} \
        -v ${PATH_TO_PROJECT}/dataset:${WORKDIR}/dataset \
        -v ${PATH_TO_PROJECT}/checkpoints:${WORKDIR}/checkpoints \
        --gpus all \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        python -m src.train.train \
            --learning_rate=${LEARNING_RATE} \
            --lr_backbone=${LEARNING_RATE_BACKBONE} \
            --augment_factor=${AUGMENT_FACTOR} \
            --epochs=${EPOCHS} \
            --batch_size=${BATCH_SIZE} \
            --set_cost_class=${SET_COST_CLASS} \
            --set_cost_ciou=${SET_COST_CIOU} \
            --set_cost_bbox=${SET_COST_BBOX} \
            --model_weight_name=${MODEL_WEIGHT_NAME} \
            --device=${DEVICE} \
            --save_as=${SAVE_AS} \
            --number_encoder_blocks=${NUM_ENC} \
            --number_decoder_blocks=${NUM_DEC} \
            --top_k=${TOPK} \
            --class_number=${CLS_NUM} \
            --hidden_dim=${HIDDEN_DIM} \
            ${RESTORE_MODEL}

if [ $? -ne 0 ]; then
    echo "Docker run failed. Exiting."
    exit 1
fi