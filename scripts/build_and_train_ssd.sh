# !/bin/sh

IMAGE_NAME="project/ssd/face_det_torch"
IMAGE_TAG="train"
CONTAINER_NAME="ssd_det_container"
PATH_TO_PROJECT="/home/daniel/cv_project/face_detection_torch"
WORKDIR="/workspace"

LEARNING_RATE=1e-5
LEARNING_RATE_BACKBONE=1e-5
AUGMENT_FACTOR=5
EPOCHS=10
BATCH_SIZE=8
COEF_CLASS_LOSS=0.2
RESUME_FROM="model_weights.pth"
DEVICE="cuda"
SAVE_AS="model.weights.pth"
CLS_NUM=20
SCALE_MIN=0.2
SCALE_MAX=0.9
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
        --coef_class_loss) SET_COST_CLASS="$2"; shift;;
        --resume_from) RESUME_FROM="$2"; shift;;
        --device) DEVICE="$2"; shift;;
        --save_as) SAVE_AS="$2"; shift;;
        --class_number) CLS_NUM="$2"; shift;;
        --scale_min) SCALE_MIN="$2"; shift;;
        --scale_max) SCALE_MAX="$2"; shift;;
        -cls) CLS_NUM="$2"; shift;;
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
        -v $(pwd)/dataset:${WORKDIR}/dataset \
        -v $(pwd)/checkpoints:${WORKDIR}/checkpoints \
        -v $(pwd)/runs:${WORKDIR}/runs \
        --gpus all \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        python -m src.train.train_ssd \
            --learning_rate=${LEARNING_RATE} \
            --lr_backbone=${LEARNING_RATE_BACKBONE} \
            --augment_factor=${AUGMENT_FACTOR} \
            --epochs=${EPOCHS} \
            --batch_size=${BATCH_SIZE} \
            --coef_class_loss=${COEF_CLASS_LOSS} \
            --resume_from=${MODEL_WEIGHT_NAME} \
            --device=${DEVICE} \
            --save_as=${SAVE_AS} \
            --class_number=${CLS_NUM} \
            --scale_min=${SCALE_MIN} \
            --scale_max=${SCALE_MAX} \
            ${RESTORE_MODEL}

if [ $? -ne 0 ]; then
    echo "Docker run failed. Exiting."
    exit 1
fi