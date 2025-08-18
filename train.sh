set +x
# rm -rf ~/.triton
echo "here"
rm -rf ${HOME}/.cache/{vllm,nvidia} ${HOME}/.triton
export OMPI_MCA_opal_cuda_support=true
export OMP_NUM_THREADS=12

# export WANDB_CACHE_DIR='/opt/dlami/nvme/cache_wandb' # defaults to home directory
# export TMPDIR='/opt/dlami/nvme/tmp_openmpi' # Fixes "It appears as if there is not enough space for shared_mem_cuda_pool (the shared-memory backing file). It is likely that your MPI job will now either abort or experience performance degradation.""
# wandb login --relogin --host=https://aft-ai.wandb.io # Then go to https://aft-ai.wandb.io/oidc/login?

# Find and delete all .pyc files, __pycache__ directories, and torchinductor cache
# find . -type d -name  "__pycache__" -exec rm -r {} + && find . -type f -name "*.pyc" -delete && rm -rf /tmp/torchinductor_ubuntu/
# find ! -path "dir1" ! -path "dir2" -iname "*.mp3"

# export WANDB_DISABLED=true
export NCCL_TUNER_PLUGIN="/opt/amazon/ofi-nccl/lib64/libnccl-ofi-tuner.so"
# export NCCL_DEBUG=info
# export CUDA_LAUNCH_BLOCKING=1

# export VAL_CHECK_INTERVAL=10000 # 200 # 1000
# export VAL_CHECK_INTERVAL=40000 # 200 # 1000
export VAL_CHECK_INTERVAL=1000 # 200 # 1000
# export VAL_CHECK_INTERVAL=10 # 200 # 1000
export DEBUG=0
# export DEBUG=1
# export TORCH_NCCL_DUMP_ON_TIMEOUT=1
# export TORCH_NCCL_TRACE_BUFFER_SIZE=1024
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES="4"
# export CUDA_VISIBLE_DEVICES="0,1"
# export CUDA_VISIBLE_DEVICES="0,1"
# export CUDA_VISIBLE_DEVICES="0,1"
# export CUDA_VISIBLE_DEVICES="2,3"
# export CUDA_VISIBLE_DEVICES="0"
# export CUDA_VISIBLE_DEVICES="6,7"
export PROJECT="multitote_hole_with_cirep"
export CONFIG_BASE="configs/modified_pbe_v2.yaml"

export DIRNAME_LOGS="logs"
export TIMESTAMP=$(date +"%Y%m%d%H%M%S")
export FPATH_LOG=$(realpath "${DIRNAME_LOGS}/log_train_eval_${TIMESTAMP}.log")
touch "${FPATH_LOG}"
ln -sf "${FPATH_LOG}" "_latest.log"
echo $"Log file created at ${FPATH_LOG}"
# export LOG_FILE="logs/train.log"
# export DIRPATH_PROJECT_DATA="${HOME}/pbe/pbe_data"
# export DIRPATH_IMAGE_THEA="${DIRPATH_PROJECT_DATA}/s3-im-canonical-fnsku-achilles-canonical-images-iad"
# export DIRPATH_IMAGE_PAIR="${DIRPATH_PROJECT_DATA}/in_tote_change_segmentation"


# if [ ! -d "logs" ]; then
#   mkdir logs
# fi

# if [ -f "${LOG_FILE}" ]; then
#     echo "Log file already exists. Appending to it."
#     echo "Creating new log file."
#     # touch "${LOG_FILE}"
#     mv "${LOG_FILE}" "${LOG_FILE}.old"
#     rm -d "${LOG_FILE}.old"
# fi

# f="${LOG_FILE}"; ext="old"; target="${f}.${ext}"; while [ -e "${target}" ]; do target="${target}.${ext}"; done; mv "${f}" "${target}"

# One-liner to archive a file like 'train.log' to 'train.old.log'
# if [ -e "${LOG_FILE}" ]; then base="${LOG_FILE%.*}"; ext="${LOG_FILE##*.}"; target="${base}.old.${ext}"; while [ -e "${target}" ]; do target="${target%.*}.old.${ext}"; done && mv "${LOG_FILE}" "${target}"; fi
# (python main.py --logdir models --pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt --base configs/v1.yaml --scale_lr True && echo "Finished Training") |& tee logs/train.log
# (python main.py --logdir models --pretrained_model "pretrained_models/sd-v1-4-modified-9channel.ckpt" --base "configs/v1-bucket0-29_v2.yaml" --scale_lr True && echo "Finished Training") |& tee logs/train.log
# (PYTHONHUNTER="~Q(filename_contains='numpy'),stdlib=False,action=CallPrinter(force_colors=True)" python main.py --logdir models --pretrained_model "pretrained_models/sd-v1-4-modified-9channel.ckpt" --base="${CONFIG_BASE}" --scale_lr True --project="${PROJECT}" --val_check_interval="${VAL_CHECK_INTERVAL}" && echo "Finished Training") |& tee "${LOG_FILE}"
# (PYTHONHUNTER="~Q(module_in=['numpy','torch']),stdlib=False,action=CallPrinter(force_colors=True)" python main.py --logdir models --pretrained_model "pretrained_models/sd-v1-4-modified-9channel.ckpt" --base="${CONFIG_BASE}" --scale_lr True --project="${PROJECT}" --val_check_interval="${VAL_CHECK_INTERVAL}" && echo "Finished Training") |& tee "${LOG_FILE}"
# (PYTHONHUNTER="~Q(module_in=['numpy','torch', 'tqdm']),~Q(filename_contains='numpy'),~Q(filename_contains='torch/'),~Q(filename_contains='_distutils_hack'),stdlib=False" python main.py --logdir models --pretrained_model "pretrained_models/sd-v1-4-modified-9channel.ckpt" --base="${CONFIG_BASE}" --scale_lr True --project="${PROJECT}" --val_check_interval="${VAL_CHECK_INTERVAL}" && echo "Finished Training") |& tee "${LOG_FILE}"
# (PYTHONHUNTER="Q(function="main"),~Q(module_in=['numpy','torch', 'tqdm', 'typing_extensions', 'sympy', 'networkx', 'mpmath', 'packaging', 'dataclasses', 'scipy', 'regex', 'urllib3', 'requests', 'idna', 'transformers']),~Q(filename_contains='requests'),~Q(filename_contains='transformers'),~Q(filename_contains='import'),~Q(filename_contains='transformers'),~Q(filename_contains='idna'),~Q(filename_contains='reqeusts'),~Q(filename_contains='urllib'),~Q(filename_contains='regex'),~Q(filename_contains='<string>'),~Q(filename_contains='doc'),~Q(filename_contains='scipy'),~Q(filename_contains='dataclass'),~Q(filename_contains='packaging'),~Q(filename_contains='mpmath'),~Q(filename_contains='networkx'),~Q(filename_contains='tqdm'),~Q(filename_contains='sympy'),~Q(filename_contains='typing_extensions'),~Q(filename_contains='numpy'),~Q(filename_contains='torch/'),~Q(filename_contains='_distutils_hack'),~Q(filename_contains='abc'),~Q(filename_contains='frozen'),stdlib=False,action=CallPrinter(force_colors=True)" python main.py --logdir models --pretrained_model "pretrained_models/sd-v1-4-modified-9channel.ckpt" --base="${CONFIG_BASE}" --scale_lr True --project="${PROJECT}" --val_check_interval="${VAL_CHECK_INTERVAL}" && echo "Finished Training") |& tee "${LOG_FILE}"
# (PYTHONHUNTER="Q(source_has='if __name__'),~Q(module_in=['numpy','torch', 'tqdm', 'typing_extensions', 'sympy', 'networkx', 'mpmath', 'packaging', 'dataclasses', 'scipy', 'regex', 'urllib3', 'requests', 'idna', 'transformers']),~Q(filename_contains='requests'),~Q(filename_contains='transformers'),~Q(filename_contains='import'),~Q(filename_contains='transformers'),~Q(filename_contains='idna'),~Q(filename_contains='reqeusts'),~Q(filename_contains='urllib'),~Q(filename_contains='regex'),~Q(filename_contains='<string>'),~Q(filename_contains='doc'),~Q(filename_contains='scipy'),~Q(filename_contains='dataclass'),~Q(filename_contains='packaging'),~Q(filename_contains='mpmath'),~Q(filename_contains='networkx'),~Q(filename_contains='tqdm'),~Q(filename_contains='sympy'),~Q(filename_contains='typing_extensions'),~Q(filename_contains='numpy'),~Q(filename_contains='torch/'),~Q(filename_contains='_distutils_hack'),~Q(filename_contains='abc'),~Q(filename_contains='frozen'),stdlib=False,action=CallPrinter(force_colors=True)" python main.py --logdir models --pretrained_model "pretrained_models/sd-v1-4-modified-9channel.ckpt" --base="${CONFIG_BASE}" --scale_lr True --project="${PROJECT}" --val_check_interval="${VAL_CHECK_INTERVAL}" && echo "Finished Training") |& tee "${LOG_FILE}"
# (PYTHONHUNTER="Q(depth=2),~Q(module_in=['numpy','torch', 'tqdm', 'typing_extensions', 'sympy', 'networkx', 'mpmath', 'packaging', 'dataclasses', 'scipy', 'regex', 'urllib3', 'requests', 'idna', 'transformers']),~Q(filename_contains='requests'),~Q(filename_contains='transformers'),~Q(filename_contains='import'),~Q(filename_contains='transformers'),~Q(filename_contains='idna'),~Q(filename_contains='reqeusts'),~Q(filename_contains='urllib'),~Q(filename_contains='regex'),~Q(filename_contains='<string>'),~Q(filename_contains='doc'),~Q(filename_contains='scipy'),~Q(filename_contains='dataclass'),~Q(filename_contains='packaging'),~Q(filename_contains='mpmath'),~Q(filename_contains='networkx'),~Q(filename_contains='tqdm'),~Q(filename_contains='sympy'),~Q(filename_contains='typing_extensions'),~Q(filename_contains='numpy'),~Q(filename_contains='torch/'),~Q(filename_contains='_distutils_hack'),~Q(filename_contains='abc'),~Q(filename_contains='frozen'),stdlib=False,action=CallPrinter(force_colors=True)" python main.py --logdir models --pretrained_model "pretrained_models/sd-v1-4-modified-9channel.ckpt" --base="${CONFIG_BASE}" --scale_lr True --project="${PROJECT}" --val_check_interval="${VAL_CHECK_INTERVAL}" && echo "Finished Training") |& tee "${LOG_FILE}"
# (PYTHONHUNTER="Q(depth=2),stdlib=False,action=CallPrinter(force_colors=True)" python main.py --logdir models --pretrained_model "pretrained_models/sd-v1-4-modified-9channel.ckpt" --base="${CONFIG_BASE}" --scale_lr True --project="${PROJECT}" --val_check_interval="${VAL_CHECK_INTERVAL}" && echo "Finished Training") |& tee "${LOG_FILE}"
# (PYTHONHUNTER="Q(depth=3),action=CallPrinter(force_colors=True)" python main.py --logdir models --pretrained_model "pretrained_models/sd-v1-4-modified-9channel.ckpt" --base="${CONFIG_BASE}" --scale_lr True --project="${PROJECT}" --val_check_interval="${VAL_CHECK_INTERVAL}" && echo "Finished Training") |& tee "${LOG_FILE}"
# export TORCH_COMPILE_DISABLE=1
# export PYTHONHUNTER="Q(depth=2),action=StackPrinter(force_colors=True)"
# (echo "Started training. See log at ${FPATH_LOG}"; python main.py --logdir models --pretrained_model "pretrained_models/sd-v1-4-modified-9channel.ckpt" --base="${CONFIG_BASE}" --scale_lr True --project="${PROJECT}" --val_check_interval="${VAL_CHECK_INTERVAL}"; echo "Finished Training. See log at ${FPATH_LOG}") |& tee "${FPATH_LOG}"
(echo "Started training. See log at ${FPATH_LOG}"; python main.py --logdir models --pretrained_model "pretrained_models/sd-v1-4-modified-9channel.ckpt" --base="${CONFIG_BASE}" --scale_lr False --project="${PROJECT}" --val_check_interval="${VAL_CHECK_INTERVAL}"; echo "Finished Training. See log at ${FPATH_LOG}") |& tee "${FPATH_LOG}"
echo "Finished Training. See log at ${FPATH_LOG}"
# (PYTHONHUNTER="~Q(module_in=['numpy','torch', 'tqdm']),~Q(filename_contains=['numpy', 'torch/', '_distutils_hack']),stdlib=False" python main.py --logdir models --pretrained_model "pretrained_models/sd-v1-4-modified-9channel.ckpt" --base="${CONFIG_BASE}" --scale_lr True --project="${PROJECT}" --val_check_interval="${VAL_CHECK_INTERVAL}" && echo "Finished Training") |& tee "${LOG_FILE}"
