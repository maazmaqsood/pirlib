EXAMPLEDIR=$(dirname $0)
ROOTDIR=$EXAMPLEDIR/../..

### Module 1: Docker_Packaging
python $ROOTDIR/bin/pircli dockerize \
    $ROOTDIR \
	--auto \
    --pipeline examples.t5_fine_tuning.tuning:t5_fine_tuning \
	--output $EXAMPLEDIR/package_argo.yml \
	--flatten \
    --docker_base_image nvidia/cuda:11.0.3-base-ubuntu20.04


# # Convert EXAMPLEDIR to absolute path since docker can't bind-mount relative paths.
EXAMPLEDIR=$([[ $EXAMPLEDIR = /* ]] && echo "$EXAMPLEDIR" || echo "$PWD/${EXAMPLEDIR#./}")

### Module 2: Argoize_Module
mkdir -p $EXAMPLEDIR/outputs
mkdir -p $EXAMPLEDIR/cache_dir
INPUT_dataset=$EXAMPLEDIR/inputs \
INPUT_raw_data=$EXAMPLEDIR/inputs \
INPUT_hparams=$EXAMPLEDIR/inputs/hparams.json \
INPUT_data_preproc_hp=$EXAMPLEDIR/inputs/stage_1_hparams.json \
INPUT_tuning_hp=$EXAMPLEDIR/inputs/stage_2_hparams.json \
INPUT_distillation_hp=$EXAMPLEDIR/inputs/hparams.json \
CACHE=$EXAMPLEDIR/cache_dir \
OUTPUT=$EXAMPLEDIR/outputs \
NFS_SERVER=k8s-master.cm.cluster \
python  $ROOTDIR/bin/pircli generate $EXAMPLEDIR/package_argo.yml \
	--target pirlib.backends.argo_batch:ArgoBatchBackend \
	--output $EXAMPLEDIR/argo-fine-tuning.yml

# Run the Argo workflow
argo submit -n argo --watch $EXAMPLEDIR/argo-fine-tuning.yml
