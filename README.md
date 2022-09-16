# [Draft] Campanile Dataset Release

This repository contains the dataset release for the Netflix Technology Blog article [insert article title and link here]. In this blogpost, we re-rendered the [1997 Campanile Movie](https://www.pauldebevec.com/Campanile/) by applying the latest radiance field methods on newly captured drone videos. More specifically, we used the official codebase of [Instant-NGP](https://nvlabs.github.io/instant-ngp/), [Plenoxels](https://alexyu.net/plenoxels/) and [Mip-NeRF 360](https://github.com/google-research/multinerf), trained on drone footage, and rendered from the same camera trajectory of the original Campanile Movie.

Here we release the 1997 Campanile Movie, the 2022 drone capture and our trained models. All models are trained on a single NVIDIA RTX 3090 GPU, and use the same [COLMAP](https://colmap.github.io/index.html) results as input. The COLMAP is reconstructed using the undistorted drone video frames from `DJI_0074.MP4` only. In `extra-drone-videos` we include additional panorama and drone videos, which provide extra viewpoints and details and using them could yield better reconstruction quality.


## Dataset

The dataset can be downloaded [here](https://drive.google.com/drive/folders/1VJcLBrlGQo2qbym8NGFahQZS8-1FFnQJ?usp=sharing).

### Structure of the data folder ($DATA_DIR)
<pre>
campanile-movie/                        # High resolution frames from the 1997 short film
    campanile-movie.mp4                 # 60fps video
    images/                             # 1491 video frames with resolution 1200x900, sampled at 60fps
    
drone-video/                            # The drone video used for training
    DJI_0074.MP4                        # The video
    images/                             # 305 video frames with resolution 3840x2160, sampled at 1fps
    images-undistorted/                 # The images folder undistorted using COLMAP, resolution 3757x2110

extra-drone-videos/                     # Extra videos and panorama. All videos share intrinsics.
    DJI_0077/
        DJI_0077.MP4                    # Extra video#1, close-up view of the campanile tower
        images/                         # Video frames sampled at 1fps
    DJI_0079/
        DJI_0079.MP4                    # Extra video#2, overview of campus
        images/                         # Video frames sampled at 1fps
    PANORAMA/
        DJI_0073.JPG                    # The panorama showing the campus
        100_0073/                       # Individual perspective images used for stitching the panorama
        
models/                                  # Trained models, <b>all share the same COLMAP</b>
    instant-ngp/                        # Files in the order of method workflow (COLMAP, json, msgpack, rendering)
        train_colmap_sparse/            # COLMAP from <b>undistorted</b> drone frames, share coordinate space with test COLMAP
        test_colmap_sparse/             # COLMAP from campanile frames, share coordinate space with train COLMAP
        train_transforms.json           # Camera poses of the campanile frames, converted from COLMAP
        test_transforms.json            # Camera poses of the drone frames, converted from COLMAP
        transforms.msgpack              # Trained instant-NGP model, 100000 steps
        train_renders_/                 # Drone frames rendered from trained model
        test_renders_/                  # Campanile frames rendered from trained model
        
    plenoxels/                          # Files in the order of method workflow (COLMAP, pose, ckpt and args, rendering)
        train_colmap_sparse/            # COLMAP from <b>undistorted and downsampled</b> drone frames (1252x703, to fit in memory)
        test_colmap_sparse/             # COLMAP from campanile frames, share coordinate space with train COLMAP
        images_train/                   # Downscaled training images to fit in memory
        pose_train/                     # Camera poses of the campanile frames, converted from COLMAP
        pose_test/                      # Camera poses of the drone frames, converted from COLMAP
        ckpt.npz                        # Trained Plenoxels model, 102400 steps
        args.json                       # Configurations of this trained model
        intrinsics_train.txt            # Train camera intrinsics
        intrinsics_test.txt             # Test camera intrinsics
        train_renders_/                 # Drone frames rendered from trained model
        test_renders_/                  # Campanile frames rendered from trained model
        
    mip-nerf-360/
        train_colmap_sparse/            # COLMAP from <b>undistorted and downsampled</b> drone frames (939x528, to fit in memory)
        test_colmap_sparse_part1/       # COLMAP from campanile frames (#0-743), share coordinate space with train COLMAP
        test_colmap_sparse_part2/       # COLMAP from campanile frames (#744-1490), share coordinate space with train COLMAP
        images_4/                       # Downscaled training images to fit in memory
        360_train.gin                   # Config file for training
        360_test_train.gin              # Config file for rendering train images
        360_test.gin                    # Config file for rendering test images
        drone_params.npy                # Stats computed from train COLMAP poses
        render_test.py                  # Load drone_params.npy to apply the same preprocessing to test COLMAP poses
        dataset_test.py                 # Imported in render_test.py
        rename_test.py                  # Rename Mip-NeRF 360 output
        checkpoint_120000               # Trained Mip-NeRF 360 model, 120000 steps
        train_renders_/                 # Drone frames rendered from trained model
        test_renders_part1_/            # Campanile frames (#0-743) rendered from trained model        
        test_renders_part2_/            # Campanile frames (#744-1490) rendered from trained model       
        test_renders_/                  # The above two folders merged and renamed

psnr.py                                 # script used for computing PSNR in the blog post, adapted from Plenoxels
</pre>

## Example usage of trained models
### Instant-NGP
To render the campanile camera path from trained model, from the [Instant-NGP](https://nvlabs.github.io/instant-ngp/) **root folder**, run:
```
CKPT_DIR=$DATA_DIR/models/instant-ngp

python scripts/run.py --mode nerf \
--scene $CKPT_DIR/train_transforms.json \
--load_snapshot $CKPT_DIR/transforms.msgpack \
--screenshot_transforms $CKPT_DIR/test_transforms.json \
--screenshot_dir $CKPT_DIR/test_renders \
--width 1200 --height 900 --n_steps 0 --screenshot_spp 1
```
Output saved in `$DATA_DIR/models/instant-ngp/test_renders`. Our version in `$DATA_DIR/models/instant-ngp/test_renders_`.

### Plenoxels
To render the campanile camera path from trained model, from the [Plenoxels](https://alexyu.net/plenoxels/) **root folder**, run:
<pre>
cd opt && mkdir tmp_data_dir
cp -a $DATA_DIR/campanile-movie/images/ tmp_data_dir/images                   # Copy images over
cp -a $DATA_DIR/models/plenoxels/pose_text tmp_data_dir/pose            # Copy pose over

python render_imgs.py $DATA_DIR/models/plenoxels/ckpt.npz tmp_data_dir
</pre>
Output saved in `$DATA_DIR/models/plenoxels/test_renders`. Our version in `$DATA_DIR/models/plenoxels/test_renders_`.

### Mip-NeRF 360
First, move some files over to the [Mip-NeRF 360](https://github.com/google-research/multinerf) **root folder**. The test camera trajectory is split into two to avoid out-of-memory (OOM) errors.
<pre>
TRAIN_DIR=tmp_data_dir/train
TEST_DIR_P1=tmp_data_dir/test_part1
TEST_DIR_P2=tmp_data_dir/test_part2

mkdir tmp_data_dir
mkdir $TRAIN_DIR
mkdir $TRAIN_DIR/checkpoints
mkdir $TEST_DIR_P1
mkdir $TEST_DIR_P2

# Move images
cp -a $DATA_DIR/models/mip-nerf-360/images_4 $TRAIN_DIR
cp -a $DATA_DIR/campanile-movie/images $TEST_DIR_P1
cp -a $DATA_DIR/campanile-movie/images $TEST_DIR_P2

# Move COLMAP
cp -a $DATA_DIR/models/mip-nerf-360/train_colmap_sparse $TRAIN_DIR/sparse
cp -a $DATA_DIR/models/mip-nerf-360/test_colmap_sparse_part1 $TEST_DIR_P1/sparse
cp -a $DATA_DIR/models/mip-nerf-360/test_colmap_sparse_part2 $TEST_DIR_P2/sparse

# Move checkpoint and config files
cp $DATA_DIR/models/mip-nerf-360/checkpoint_120000 $TRAIN_DIR/checkpoints
cp $DATA_DIR/models/mip-nerf-360/drone_params.npy $TEST_DIR_P1
cp $DATA_DIR/models/mip-nerf-360/drone_params.npy $TEST_DIR_P2
cp $DATA_DIR/models/mip-nerf-360/360_train.gin configs
cp $DATA_DIR/models/mip-nerf-360/360_test_train.gin configs
cp $DATA_DIR/models/mip-nerf-360/360_test.gin configs

# Move scripts
cp $DATA_DIR/models/mip-nerf-360/dataset_test.py internal
cp $DATA_DIR/models/mip-nerf-360/render_test.py .
</pre>

To render the drone camera path from trained model, run:
<pre>
python -m render \
  --gin_configs=configs/360_test_train.gin \
  --gin_bindings="Config.data_dir = '${TRAIN_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${TRAIN_DIR}/checkpoints'" \
  --gin_bindings="Config.render_dir = '${TRAIN_DIR}/render'" --logtostderr

mkdir $DATA_DIR/models/mip-nerf-360/train_render
cp $TRAIN_DIR/render/test_preds_step_120000/color* $DATA_DIR/models/mip-nerf-360/train_render
</pre>
Output saved in `$DATA_DIR/models/mip-nerf-360/train_renders`. Our version in `$DATA_DIR/models/mip-nerf-360/train_renders_`.

To render the campanile camera path from trained model, run:
<pre>
python -m render_test \
  --gin_configs=configs/360_test.gin \
  --gin_bindings="Config.data_dir = '${TEST_DIR_P1}'" \
  --gin_bindings="Config.checkpoint_dir = '${TRAIN_DIR}/checkpoints'" \
  --gin_bindings="Config.render_dir = '${TRAIN_DIR}/render_test_part1'" --logtostderr
 
python -m render_test \
  --gin_configs=configs/360_test.gin \
  --gin_bindings="Config.data_dir = '${TEST_DIR_P2}'" \
  --gin_bindings="Config.checkpoint_dir = '${TRAIN_DIR}/checkpoints'" \
  --gin_bindings="Config.render_dir = '${TRAIN_DIR}/render_test_part2'" --logtostderr
 
python $DATA_DIR/models/mip-nerf-360/rename_test.py --input_dir $TRAIN_DIR --output_dir $DATA_DIR/models/mip-nerf-360/test_render
</pre>

Output saved in `$DATA_DIR/models/mip-nerf-360/test_renders`. Our version in `$DATA_DIR/models/mip-nerf-360/test_renders_`.
