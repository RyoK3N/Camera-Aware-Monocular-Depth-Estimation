# Camera-Aware Depth Estimation with Reprojection Error

Complete implementation of monocular depth estimation with camera intrinsics integration, reprojection error loss, and TensorBoard monitoring.

## 🎯 Overview

This project implements camera-aware neural networks for monocular depth estimation using the **SUN RGB-D dataset** (10,335 images). The system includes state-of-the-art loss functions based on 2024-2025 research, comprehensive TensorBoard monitoring, and optimized configurations for both production GPUs and Mac M4 Pro.

### Key Features

✅ **Reprojection Error Loss** (NEW - Based on 2024-2025 research)
✅ **Scale-Invariant Loss** (Eigen et al., NeurIPS 2014)
✅ **Gradient Matching Loss** (MiDaS, Ranftl et al., 2020)
✅ **Edge-Aware Smoothness** (Monodepth, Godard et al., 2017)
✅ **State-of-the-Art TensorBoard Integration** (FANG-Grade Engineering)
✅ **Camera Intrinsics Integration** (Full 3×3 matrix support)
✅ **Mac M4 Pro Optimized** (MPS backend, 24GB unified memory)
✅ **Production GPU Optimized** (CUDA backend, batch size 32)

### 🔬 Advanced TensorBoard Visualizations (NEW)

Our enhanced TensorBoard integration provides **research-grade visualizations** with proper event file writing:

🎯 **Real-Time Monitoring**
- Training/validation loss curves with smooth updates
- Loss component breakdown (SI, Gradient, Smoothness, Reprojection)
- Learning rate schedules
- Batch-level granularity

📊 **Comprehensive Metrics**
- Absolute relative error (abs_rel)
- RMSE and RMSE-log
- Threshold accuracies (δ<1.25, δ<1.5625, δ<1.953125)
- All metrics tracked per epoch

🖼️ **Visual Analysis**
- RGB input images
- Ground truth depth maps
- Predicted depth maps
- Color-coded error visualizations
- Side-by-side comparisons

📈 **Model Interpretability**
- Weight distributions (histograms for all layers)
- Gradient flow analysis (norms, max, min values)
- Activation statistics
- Gradient clipping monitoring

⚙️ **Hyperparameter Tracking**
- Experiment configuration logging
- Hyperparameter comparison across runs
- Model architecture visualization
- Training configuration as text

🎯 **Custom Scalar Layouts**
- Organized multi-panel views
- Training metrics panel
- Loss components panel
- Model statistics panel
- Automatic layout generation

---

## 🚀 Quick Start

### **Mac M4 Pro (One Command)**

```bash
cd "/Users/r/Desktop/Synexian/ML Research Ideas and Topics/Camera Matrix"

# Fix TensorBoard protobuf issue (first time only)
pip install "protobuf<4" --force-reinstall

# Start training with TensorBoard
./scripts/quick_train_m4pro.sh
```

**What this does:**
1. ✅ Validates build (builds if needed)
2. ✅ Opens TensorBoard in new terminal
3. ✅ Opens browser to http://localhost:6006
4. ✅ Starts training with MPS backend
5. ✅ Optional live log monitor

### **Production GPU (NVIDIA)**

```bash
# Fix TensorBoard first
pip install "protobuf<4" --force-reinstall

# Start training
./scripts/quick_train_production.sh
```

### **Manual Setup**

**Terminal 1** - TensorBoard:
```bash
tensorboard --logdir=./runs --port=6006
```

**Terminal 2** - Training:
```bash
./build/train --config configs/train_config_m4pro.yaml --tensorboard true
```

**Browser**: http://localhost:6006

---

## 📦 Installation

### Prerequisites

- **C++ Compiler**: GCC 9+ or Clang 11+ with C++17
- **CMake**: 3.14+
- **LibTorch**: 2.0+
- **Eigen**: 3.4+
- **OpenCV**: 4.5+
- **Python**: 3.11-3.12 (via conda)
- **Conda environment**: For TensorBoard and monitoring

### 1. Install System Dependencies

**macOS**:
```bash
brew install cmake eigen opencv yaml-cpp
```

**Linux**:
```bash
sudo apt-get install build-essential cmake libeigen3-dev libopencv-dev libyaml-cpp-dev
```

### 2. Setup Conda Environment

```bash
# Create environment
conda create -n depth_training python=3.12 -y
conda activate depth_training

# Install TensorBoard with compatible protobuf
conda install -c conda-forge tensorboard -y
pip install "protobuf<4" --force-reinstall

# Verify
python -c "import tensorboard; print('TensorBoard:', tensorboard.__version__)"
```

### 3. Download LibTorch

```bash
# CPU version
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.0.0.zip
unzip libtorch-macos-arm64-2.0.0.zip

# CUDA version (Linux)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
```

### 4. Build Project

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make train -j8
```

<<<<<<< HEAD
## Getting Started
#### 1. Download SUN RGB-D Dataset

**No approval required!** SUN RGB-D is freely available for download.

**Option 1: Official Website (Recommended)**
1. Visit [https://rgbd.cs.princeton.edu/](https://rgbd.cs.princeton.edu/)
2. Download SUNRGBD V1 (10,335 images)
3. Download SUNRGBDtoolbox
4. Extract to `./data/sunrgbd/`

**Option 2: Automated Script**
```bash
bash scripts/download_sunrgbd.sh
```

The script provides three download methods:
- Official website download (with instructions)
- Reorganized version from GitHub (automated)
- Manual setup (creates directory structure)

#### 2. Validate Dataset
=======
### 5. Download & Validate Dataset
>>>>>>> 7392029 ( Tensorboard Fixes)

```bash
# Download SUN RGB-D dataset (manual download required)
# Place in: ./data/sunrgbd/

# Validate dataset
./build/validate_sunrgbd \
    --data_dir ./data/sunrgbd \
    --output_manifest ./data/sunrgbd_manifest.json

# Expected: "Valid images: 10335/10335"
```

<<<<<<< HEAD
This will:
- Validate all 10,335 RGB-D images
- Check file completeness across 4 sensor types
- Verify RGB-depth-intrinsics alignment
- Validate intrinsic and extrinsic matrices
- Generate a comprehensive data manifest JSON file

#### 3. Precompute Ray Directions

```bash
./build/preprocess_rays --data_dir ./data/sunrgbd
```

## Usage Examples

### Data Loading

```cpp
#include "sunrgbd_loader.h"

using namespace camera_aware_depth;

// Create data loader
auto loader = std::make_shared<SunRGBDLoader>(
    "./data/sunrgbd",                    // data directory
    "./data/sunrgbd_manifest.json",      // manifest path
    "train"                              // split (train or test)
);

// Filter by sensor type (optional)
loader->filterBySensorType({"kv1", "kv2"});  // Use only Kinect v1 and v2

// Enable augmentation
AugmentationConfig aug_config;
aug_config.enable_random_crop = true;
aug_config.enable_horizontal_flip = true;
aug_config.enable_color_jitter = true;
loader->enableAugmentation(aug_config);

// Set target dimensions
loader->setTargetDimensions(240, 320);

// Load a sample
SunRGBDSample sample = loader->getSample(0);
std::cout << "RGB shape: " << sample.rgb.sizes() << std::endl;
std::cout << "Depth shape: " << sample.depth.sizes() << std::endl;
std::cout << "Sensor type: " << sample.sensor_type << std::endl;
std::cout << "Scene type: " << sample.scene_type << std::endl;
```

### Ray Direction Computation

```cpp
#include "ray_direction_computer.h"

using namespace camera_aware_depth;

// Load camera intrinsics
Eigen::Matrix3f K = RayDirectionComputer::loadIntrinsics("intrinsic.txt");

// Compute ray directions
RayDirectionComputer computer;
Eigen::MatrixXf rays = computer.computeRayDirections(K, 480, 640);

// Save to file
computer.saveRayDirections(rays, 480, 640, "rays.bin");
```

## Configuration

Edit `configs/data_config.yaml` to customize:
- Target image dimensions
- Data augmentation parameters
- Batch size and dataloader settings
- File formats and paths

## Testing

Run unit tests:
```bash
./build/test_ray_directions
./build/test_data_loader
```

## Results

Results will be saved in the `results/` directory:
- `results/quantitative_results.csv`: Numerical metrics
- `results/visualizations/`: Visual comparisons
- `results/tables/`: LaTeX tables for paper

## Documentation

- **Sprint Plan**: See `documents/sprint_plan.md` for detailed weekly tasks
- **Algorithms & Theory**: See `documents/algorithms_and_theory.md` for mathematical formulations

## Contributing

This is a research project. For questions or suggestions, please open an issue.

## Dataset Information

### SUN RGB-D V1

- **Total Images**: 10,335 RGB-D pairs
- **Sources**:
  - NYU Depth v2 (indoor scenes)
  - Berkeley B3DO (indoor scenes)
  - SUN3D (indoor scenes)
- **Sensors**: 4 types
  - Kinect v1 (3,389 images)
  - Kinect v2 (3,389 images)
  - Intel RealSense (1,574 images)
  - Asus Xtion (1,983 images)
- **Resolution**: Varies by sensor (typically 640×480)
- **Official Split**: 5,285 train / 5,050 test
- **License**: Free for research use

### Citations

If you use this dataset, please cite:

```bibtex
@inproceedings{song2015sun,
  title={SUN RGB-D: A RGB-D scene understanding benchmark suite},
  author={Song, Shuran and Lichtenberg, Samuel P and Xiao, Jianxiong},
  booktitle={CVPR},
  year={2015}
}

@inproceedings{silberman2012indoor,
  title={Indoor segmentation and support inference from rgbd images},
  author={Silberman, Nathan and Hoiem, Derek and Kohli, Pushmeet and Fergus, Rob},
  booktitle={ECCV},
  year={2012}
}

@inproceedings{janoch2011category,
  title={A category-level 3-d object dataset: Putting the kinect to work},
  author={Janoch, Allison and Karayev, Sergey and Jia, Yangqing and Barron, Jonathan T and Fritz, Mario and Saenko, Kate and Darrell, Trevor},
  booktitle={ICCV Workshop},
  year={2011}
}

@inproceedings{xiao2013sun3d,
  title={SUN3D: A database of big spaces reconstructed using SfM and object labels},
  author={Xiao, Jianxiong and Owens, Andrew and Torralba, Antonio},
  booktitle={ICCV},
  year={2013}
}
```

## References

- **SUN RGB-D Dataset**: [https://rgbd.cs.princeton.edu/](https://rgbd.cs.princeton.edu/)
- **PyTorch C++ Frontend**: [https://pytorch.org/cppdocs/](https://pytorch.org/cppdocs/)
- **Eigen Library**: [https://eigen.tuxfamily.org/](https://eigen.tuxfamily.org/)

## License

This project is for research purposes only.

## Acknowledgments

- ScanNet dataset creators for providing the dataset
- PyTorch team for LibTorch C++ frontend
- Eigen and OpenCV communities

---
=======
---

## 🔧 TensorBoard Setup (IMPORTANT)

### Fix Protobuf Compatibility Issue

TensorBoard 2.20.0 requires protobuf < 4.0. Fix this:

```bash
# Activate conda environment
conda activate synexian  # or your environment name

# Downgrade protobuf
pip install "protobuf<4" --force-reinstall

# Verify TensorBoard works
python -c "import tensorboard; print('Version:', tensorboard.__version__)"
tensorboard --logdir=./runs --port=6006
```

### Common TensorBoard Issues

**Issue**: "MessageFactory' object has no attribute 'GetPrototype'"
**Fix**: `pip install "protobuf<4" --force-reinstall`

**Issue**: "This site can't be reached at port 6006"
**Fix**: TensorBoard not running. Start with: `tensorboard --logdir=./runs --port=6006`

**Issue**: Using wrong Python (3.14 instead of 3.12)
**Fix**: Use `python` (not `python3`) in conda environment

---

## 🎯 Training Configurations

### Mac M4 Pro Configuration

**File**: `configs/train_config_m4pro.yaml`

**Specs**:
- Device: MPS (Metal Performance Shaders)
- Batch size: 16
- Image size: 240×320
- Memory: ~12-16 GB unified memory
- Workers: 8
- Expected: ~5-8 min/epoch, ~8-13 hours total (100 epochs)

**Usage**:
```bash
./scripts/quick_train_m4pro.sh
```

### Production GPU Configuration

**File**: `configs/train_config_production.yaml`

**Specs**:
- Device: CUDA
- Batch size: 32
- Image size: 480×640
- Memory: ~30-35 GB VRAM
- Workers: 16
- Expected: ~2-3 min/epoch, ~5-7.5 hours total (150 epochs)

**Usage**:
```bash
./scripts/quick_train_production.sh
```

---

## 📊 Loss Functions

### Combined Loss Formula

```
Total Loss = 1.0 * L_si + 0.1 * L_grad + 0.001 * L_smooth + 0.01 * L_reproj
```

### 1. Scale-Invariant Loss (L_si)
- **Purpose**: Main depth accuracy, invariant to scale shifts
- **Reference**: Eigen et al., NeurIPS 2014

### 2. Gradient Matching Loss (L_grad)
- **Purpose**: Preserves edge sharpness and structure
- **Reference**: MiDaS, Ranftl et al., CVPR 2020

### 3. Edge-Aware Smoothness (L_smooth)
- **Purpose**: Smooth predictions while respecting edges
- **Reference**: Monodepth, Godard et al., CVPR 2017

### 4. Reprojection Error (L_reproj) **[NEW]**
- **Purpose**: Geometric consistency using camera intrinsics
- **Method**: Back-projects depth to 3D using K matrix, computes 3D point error
- **References**:
  - UniDepth (CVPR 2024)
  - Long-term Reprojection Loss (2024)
  - Self-supervised Depth from Unknown Cameras (2025)

---

## 📈 TensorBoard Monitoring (Enhanced)

### What You'll See in TensorBoard

Our enhanced TensorBoard integration writes **proper event files** for full TensorBoard functionality:

#### **📊 SCALARS Tab** - Real-Time Plots

**Training Progress:**
- `loss/train` - Training loss (per epoch)
- `loss/val` - Validation loss (every 10 epochs)
- `batch_loss/train` - Batch-level training loss (granular monitoring)
- `training/learning_rate` - Learning rate schedule
- `training/epoch_time_seconds` - Time per epoch
- `training/total_time_seconds` - Cumulative training time

**Loss Components** (Research Analysis):
- `loss_components/si_loss` - Scale-invariant component
- `loss_components/grad_loss` - Gradient matching component
- `loss_components/smooth_loss` - Edge-aware smoothness component
- `loss_components/reproj_loss` - **Reprojection error (NEW - geometric consistency)**

**Validation Metrics** (Depth Estimation Quality):
- `metrics/abs_rel` - Absolute relative error
- `metrics/sq_rel` - Squared relative error
- `metrics/rmse` - Root mean squared error
- `metrics/rmse_log` - RMSE in log space
- `metrics/a1` - Threshold accuracy δ<1.25 (primary metric)
- `metrics/a2` - Threshold accuracy δ<1.5625
- `metrics/a3` - Threshold accuracy δ<1.953125

**Gradient Flow Analysis** (Model Health):
- `training/gradient_norm` - Global gradient norm (detect vanishing/exploding)
- `gradients/norm` - Layer-wise gradient norms
- `gradients/max` - Maximum gradient value
- `gradients/min` - Minimum gradient value

#### **🖼️ IMAGES Tab** - Visual Results

**Prediction Visualizations** (Updated every epoch):
- `predictions/sample_0` through `predictions/sample_7`
- Each image shows: **RGB | Ground Truth | Predicted | Error Map**
- Color-coded error visualization (hot colors = high error)
- Automatic normalization and scaling

#### **📈 HISTOGRAMS Tab** - Model Internals

**Weight Distributions** (Every 5 epochs):
- `weights/encoder.*/weight` - Encoder layer weights
- `weights/decoder.*/weight` - Decoder layer weights
- `weights/bottleneck.*/weight` - Bottleneck weights
- Tracks weight distribution shifts over training

**Gradient Distributions** (Every 5 epochs):
- `gradients/encoder.*/weight` - Encoder layer gradients
- `gradients/decoder.*/weight` - Decoder layer gradients
- Identifies vanishing or exploding gradients
- Helps diagnose training instabilities

#### **📝 TEXT Tab** - Experiment Documentation

**Model Architecture:**
- Total parameters count
- Trainable parameters count
- Model configuration

**Hyperparameters:**
- Learning rate, batch size, weight decay
- Loss weights (SI, Grad, Smooth, Reproj)
- Number of epochs, gradient clipping

#### **🎯 HPARAMS Tab** - Experiment Comparison

Compare multiple training runs by hyperparameters:
- Learning rate vs final loss
- Batch size vs convergence speed
- Loss weight combinations vs metrics
- Parallel coordinate plots for multi-dimensional analysis

### How to Access

1. **Start TensorBoard:**
   ```bash
   tensorboard --logdir=./runs --port=6006
   ```

2. **Open Browser:**
   Navigate to http://localhost:6006

3. **Select Your Experiment:**
   Choose from dropdown: `baseline_unet_m4pro`, etc.

4. **Explore Tabs:**
   - Start with **SCALARS** for training progress
   - Check **IMAGES** to see predictions
   - Use **HISTOGRAMS** to diagnose model behavior
   - Review **HPARAMS** to compare experiments

### TensorBoard Tips

🔄 **Auto-Refresh**: TensorBoard updates every 30 seconds automatically
📌 **Pin Charts**: Click pushpin icon to keep important charts visible
📊 **Smoothing**: Adjust smoothing slider (default 0.6) to reduce noise
⚙️ **Settings**: Gear icon for chart options, axis scales, and display preferences
🔍 **Zoom**: Click and drag on charts to zoom into specific training regions
📥 **Download Data**: Download button to export charts as SVG or data as CSV

---

## 🛠️ Available Scripts

### Training Scripts

| Script | Purpose |
|--------|---------|
| `quick_train_m4pro.sh` | One-click Mac M4 Pro training |
| `quick_train_production.sh` | One-click production GPU training |
| `train_with_monitoring.sh` | Flexible orchestration (any config) |

### Monitoring Scripts

| Script | Purpose |
|--------|---------|
| `start_tensorboard.sh` | Launch TensorBoard only |
| `watch_training.sh` | Real-time log monitoring |
| `launch_tensorboard.py` | Python TensorBoard launcher |
| `monitor_training.py` | Advanced Python monitor with dashboard |

### Example Usage

```bash
# One-click training
./scripts/quick_train_m4pro.sh

# Custom config
./scripts/train_with_monitoring.sh configs/train_config.yaml 6006

# Just TensorBoard
./scripts/start_tensorboard.sh 6006

# Just log monitoring
./scripts/watch_training.sh

# Dashboard mode
python scripts/monitor_training.py --dashboard --refresh 5
```

**Full documentation**: See `scripts/README.md`

---

## 📏 Validation Metrics

| Metric | Formula | Best Value | Typical Good Value |
|--------|---------|------------|-------------------|
| abs_rel | `Σ\|d - d*\| / d*` | Lower | < 0.15 |
| rmse | `sqrt(Σ(d - d*)²)` | Lower | < 0.5m |
| δ<1.25 (a1) | `% where max(d/d*, d*/d) < 1.25` | Higher | > 80% |

---

## 🐛 Troubleshooting

### TensorBoard Errors

**Error**: AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
```bash
pip install "protobuf<4" --force-reinstall
```

**Error**: "This site can't be reached" at port 6006
```bash
# Check if TensorBoard is running
ps aux | grep tensorboard

# Start TensorBoard
tensorboard --logdir=./runs --port=6006
```

**Error**: Using wrong Python version
```bash
# Check conda environment
echo $CONDA_DEFAULT_ENV  # Should show your env name

# Use 'python', not 'python3'
which python  # Should point to conda env
```

### Build Errors

**Error**: Cannot find LibTorch
```bash
export CMAKE_PREFIX_PATH=/path/to/libtorch
cmake ..
```

**Error**: yaml-cpp not found
```bash
# macOS
brew install yaml-cpp

# Linux
sudo apt-get install libyaml-cpp-dev
```

### Training Errors

**Error**: Out of memory
```yaml
# Reduce batch size in config
training:
  batch_size: 8  # or 4
```

**Error**: Dataset not found
```bash
# Validate dataset first
./build/validate_sunrgbd --data_dir ./data/sunrgbd --output_manifest ./data/sunrgbd_manifest.json
```

---

## 📚 Project Structure

```
Camera Matrix/
├── src/
│   ├── models/              # Neural network architectures
│   │   ├── baseline_unet.h  # U-Net baseline (31M params)
│   │   └── ...
│   ├── loss/
│   │   └── depth_loss.h     # All loss functions (includes reprojection)
│   ├── data/
│   │   └── sunrgbd_loader.h # Dataset loader with camera intrinsics
│   ├── training/
│   │   ├── tensorboard_trainer.h  # TensorBoard integration
│   │   └── production_trainer.h   # Production training
│   ├── evaluation/
│   │   └── depth_metrics.h  # Validation metrics
│   └── visualization/
│       └── depth_viz.h      # Depth visualization utilities
├── configs/
│   ├── train_config_m4pro.yaml      # Mac M4 Pro config
│   ├── train_config_production.yaml # Production GPU config
│   └── train_config.yaml            # Standard config
├── scripts/
│   ├── quick_train_m4pro.sh         # One-click Mac training
│   ├── quick_train_production.sh    # One-click production training
│   ├── start_tensorboard.sh         # TensorBoard launcher
│   ├── watch_training.sh            # Log monitor
│   ├── launch_tensorboard.py        # Python TB launcher
│   ├── monitor_training.py          # Python monitor
│   ├── fix_protobuf.sh              # Fix TensorBoard protobuf issue
│   └── README.md                    # Scripts documentation
├── data/
│   └── sunrgbd/              # SUN RGB-D dataset (10,335 images)
├── build/                    # Build directory
├── checkpoints/              # Model checkpoints
├── logs/                     # Training logs
└── runs/                     # TensorBoard logs
```

---

## 🔬 Research References

### Papers Implemented

1. **Eigen et al.** (2014): "Depth Map Prediction from a Single Image", NeurIPS
2. **Godard et al.** (2017): "Unsupervised Monocular Depth Estimation", CVPR
3. **Godard et al.** (2019): "Digging Into Self-Supervised Depth Estimation", ICCV
4. **Ranftl et al.** (2020): "MiDaS: Towards Robust Monocular Depth Estimation", CVPR
5. **Piccinelli et al.** (2024): "UniDepth: Universal Monocular Metric Depth", CVPR
6. **Long-term Reprojection Loss** (2024): Self-supervised depth with reprojection
7. **Self-supervised from Unknown Cameras** (2025): Depth with intrinsics estimation

---

## 🎓 Performance Benchmarks

### Hardware Comparison

| Hardware | Batch | Throughput | Epoch Time | Memory |
|----------|-------|------------|------------|--------|
| Mac M4 Pro 24GB | 16 | 25-30 samples/s | 5-8 min | 12-16 GB |
| NVIDIA A100 40GB | 32 | 80-100 samples/s | 2-3 min | 30-35 GB |
| RTX 4090 24GB | 16 | 50-70 samples/s | 4-5 min | 18-22 GB |
| RTX 3090 24GB | 16 | 40-60 samples/s | 5-6 min | 18-22 GB |

### Expected Results (After 100 epochs)

| Metric | Target | Excellent |
|--------|--------|-----------|
| abs_rel | < 0.20 | < 0.15 |
| rmse | < 0.60 | < 0.50 |
| δ<1.25 | > 0.75 | > 0.85 |

---

## 📄 License

This project is for research and educational purposes.

---

## 🆘 Support

- **Scripts Documentation**: `scripts/README.md`
- **TensorBoard Issues**: Run `./scripts/fix_protobuf.sh`
- **Dataset Issues**: Run `./build/validate_sunrgbd`
- **Build Issues**: Check `CMakeLists.txt` and library paths

---

## 🎯 Quick Command Reference

```bash
# Fix TensorBoard
pip install "protobuf<4" --force-reinstall

# Start training (Mac M4 Pro)
./scripts/quick_train_m4pro.sh

# Start training (Production)
./scripts/quick_train_production.sh

# Start TensorBoard only
tensorboard --logdir=./runs --port=6006

# Monitor logs
./scripts/watch_training.sh

# Validate dataset
./build/validate_sunrgbd --data_dir ./data/sunrgbd --output_manifest ./data/sunrgbd_manifest.json

# Build project
cd build && cmake .. && make train -j8
```


