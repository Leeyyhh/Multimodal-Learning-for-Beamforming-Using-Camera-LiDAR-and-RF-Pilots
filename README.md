# Multimodal-Learning-for-Beamforming-Using-Camera-LiDAR-and-RF-Pilots
Code and Dataset for our paper: Multimodal Learning for Beamforming Using Camera, LiDAR, and RF Pilots

## Dataset

We provide the simulation dataset used in this project through Zenodo:

[Dataset on Zenodo](https://zenodo.org/records/20147422)

The shared dataset includes the following components:

- Raw camera images
- Sampled LiDAR points
- Wireless communication channel data
- Image detection results
- 3D bounding box information

The dataset is generated in a virtual simulation environment and does not contain real-world personal data.

### Raw LiDAR Point Generation

Due to the large file size, we do not directly include the full raw LiDAR point clouds in the Zenodo dataset. Instead, we provide the files and scripts needed to generate the raw LiDAR data in this GitHub repository.

The LiDAR data generation files are located in the following directory:

```text
LiDAR_generate/
