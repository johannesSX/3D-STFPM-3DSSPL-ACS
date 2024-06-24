# 3D Patch-based Student-Teacher Pyramid Matching in 3D Magnetic Resonance Imaging

PyTorch lightning implementation of A [Patch-based Student-Teacher Pyramid Matching Approach to Anomaly Detection in 3D Magnetic Resonance Imaging](https://openreview.net/forum?id=vh01Nd5PCl&noteId=vh01Nd5PCl) (MIDL 2024 Submission).


## Getting Started

Install packages with:

```
$ pip install -r requirements.txt
```

## Dataset
The data can be accessed at the following link.
- [IXI](https://brain-development.org/ixi-dataset/)
- [BRATS](http://www.braintumorsegmentation.org/)

The path to the data is `./data/SKULL/{BRATS, IXI}`.


## Train the teacher
Start training the teacher with the following line:
```
python run_stpm3D --run_mode pretrain
```

## Train the student
Start training the student with the following line:
```
python run_stpm3D --run_mode train
```

## Evaluation
The evaluation of the network can be started with the following line:
```
python run_stpm3D --run_mode eval
```

## Code References
- [Student-Teacher Feature Pyramid Matching for Anomaly Detection](https://github.com/hcw-00/STPM_anomaly_detection)
- [Reinventing 2D Convolutions for 3D Images](https://github.com/M3DV/ACSConv)


## Citation
- ...
