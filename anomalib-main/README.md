# Semi-Supervised-Learning-for-Anomaly-Traffic-Detection-via-Bidirectional-Normalizing-Flows

Our project is implemented based on [Anomalib](https://github.com/openvinotoolkit/anomalib/tree/main). We integrate the code of our model into Anomalib, allowing for execution using the mature framework of Anomalib.

Our code of model is in anomalib-main/anomalib/models/semiflow, which contains the code of all modules in torch_model.py, the code of all loss functions in loss.py and the code of training steps, validation steps in lightning_model.py.

The saved parameters of the modules can be found in /anomalib-main/anomalib/model_save, which can be obtained from the following link provided by anonymous accounts:https://github.com/Anunknownresearcher/save_models/releases/tag/save_models

The one_dimension_packet.py in /anomalib-main/anomalib/data is the data module special for traffic.

The implementation details of our model can be found in /anomalib-main/configs/model/semiflow_config.yaml.

## Local Install 
The dependency lists are in the /anomalib-main/requirements. We highly recommend utilizing the Local Install method to install Anomalib.

```python
cd anomalib-main
pip install -e .
```

## Training
To train our model on a specific dataset and category, the config file is to be provided:
```python
python tools/train.py --config /anomalib-main/configs/model/semiflow_config.yaml
```

## Evaluation
To evluate our model on a specific dataset and category, the config file is to be provided:
```python
python tools/test.py --config /anomalib-main/configs/model/semiflow_config.yaml --weight_file <path/to/model/weight.pth>
```

## Pre-trained models
The pre-trained modules of our model in /anomalib-main/anomalib/model_save.

## Table of results
### 

| Method    |    Reverse Distillation             |    DFKDE    |  DFM   |   DRAEM    |  FastFlow  |   PaDiM    |   PatchCore    |  STFPM   |   CFLOW   |  GANomaly  | GANomaly_1d  | Ours |
| --------- | ------------------ | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| **VPN**   | 0.6116 | 0.5907 |   0.7156   |   0.5698   |   0.6195   | 0.6726 | 0.7058 | 0.5657 | 0.5433 | 0.6239 |  0.5913   | 0.8822 |
| **TOR**    | 0.7450|   0.7356   |   0.7514   |   0.7028   |   0.6689   |   7516   |   0.7434   |   0.7371   |   0.7025   |   0.7823   |   0.7166   |   0.8784   |
| **DataCon** | 0.6762     |   0.3969   |   0.6744   |   0.6479   | 0.6571   |  0.6768   |   0.4605   |   0.6292   | 0.5850 |   0.6871   |  0.6884  |   0.7063   |

