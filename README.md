# GenerateAnomaliesFromNormal
Generate Anomalies From Normal:A Partial Pseudo Anomaly Augmented Approach For Video Anomaly Detection

# Official PyTorch implementation of "Generate Anomalies From Normal:A Partial Pseudo Anomaly Augmented Approach For Video Anomaly Detection"

## Dependencies
* Python 3.6
* PyTorch = 1.7.0 
* Numpy
* Sklearn

## Datasets
* USCD Ped2 [[dataset](https://drive.google.com/file/d/1w1yNBVonKDAp8uxw3idQkUr-a9Gj8yu1/view?usp=sharing)]
* CUHK Avenue [[dataset](https://drive.google.com/file/d/1q3NBWICMfBPHWQexceKfNZBgUoKzHL-i/view?usp=sharing)]
* ShanghaiTech [[dataset](https://drive.google.com/file/d/1rE1AM11GARgGKf4tXb2fSqhn_sX46WKn/view?usp=sharing)]
* CIFAR-100 (for object-based pseudo anomalies)

Download the datasets into ``dataset`` folder, like ``./dataset/ped2/``, ``./dataset/avenue/``, ``./dataset/shanghai/``, ``./dataset/cifar100/``, ``./dataset/imagenet/``

## Training
```bash
git clone https://github.com/aseuteurideu/LearningNotToReconstructAnomalies](https://github.com/OctCjy/GenerateAnomaliesFromNormal
```

* Training baseline
```bash
python trainobjectloss.py --dataset_type ped2
```

* Training pseudo-anomaly model
```bash
python trainobjectloss.py --dataset_type ped2 --pseudo_anomaly_mask 0.2 --object_loss_weight 0.5 
```

Select --dataset_type from ped2, avenue, or shanghai.

For more details, check trainobjectloss.py


## Pre-trained models
| Model           | Dataset       | AUC           | Weight        |
| -------------- | ------------- | ------------- | ------------- | 
|pseudo-anomaly | Ped2          |   92.49%       |  |
|pseudo-anomaly | Avenue        |   85.35%       |  |
|pseudo-anomaly | ShanghaiTech  |   73.31%       |  |

## Evaluation
* Test the model
```bash
python evaluate.py --dataset_type ped2 --model_dir path_to_weight_file.pth
```
* Test the model and save result image
```bash
python evaluate.py --dataset_type ped2 --model_dir path_to_weight_file.pth --img_dir folder_path_to_save_image_results
```
* Test the model and generate demonstration video frames
```bash
python evaluate.py --dataset_type ped2 --model_dir path_to_weight_file.pth --vid_dir folder_path_to_save_video_results
```
Then compile the frames into video. For example, to compile the first video in ubuntu:
```bash
ffmpeg -framerate 10 -i frame_00_%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p video_00.mp4
```


## Acknowledgement
The code is built on top of code provided by Astrid et al. [https://github.com/aseuteurideu/LearningNotToReconstructAnomalies] and Gong et al. [https://github.com/donggong1/memae-anomaly-detection]
