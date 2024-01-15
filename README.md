# Verification for Object Detection – IBP IoU


![Licence](https://img.shields.io/github/license/NoCohen66/Verification4ObjectDetection)

## IBP IoU an approach for the formal verificaion of object detection models.


To verify stability, we need to rely on the
*Intersection over Union* [(IoU)](https://en.wikipedia.org/wiki/Jaccard_index), a common metric for evaluating the performance of object detection. 


<br>
<!-- Badge section -->
<!--<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.6, 3.7, 3.8-efefef">
    </a>
</div>-->
<div align="center">
    <img src="results/images/fig_impact.png" width="100%" alt=" Impact_of_a perturbation_on_the_object detection" align="center" />
</div>
<br>

## Abstract

We introduce a novel Interval Bound Propagation (IBP) approach for the formal verification of object detection models, specifically targeting the Intersection over Union (IoU) metric.
The approach is compatible with popular abstract interpretation based verification tools.
The resulting verifier is evaluated on landing approach runway detection and handwritten digit recognition case studies.
Comparisons against a baseline (Vanilla IBP IoU) highlight the superior performance of Optimal IBP IoU in ensuring accuracy and stability, contributing to more secure and robust machine learning applications. 

## Software implementation


<div align="center">
    <img src="results/images/overview_IBP_IoU_approach.png" width="100%" alt="overview_IBP_IoU_approach" align="center" />
</div>


We propose a two-step approach as shown in the figure above.
* Step 1 **solver**: we apply a perturbation on the input and utilize [Auto-LIRPA](https://github.com/Verified-Intelligence/auto_LiRPA) for verifying reachable outputs. The output comprises extended bounding boxes, defined not by fixed coordinates, but by reachable intervals for each coordinate. Source codes are in the `detection` folder.
* Step 2 **IBP IoU**:  We estimate the propagation effect on the IoU. Source codes are in the `iou_calculator` folder. 

## Getting started

### Download

You can download a copy of all the files in this repository by cloning the git repository:
`git clone https://github.com/NoCohen66/Verification4ObjectDetection.git`

### Run

To run the code, you can use the following command: 
```
python main.py
```

### Configuration

In the `main.py` script of the Verification4ObjectDetection repository, various parameters are set up for running experiments. Here's an example using the default values:

- `--dataset_model`: Specifies the dataset and model to use. Default is "LARD", which likely refers to a specific dataset/model combination in the domain of object detection.
- `--eps_list_whitenoise`: Sets the range of variation for whitenoise perturbation. The default range is from 0 to 0.002, divided into 10 intervals.
- `--eps_list_brightness`: Similar to whitenoise, this argument sets the range for brightness perturbation, also defaulting to a range from 0 to 0.002 over 10 intervals.
- `--eps_list_contrast`: Specifies the range for contrast perturbation, with a default range from 0 to 0.01 over 10 intervals.
- `--methods_list`: Defines the methods used to compute bounds. By default, it includes 'IBP', 'IBP+backward (CROWN-IBP)', and 'backward (CROWN)'.
- `--nb_images`: Determines the number of images to be processed. The default value is set to 40.



## Setup


- **`pip`** install: You can use the `requirements.txt` to install packages through:
```
pip install -r requirements.txt
```


## Licence

This project is licensed under the GNU Lesser General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.
 

For more information on the GNU LGPL v3, please visit [LGPL-3.0.html](https://www.gnu.org/licenses/lgpl-3.0.html).

