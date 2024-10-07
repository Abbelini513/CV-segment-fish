# Fish Segmentation using Color Spaces

This repository contains an implementation of fish segmentation using color space properties. The goal is to segment fish from the background by analyzing their color features. The segmentation performance is evaluated using the Intersection over Union (IoU) metric between your masks and the ground truth annotations.

## Project Description

In this task, you are required to implement the `segment_fish` method in the `main.py` script. This method will take an image in BGR format and return a binary mask highlighting the fish regions.

### Task

- Implement the fish segmentation algorithm in `segment_fish`.
- Evaluate the quality of the segmentation using the IoU metric on the training dataset.
- Visualize and verify your results using `main.ipynb` notebook.

### Dataset

You can find the training images in the `dataset/train` folder. The dataset consists of images of fish with corresponding mask annotations. Use these to develop and test your segmentation method.

### Usage

To run the segmentation on the training data and calculate IoU, run the following command:

```
python main.py --is_train
```

This will display the IoU score on the training data, helping you evaluate your segmentation performance.

### Evaluation Criteria

You can use the metric based on the IoU score:
- **IoU > 0.45**: 15 points (Satisfactory)
- **IoU > 0.50**: 18 points (Average)
- **IoU > 0.55**: 20 points (Good)

## Implementation Details

### `segment_fish` in `main.py`

The method `segment_fish` is implemented using the following approach:
1. **HSV Color Space**: The image is converted to HSV format, where specific ranges of colors (for fish and background) are easier to isolate.
2. **Color Thresholding**: Color ranges for the fish (orange) and white stripes are defined to create two masks:
    - An orange color mask for the fish body.
    - A white mask for the stripes.
3. **Morphological Operations**: The masks are cleaned and refined using morphological operations to remove noise and fill gaps.
4. **Mask Combination**: The final mask is created by combining the color masks.

### Analysis in `CV3.ipynb`

The `CV3.ipynb` notebook contains exploratory data analysis (EDA) and research into the best color ranges for segmenting fish. This includes:
- Loading and displaying images and masks.
- Investigating the best HSV ranges to segment the fish.
- Applying morphological transformations to clean up the masks.

### Visualization and Testing in `main.ipynb`

The `main.ipynb` notebook is used for:
- Visualizing the segmented masks alongside the original images.
- Computing the IoU metric to assess segmentation quality.
- Running experiments with the provided `segment_fish` function.

## Results

By running `main.py`, you will get the IoU score for your segmentation method. Use the IoU metric to improve your algorithm iteratively.
