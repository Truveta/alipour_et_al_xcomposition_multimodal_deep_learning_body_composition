# XComposition

XComposition is a multi-task, multimodal model that estimates both volumetric and L3-level body composition metrics from chest radiographs and four clinical variables: **Age, Sex, Height, and Weight**.
This model is based on our published work, which you can read [here](https://www.medrxiv.org/content/10.1101/2025.01.16.25320684v1).

## Overview

Trained on a large multi-institutional cohort using abdominal CT-based body composition metrics as ground truth, XComposition is capable of estimating:

- **L3-Level metrics:** Commonly used in literature (calculated as area in squared centimeters).
- **Volumetric metrics:** Calculated over the T12 to L5 region (measured in cubic centimeters).

## Body composition metrics

XComposition estimates a range of metrics, including:

- **Skeletal Muscle**: Area/Volume
- **Visceral Fat**: Area/Volume 
- **Subcutaneous Fat**: Area/Volume
- **Vertebral Bone**: Area/Volume
- **Fat Free**: Area/Volume
- **Intramuscular Fat**: Area/Volume
- **Vertebral Bone Density**: Mean HU 
- **Muscle Radiodensity**: Mean HU 
- **Aortic Calcification**: 
  - Agatston Score on the Abdominal Aorta
  - Number of Plaques
- **Indexes** (normalized by Height²):
  - Skeletal Muscle Index (Area/Volume)
  - Visceral Fat Index (Area/Volume)
  - Subcutaneous Fat Index (Area/Volume)
  - Fat Free Index (Area/Volume)

> **Note:** The model's performance varies across these metrics. It excels particularly in estimating Subcutaneous Fat, Visceral Fat, Vertebral Bone, and Skeletal Muscle (area, volume, or index).

## Data fusion strategies

XComposition was developed using three fusion strategies:

- **Early fusion**
- **Intermediate fusion**
- **Late fusion**

For the most accurate body composition estimates, we recommend the **late fusion** approach. Specifically:
- Use **late fusion** for volumetric measures.
- Use **intermediate fusion** for L3-level metrics.

## Body composition calculation

An example notebook is provided with the package to demonstrate usage.

### How to use

1. **Initialize the Model:**  
   Create an instance of the `XComposition` class with your desired fusion strategy and metric type (`L3` or `Volumetric`).

2. **Provide Inputs:**  
   Pass the following inputs to the model:
   - **Chest Radiograph:** 2D NumPy array.
   - **Sex:** `0` for female, `1` for male.
   - **Age:** in years.
   - **Height:** in meters.
   - **Weight:** in kilograms.

### Example

```python
from xcomposition import XComposition

# Instantiate the model with desired settings
XComp = XComposition(model_type="late", metric_type="Volumetric")

# Calculate body composition metrics
results = XComp(image, sex, age, height, weight)
```

An explain function is also available, which computes the metrics and provides occlusion-based attribution maps:

```python
# Get metrics along with attribution maps for explainability
body composition, attribution_map = XComp.explain(image, sex, age, height, weight)
```

## Example data

Due to institutional policies and patient privacy concerns, we cannot share our training data. However, you can experiment with publicly available chest radiograph datasets as toy data. Keep in mind that some of these datasets might not include all the clinical variables required for accurate calculations.

- [VinBigData Chest X-Ray Dataset](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)
- [NIH Chest X-Ray Dataset](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest)

## License

This project is licensed under the **BSD-3-Clause License**. See the [LICENSE.txt](LICENSE.txt) file for details.
