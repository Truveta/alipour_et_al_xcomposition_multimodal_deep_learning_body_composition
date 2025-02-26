import pathlib
from typing import Literal, Sequence, TypedDict

import dicom2nifti
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
from ipywidgets import IntSlider, interact
from nibabel.nifti1 import Nifti1Image
from scipy.ndimage import label
from totalsegmentator.python_api import totalsegmentator


class SegmentationStats(TypedDict):
    total_df: pd.DataFrame | None
    body_df: pd.DataFrame | None
    tissue_df: pd.DataFrame | None
    coronary_df: pd.DataFrame | None

class SegmentationResult(TypedDict):
    total_img: Nifti1Image | None
    body_img: Nifti1Image | None
    tissue_img: Nifti1Image | None
    coronary_img: Nifti1Image | None
    stats: SegmentationStats | None

def generate_segmentations(nifti_image: Nifti1Image, output_folder: pathlib.Path, person_id:str, statistics: bool = False, verbose: bool = True, quiet: bool = True, include_coronary: bool = False) -> SegmentationResult:
    result: SegmentationResult = {}
    stats: SegmentationStats = {}

    ### Task 1, total body segmentation
    if statistics: 
        total_nifti_output, total_stats_dict = totalsegmentator(nifti_image, task="total", statistics = True, verbose = verbose, quiet = quiet)
        stats['total_df'] = pd.DataFrame.from_dict(total_stats_dict, orient = 'index')
    else:
        total_nifti_output = totalsegmentator(nifti_image, task="total", statistics = False,verbose = verbose, quiet = quiet)
    nib.save(total_nifti_output, output_folder / f"{person_id}_total_masks")
    result['total_img'] = total_nifti_output

    ### Task 2, body segmentation
    if statistics: 
        body_nifti_output, body_stats_dict = totalsegmentator(nifti_image, task="body", statistics = True, quiet = quiet)
        stats['body_df'] = pd.DataFrame.from_dict(body_stats_dict, orient = 'index')
    else:
        body_nifti_output = totalsegmentator(nifti_image, task="body", statistics = False, quiet = quiet)
    nib.save(body_nifti_output, output_folder / f"{person_id}_body_masks")
    result['body_img'] = body_nifti_output
        
    
    ### Task 2, tissue composition segmentation
    if statistics: 
        tissue_nifti_output, tissue_stats_dict = totalsegmentator(nifti_image, task="tissue_types", statistics = True, quiet= quiet)
        stats['tissue_df'] = pd.DataFrame.from_dict(tissue_stats_dict, orient = 'index')
    else:
        tissue_nifti_output = totalsegmentator(nifti_image, task="tissue_types", statistics = False, quiet = quiet)
    nib.save(tissue_nifti_output, output_folder/ f"{person_id}_tissue_masks")
    result['tissue_img'] = tissue_nifti_output

    ### Task 3, coronary segmentation
    if include_coronary:
        try:
            if statistics: 
                coronary_nifti_output, coronary_stats_dict = totalsegmentator(nifti_image, task="coronary_arteries", statistics = True, quiet = quiet)
                stats['coronary_df'] = pd.DataFrame.from_dict(coronary_stats_dict, orient = 'index')
            else:
                coronary_nifti_output = totalsegmentator(nifti_image, task="coronary_arteries", statistics = False, quiet = quiet)
            nib.save(coronary_nifti_output, output_folder / f"{person_id}_coronary_masks")
            result[''] = coronary_nifti_output
        except:
            pass

    if stats:
        result['stats'] = stats

    return result

def _get_patient_id_from_dicom(directory_path: pathlib.Path) -> str:
    # Iterate over the files in the directory
    f = next(directory_path.glob('*.dcm'), None)
    if f is None:
        raise ValueError("No DICOM files found in the directory.")
    # Read the DICOM file
    dicom_data = pydicom.dcmread(str(f))
    # Extract the patient ID
    return dicom_data.PatientID

def plot_nifti_slices(nifti_ct_path: pathlib.Path):
    """
    Plots a 2x2 grid of slices from four NIfTI images with a slider to scroll through slices.
    """

    base_dir = nifti_ct_path.parent
    person_id = base_dir.name
    
    ct_img = nib.load(nifti_ct_path)
    total_img = nib.load(base_dir / f"{person_id}_total_masks.nii")
    body_img = nib.load(base_dir / f"{person_id}_body_masks.nii")
    tissue_img = nib.load(base_dir / f"{person_id}_tissue_masks.nii")
    # Load the image data
    ct_scan_data = ct_img.get_fdata()
    mask1_data = total_img.get_fdata()
    mask2_data = body_img.get_fdata()
    mask3_data = tissue_img.get_fdata()

    # Check the number of slices
    num_slices = ct_scan_data.shape[2]

    # Function to plot the images
    def plot_slices(slice_idx):
        fig, axs = plt.subplots(2, 3, figsize=(10, 10))

        axs[0, 0].imshow(ct_scan_data[:, :, slice_idx], cmap='gray')
        axs[0, 0].set_title('CT Scan')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(mask1_data[:, :, slice_idx], cmap='gray')
        axs[0, 1].set_title('Mask 1')
        axs[0, 1].axis('off')

        axs[1, 0].imshow(mask2_data[:, :, slice_idx], cmap='gray')
        axs[1, 0].set_title('Mask 2')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(mask3_data[:, :, slice_idx], cmap='gray')
        axs[1, 1].set_title('Mask 3')
        axs[1, 1].axis('off')

        axs[0, 2].imshow(np.where(mask1_data[:, :, slice_idx] == 29, 1, 0), cmap='gray')
        axs[0, 2].set_title('L3')
        axs[0, 2].axis('off')

        axs[1, 2].imshow(np.where((mask1_data[:, :, slice_idx] == 52 ) & (ct_scan_data[:, :, slice_idx] >= 130 ), 1, 0), cmap='gray')
        axs[1, 2].set_title('Aortic Calcifications')
        axs[1, 2].axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)


        plt.show()

    # Create the slider and interact
    interact(plot_slices, slice_idx=IntSlider(min=0, max=num_slices-1, step=1, value=num_slices//2))

def _calculate_voxel_volume(header, output_unit: Literal['mm', 'cm', 'm'] = 'mm') -> tuple[float, float]:
    voxel_dimensions = np.array(header.get_zooms())
    if output_unit == 'mm':
        pass
    elif output_unit == 'cm':
        voxel_dimensions = voxel_dimensions/10
    elif output_unit == 'm':
        voxel_dimensions = voxel_dimensions/1000
    else:
        print('No support for requested output unit. Will default to mm.')
    voxel_volume = np.prod(voxel_dimensions)
    pixel_area = voxel_dimensions[0]*voxel_dimensions[1]
    return voxel_volume, pixel_area

def _identify_l3_level(mask: np.ndarray, vt_index: int) -> tuple[int, int]:
    mask = np.where(mask == vt_index, 1, 0)
    # Find the slices that contain 1s
    slice_sums = np.sum(mask, axis=(0, 1))  # Sum over the first two dimensions
    slice_indices = np.where(slice_sums > 0)[0]    # Get indices of slices containing 1s

    if len(slice_indices) == 0:
        return (None, None)  # No slices contain the L3 level

    first_slice = slice_indices[0]
    last_slice = slice_indices[-1]

    return (first_slice, last_slice)

def _crop_to_nonzero_region(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Find the coordinates of non-zero regions in the mask
    non_zero_coords = np.argwhere(mask != 0)
    
    # Get the bounding box of the non-zero regions
    top_left = non_zero_coords.min(axis=0)
    bottom_right = non_zero_coords.max(axis=0) + 1  # +1 to include the bottom_right edge
    
    # Use the bounding box to crop the image
    cropped_image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], top_left[2]:bottom_right[2]]
    cropped_mask = mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], top_left[2]:bottom_right[2]]
    
    return cropped_image, cropped_mask

def _agatston_density_score(max_density: int) -> int:
    if 130 <= max_density < 200:
        return 1
    elif 200 <= max_density < 300:
        return 2
    elif 300 <= max_density < 400:
        return 3
    elif max_density >= 400:
        return 4
    else:
        return 0
    
def _identify_span(mask: np.ndarray, higher_span_limit_index: int, lower_span_limit_index: int) -> tuple[int | None, int | None]:
    mask = np.where( (mask == higher_span_limit_index) | (mask == lower_span_limit_index), 1, 0)
    # Find the slices that contain 1s
    slice_sums = np.sum(mask, axis=(0, 1))  # Sum over the first two dimensions
    slice_indices = np.where(slice_sums > 0)[0]    # Get indices of slices containing 1s

    if len(slice_indices) == 0:
        return (None, None)  # No slices contain the L3 level

    first_slice = slice_indices[0]
    last_slice = slice_indices[-1]

    return (first_slice, last_slice)

def _calculate_agatston_score(ct_image: np.ndarray, voxel_volume: float, threshold=130, too_small = 2):
    # Identify calcified lesions above the threshold
    calcified_mask = ct_image > threshold
    labeled_array, num_features = label(calcified_mask)
    total_agatston_score = 0
    num_plaques = 0
    # Iterate over each calcified lesion
    for region in range(1, num_features + 1):
        lesion_mask = labeled_array == region
        
        # Exclude regions with only one pixel
        if np.sum(lesion_mask) <= too_small: # FIXME for volume = 3
            continue
        num_plaques += 1
        lesion_volume = np.sum(lesion_mask) * voxel_volume  # Volume in qubic millimeters
        max_density = np.max(ct_image[lesion_mask])
        density_score = _agatston_density_score(max_density)
        lesion_agatston_score = lesion_volume * density_score
        total_agatston_score += lesion_agatston_score

    return total_agatston_score, num_plaques

def _calculate_aortic_calcification(orig_image: Nifti1Image, masks_total: Nifti1Image, voxel_volume: float, calc_threshold: Literal['adaptive'] | float, too_small: float) -> tuple[float, float]:
    abdominal_aorta_boundaries = _identify_span(masks_total, higher_span_limit_index = 32, lower_span_limit_index = 28)

    if not abdominal_aorta_boundaries[0] or not abdominal_aorta_boundaries[1]:
        return None,None
    
    mask = masks_total[..., abdominal_aorta_boundaries[0] : abdominal_aorta_boundaries[1]+1]
    trimmed_image = orig_image[..., abdominal_aorta_boundaries[0] : abdominal_aorta_boundaries[1]+1]
    abd_aorta = np.where(mask == 52, trimmed_image, 0)
    abd_aorta_zoomed, mask_zoomed = _crop_to_nonzero_region(trimmed_image, abd_aorta)
    if np.sum(abd_aorta) == 0:
        return None, None
    if calc_threshold == 'adaptive':   #### This part is adapted from the comp2comp package   
        # equal to one standard deviation to the left of the curve
        quant = 0.158
        quantile_median_dist = np.median(abd_aorta_zoomed) - np.quantile(
            abd_aorta_zoomed, q=quant
        )
        num_std = 3
        calc_threshold = np.median(abd_aorta_zoomed) + quantile_median_dist * num_std
   
    return _calculate_agatston_score(abd_aorta, voxel_volume, threshold = calc_threshold, too_small=too_small)
    
def _calculate_coronary_calcification(orig_image: Nifti1Image, masks_total: Nifti1Image, masks_coronary: Nifti1Image, voxel_volume: float, calc_threshold: Literal['adaptive'] | float, too_small: float) -> tuple[float, float]:
    heart_boundaries = _identify_span(masks_total, higher_span_limit_index = 51, lower_span_limit_index = 51)

    if not heart_boundaries[0] or not heart_boundaries[1]:
        return None,None
    mask = masks_coronary[..., heart_boundaries[0] : heart_boundaries[1]+1]
    trimmed_image = orig_image[..., heart_boundaries[0] : heart_boundaries[1]+1]
    coronary_arteries = np.where(mask == 1, trimmed_image, 0)
    coronary_zoomed, mask_zoomed = _crop_to_nonzero_region(trimmed_image, coronary_arteries)
    if np.sum(coronary_arteries) == 0:
        print("Coronary arteries were not identified.")
        return

    if calc_threshold == 'adaptive':   #### This part is adapted from the comp2comp package
        # equal to one standard deviation to the left of the curve
        quant = 0.158
        quantile_median_dist = np.median(coronary_zoomed) - np.quantile(
            coronary_zoomed, q=quant
        )
        num_std = 3
        calc_threshold = np.median(coronary_zoomed) + quantile_median_dist * num_std

    return _calculate_agatston_score(coronary_arteries, voxel_volume, threshold = calc_threshold, too_small=too_small)
    

class BodyCompositionMeasures(TypedDict):
    SkeletalMuscleArea: float | None
    SkeletalMuscleIndex: float | None        
    VisceralFatArea: float | None
    SubcutaneousFatArea: float | None
    VertebralBoneArea: float | None
    VertebralBoneDensity: float | None
    MuscleRadiodensity: float | None
    FatFreeArea: float | None
    IntramuscularFat: float | None
    LiverFatContent: float | None
    AorticCalcification: float | None
    AorticCalcificationNumberOfPlaques: int | None
    CoronaryCalcification: float | None
    CoronaryCalcificationNumberOfPlaques: float | None
    
    PersonId: str    

Measures = Literal[
    'SkeletalMuscleArea',
    'SkeletalMuscleIndex',
    'VisceralFatArea',
    'SubcutaneousFatArea', 
    'MuscleRadiodensity', 
    'FatFreeArea',
    'IntramuscularFat',
    'LiverFatContent',
    'AorticCalcification',
    'VertebralBoneArea',
    'VertebralBoneDensity'
]

def calculate_body_composition_measures(
        nifti_image: Nifti1Image,
        masks_dict: SegmentationResult,
        height: float = 1.75,
        fat_hu_range = (-190, -30),
        calc_threshold = 130,
        output_unit: Literal['mm', 'cm', 'm'] = 'mm',
        include_coronary = False,
        vertebra = "L3",
        measures_list: Sequence[Measures] = [
            'SkeletalMuscleArea',
            'SkeletalMuscleIndex',
            'VisceralFatArea',
            'SubcutaneousFatArea', 
            'MuscleRadiodensity', 
            'FatFreeArea',
            'IntramuscularFat',
            'LiverFatContent',
            'AorticCalcification',
            'VertebralBoneArea',
            'VertebralBoneDensity'
        ]) -> BodyCompositionMeasures:
    """
    calculates the body composition metrics based on the given nifti image and segmentation masks
    """


    orig_image = nifti_image.get_fdata()
    masks_total = masks_dict['total_img'].get_fdata()
    masks_body = masks_dict['body_img'].get_fdata()
    masks_tissue = masks_dict['tissue_img'].get_fdata()

    if include_coronary:
        masks_coronary = masks_dict['coronary_img'].get_fdata()

    voxel_volume, pixel_size = _calculate_voxel_volume(nifti_image.header, output_unit)
    
    if vertebra == "T12":
        vt_index = 32
    else:
        vt_index = 29
    l3_level = _identify_l3_level(masks_total, vt_index)
    
    if not l3_level[0]:
        print(f"Could not identify the {vertebra} level.")
        return None
    else:
        l3_level = int(np.mean(l3_level))

    body_comp: BodyCompositionMeasures = {}
    
    if 'SkeletalMuscleArea' in measures_list or 'SkeletalMuscleIndex' in measures_list:
        mask = masks_tissue[...,l3_level]
        skeletal_muscle_area = np.sum(mask == 3) * pixel_size
        body_comp['SkeletalMuscleArea'] = skeletal_muscle_area
        
    if 'SkeletalMuscleIndex' in measures_list:
        skeletal_muscle_index = skeletal_muscle_area / (height*height)
        body_comp['SkeletalMuscleIndex'] = skeletal_muscle_index
        
    if 'VisceralFatArea' in measures_list:
        mask = masks_tissue[...,l3_level]
        visceral_fat_area = np.sum(mask == 2) * pixel_size
        body_comp['VisceralFatArea'] = visceral_fat_area

    if 'SubcutaneousFatArea' in measures_list: 
        mask = masks_tissue[...,l3_level]
        subcutaneous_fat_area = np.sum(mask == 1) * pixel_size
        body_comp['SubcutaneousFatArea'] = subcutaneous_fat_area

    if 'VertebralBoneArea' in measures_list: 
        mask = masks_total[...,l3_level]
        vertebral_bone_area = np.sum(mask == vt_index) * pixel_size
        body_comp['VertebralBoneArea'] = vertebral_bone_area

    if 'VertebralBoneDensity' in measures_list: 
        vertebral_bone_density = np.mean(orig_image[masks_total == vt_index])
        body_comp['VertebralBoneDensity'] = vertebral_bone_density
        
    if 'MuscleRadiodensity' in measures_list: 
        mask = masks_tissue[...,l3_level]
        muscle_radiodensity = np.mean(orig_image[..., l3_level][mask == 3])
        body_comp['MuscleRadiodensity'] = muscle_radiodensity

    if 'FatFreeArea' in measures_list: ##### This is mass but we are calculating area. We need to make sure we are looking at the right measure.
        mask = masks_tissue[...,l3_level]
        body_mask = masks_body[...,l3_level]
        fat_free_area = (np.sum(body_mask == 1) - (np.sum(mask == 1) + np.sum(mask == 2))) * pixel_size
        body_comp['FatFreeArea'] = fat_free_area

    if 'IntramuscularFat' in measures_list: 
        mask = masks_tissue[...,l3_level]
        muscle_region = np.where(mask == 3, orig_image[...,l3_level], 0)
        intramuscular_fat_area = np.sum((muscle_region >= fat_hu_range[0]) & (muscle_region <= fat_hu_range[1])) * pixel_size
        body_comp['IntramuscularFat'] = intramuscular_fat_area

    if 'LiverFatContent' in measures_list: 
        mask = masks_total
        liver_region = np.where(mask == 5, orig_image, 0)
        liver_fat_regions = np.where((liver_region >= fat_hu_range[0]) & (liver_region <= fat_hu_range[1]),1,0)
        liver_fat_content = np.sum(liver_fat_regions) * voxel_volume
        body_comp['LiverFatContent'] = liver_fat_content

    if 'AorticCalcification' in measures_list:
        score, num_plaques = _calculate_aortic_calcification(orig_image, masks_total, voxel_volume, calc_threshold, too_small = 2)
        if score is None:
            print("Abdominal aorta is not completely present.")
        else:
            body_comp['AorticCalcification'] = score
            body_comp['AorticCalcificationNumberOfPlaques'] = num_plaques

    if 'CoronaryCalcification' in measures_list and include_coronary:
        score, num_plaques = _calculate_coronary_calcification(orig_image, masks_total, masks_coronary, voxel_volume, calc_threshold, too_small = 2)
        if score is None:
            print("Coronary arteries is not completely present.")
        else:
            body_comp['CoronaryCalcification'] = score
            body_comp['CoronaryCalcificationNumberOfPlaques'] = num_plaques
    
    return body_comp

class BodyCompositionVolumeMeasures(TypedDict):
    SkeletalMuscleVolume: float | None
    SkeletalMuscleIndex: float | None
    VisceralFatVolume: float | None
    SubcutaneousFatVolume: float | None
    MuscleRadiodensity: float | None
    FatFreeVolume: float | None
    IntramuscularFat: float | None
    LiverFatContent: float | None
    AorticCalcification: float | None
    VertebralBoneVolume: float | None
    VertebralBoneDensity: float | None
    AorticCalcification: float | None
    AorticCalcificationNumberOfPlaques: int | None
    CoronaryCalcification: float | None
    CoronaryCalcificationNumberOfPlaques: float | None
    
    PersonId: str  

VolumeMeasures = Literal['SkeletalMuscleVolume',
    'SkeletalMuscleIndex',
    'VisceralFatVolume',
    'SubcutaneousFatVolume', 
    'MuscleRadiodensity', 
    'FatFreeVolume',
    'IntramuscularFat',
    'LiverFatContent',
    'AorticCalcification',
    'VertebralBoneVolume',
    'VertebralBoneDensity'
]

def calculate_body_composition_volume_measures(
        nifti_image: Nifti1Image,
        masks_dict: SegmentationResult,
        height: float = 1.75,
        fat_hu_range = (-190, -30),
        calc_threshold = 130,
        output_unit: Literal['mm', 'cm', 'm'] = 'mm',
        include_coronary = False,
        measures_list: Sequence[VolumeMeasures] = [
            'SkeletalMuscleVolume',
            'SkeletalMuscleIndex',
            'VisceralFatVolume',
            'SubcutaneousFatVolume', 
            'MuscleRadiodensity', 
            'FatFreeVolume',
            'IntramuscularFat',
            'LiverFatContent',
            'AorticCalcification',
            'VertebralBoneVolume',
            'VertebralBoneDensity'
        ]) -> BodyCompositionVolumeMeasures:
    """
    calculates the body composition volumetric metrics based on the given nifti image and segmentation masks
    """

    orig_image = nifti_image.get_fdata()
    masks_total = masks_dict['total_img'].get_fdata()
    masks_body = masks_dict['body_img'].get_fdata()
    masks_tissue = masks_dict['tissue_img'].get_fdata()

    if include_coronary:
        masks_coronary = masks_dict['coronary_img'].get_fdata()

    voxel_volume, pixel_size = _calculate_voxel_volume(nifti_image.header, output_unit)
    
    l3_volume_level = _identify_span(masks_total, higher_span_limit_index = 32, lower_span_limit_index = 27)
    if not l3_volume_level[0] or not l3_volume_level[1]:
        # TODO only if volume
        print(f"Could not identify the whole abdomen.")
        return None

    body_comp: BodyCompositionVolumeMeasures = {}
    
    if 'SkeletalMuscleVolume' in measures_list or 'SkeletalMuscleIndex' in measures_list:
        mask = masks_tissue[...,l3_volume_level[0]:l3_volume_level[1]]
        skeletal_muscle_area = np.sum(mask == 3) * voxel_volume
        body_comp['SkeletalMuscleVolume'] = skeletal_muscle_area
        
    if 'SkeletalMuscleIndex' in measures_list:
        skeletal_muscle_index = skeletal_muscle_area / (height*height)
        body_comp['SkeletalMuscleIndex'] = skeletal_muscle_index
        
    if 'VisceralFatVolume' in measures_list:
        mask = masks_tissue[...,l3_volume_level[0]:l3_volume_level[1]]
        visceral_fat_area = np.sum(mask == 2) * voxel_volume
        body_comp['VisceralFatVolume'] = visceral_fat_area

    if 'SubcutaneousFatVolume' in measures_list: 
        mask = masks_tissue[...,l3_volume_level[0]:l3_volume_level[1]]
        subcutaneous_fat_area = np.sum(mask == 1) * voxel_volume
        body_comp['SubcutaneousFatVolume'] = subcutaneous_fat_area

    if 'VertebralBoneVolume' in measures_list: 
        mask = masks_total[...,l3_volume_level[0]:l3_volume_level[1]]
        vertebral_bone_area = np.sum((mask >= 27) & (mask <= 32)) * voxel_volume
        body_comp['VertebralBoneVolume'] = vertebral_bone_area

    if 'VertebralBoneDensity' in measures_list: 
        vertebral_bone_density = np.mean(orig_image[(masks_total >= 27) & (masks_total <= 32)])
        body_comp['VertebralBoneDensity'] = vertebral_bone_density
        
    if 'MuscleRadiodensity' in measures_list: 
        mask = masks_tissue[...,l3_volume_level[0]:l3_volume_level[1]]
        muscle_radiodensity = np.mean(orig_image[..., l3_volume_level[0]:l3_volume_level[1]][mask == 3])
        body_comp['MuscleRadiodensity'] = muscle_radiodensity

    if 'FatFreeVolume' in measures_list: ##### This is mass but we are calculating area. We need to make sure we are looking at the right measure.
        mask = masks_tissue[...,l3_volume_level[0]:l3_volume_level[1]]
        body_mask = masks_body[...,l3_volume_level[0]:l3_volume_level[1]]
        fat_free_area = (np.sum(body_mask == 1) - (np.sum(mask == 1) + np.sum(mask == 2))) *  voxel_volume
        body_comp['FatFreeVolume'] = fat_free_area

    if 'IntramuscularFat' in measures_list: 
        mask = masks_tissue[...,l3_volume_level[0]:l3_volume_level[1]]
        muscle_region = np.where(mask == 3, orig_image[...,l3_volume_level[0]:l3_volume_level[1]], 0)
        intramuscular_fat_area = np.sum((muscle_region >= fat_hu_range[0]) & (muscle_region <= fat_hu_range[1])) *  voxel_volume
        body_comp['IntramuscularFat'] = intramuscular_fat_area

    if 'LiverFatContent' in measures_list: 
        mask = masks_total
        liver_region = np.where(mask == 5, orig_image, 0)
        liver_fat_regions = np.where((liver_region >= fat_hu_range[0]) & (liver_region <= fat_hu_range[1]),1,0)
        liver_fat_content = np.sum(liver_fat_regions) * voxel_volume
        body_comp['LiverFatContent'] = liver_fat_content

    if 'AorticCalcification' in measures_list:
        score, num_plaques = _calculate_aortic_calcification(orig_image, masks_total, voxel_volume, calc_threshold, too_small = 3)
        if score is None:
            print("Abdominal aorta is not completely present.")
        else:
            body_comp['AorticCalcification'] = score
            body_comp['AorticCalcificationNumberOfPlaques'] = num_plaques

    if 'CoronaryCalcification' in measures_list and include_coronary:
        score, num_plaques = _calculate_coronary_calcification(orig_image, masks_total, masks_coronary, voxel_volume, calc_threshold, too_small = 3)
        if score is None:
            print("Coronary arteries is not completely present.")
        else:
            body_comp['CoronaryCalcification'] = score
            body_comp['CoronaryCalcificationNumberOfPlaques'] = num_plaques
    
    return body_comp


def dicom_to_comp(dicom_path, height = 1.75, output_folder = './Data/nifti', output_unit: Literal['mm', 'cm', 'm'] = 'mm', statistics = False, verbose = False, quiet = True):
    dicom_path = pathlib.Path(dicom_path)
    output_folder = pathlib.Path(output_folder)
    person_id = _get_patient_id_from_dicom(dicom_path)
    output_folder_unique = output_folder / person_id
    output_folder_unique.mkdir(exists_ok = True)

    dicom2nifti.convert_directory(dicom_path, output_folder_unique, compression=True, reorient=True)

    output_names = list(output_folder_unique.glob('*.nii.gz'))
    if len(output_names) > 1:
        print(f"Warning, multiple images detected for {person_id}")
    
    nifti_image: Nifti1Image = nib.load(output_names[0])
    
    output = generate_segmentations(nifti_image, output_folder_unique, person_id, statistics = statistics, verbose = verbose, quiet = quiet)
    body_comp = calculate_body_composition_measures(nifti_image, output, height = height, fat_hu_range = (-190, -30), calc_threshold = 'adaptive', output_unit = output_unit)
    body_comp['PersonId'] = person_id
    
    return {
        'Dict' : body_comp,
        'DataFrame': pd.DataFrame(body_comp, index=[person_id])
    }

def nifti_to_comp(
        nifti_path: pathlib.Path,
        height: float,
        output_unit: Literal['mm', 'cm', 'm'] = 'cm', 
        calc_threshold: Literal['adaptive'] | int = 'adaptive', 
        mode: Literal["L3", "T12", "Volume"] = "L3"):
    
    base_dir = nifti_path.parent
    person_id = base_dir.parent.name
    output = {
        'total_img' : nib.load(base_dir / f"{person_id}_total_masks.nii"),
        'body_img' : nib.load(base_dir / f"{person_id}_body_masks.nii"),
        'tissue_img' : nib.load(base_dir / f"{person_id}_tissue_masks.nii")
    }
    
    nifti_image = nib.load(nifti_path)

    if mode == 'Volume':
        body_comp = calculate_body_composition_volume_measures(
        nifti_image,
        output,
        height = height,
        fat_hu_range = (-190, -30),
        calc_threshold = calc_threshold,
        vertebra = mode,
        output_unit = output_unit
    )
    else:
        body_comp = calculate_body_composition_measures(
            nifti_image,
            output,
            height = height,
            fat_hu_range = (-190, -30),
            calc_threshold = calc_threshold,
            vertebra = mode,
            output_unit = output_unit
        )

    body_comp['PersonId'] = person_id
    return {
        'Dict' : body_comp,
        'DataFrame': pd.DataFrame(body_comp, index=[person_id])
    }


