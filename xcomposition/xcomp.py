import random
from importlib.resources import path as rpath
from typing import Any, Literal, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler

from . import weights


class RegressionModel_EarlyFusion(nn.Module):
    def __init__(self,
                 image_size: int,
                 output_size = 15,
                 num_additional_features = 4,
                 clin_dropout = 0,
                 image_dropout = 0,
                ):
        super(RegressionModel_EarlyFusion, self).__init__()
        self.image_size = image_size
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels,
                                     kernel_size=self.model.conv1.kernel_size,
                                     stride=self.model.conv1.stride,
                                     padding=self.model.conv1.padding,
                                     bias=False) 


        num_resnet_features = self.model.fc.in_features
        self.clin_fc = nn.Linear(num_additional_features, image_size*image_size) # FIXME
        
        self.fc = nn.Linear(num_resnet_features, output_size)
        self.model.fc = nn.Identity()
        
        self.clin_dropout = clin_dropout
        self.image_dropout = image_dropout

        
    def forward(self, x_image, x_clinical):
        clin_out = self.clin_fc(x_clinical)
        clin_out = clin_out.view(-1, 1, self.image_size, self.image_size)
        x_image = x_image.view(-1, 1, self.image_size, self.image_size)

        if self.training:            
            dropout_mask = torch.bernoulli(torch.full((clin_out.size(0), clin_out.size(1)), 1 - self.clin_dropout)).to(clin_out.device)
            dropout_mask = dropout_mask.unsqueeze(2).unsqueeze(3)  # Reshape mask to match (batch_size, num_channels, 1, 1)
            clin_out = clin_out * dropout_mask  # Apply mask to input tensor
           
        #combined = torch.cat((x_image, clin_out), dim=1)
        combined = x_image + clin_out
        resnet_out = self.model(combined)
        output = self.fc(resnet_out)
        #output = self.final_activation(output)
        return output

class RegressionModel(nn.Module):
    def __init__(self, 
                 output_size = 15,
                 num_additional_features = 4,
                 clin_dropout = 0,
                 image_dropout = 0,
                 concatenate_modes = False,
                ):
        super(RegressionModel, self).__init__()
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels,
                                     kernel_size=self.model.conv1.kernel_size,
                                     stride=self.model.conv1.stride,
                                     padding=self.model.conv1.padding,
                                     bias=False) 

        num_resnet_features = self.model.fc.in_features
        
        self.clin_fc = nn.Linear(num_additional_features, num_resnet_features)
        self.fc = nn.Linear(num_resnet_features, output_size)
        self.model.fc = nn.Identity()
        self.clin_dropout = clin_dropout
        self.image_dropout = image_dropout
        self.concatenate_modes = concatenate_modes 
        if concatenate_modes:
            self.fc_con = nn.Linear(num_resnet_features*2, output_size)
        
    def forward(self, x_image, x_clinical):
        if self.concatenate_modes:
            resnet_out = self.model(x_image)
            clin_fc_out = self.clin_fc(x_clinical)
            output = self.fc_con(torch.cat([resnet_out, clin_fc_out], dim =1))
        else:
            if (self.training and random.random()<self.clin_dropout) or self.clin_dropout == 1:
                #print('ignoring clinical data')
                resnet_out = self.model(x_image)
                output = self.fc(resnet_out)
            elif (self.training and random.random()<self.image_dropout) or self.image_dropout == 1:
                #print('ignoring imaging data')
                clin_fc_out = self.clin_fc(x_clinical)
                output = self.fc(clin_fc_out)
            else:
                #print('using both clinical and imaging data')
                resnet_out = self.model(x_image)
                clin_fc_out = self.clin_fc(x_clinical)
                output = self.fc(resnet_out+clin_fc_out)
            
        #output = self.final_activation(output)
        return output
    
class LateFusionFC(nn.Module):
    def __init__(self, n):
        super(LateFusionFC, self).__init__()
        # Define a fully connected layer
        self.fc = nn.Linear(2 * n, n)
        
    def forward(self, x):
        # Forward pass through the fully connected layer
        x = self.fc(x)
        return x


class VolumetricBodyComposition(TypedDict):
    SkeletalMuscleVolume: float
    SkeletalMuscleIndex: float
    VisceralFatVolume: float
    SubcutaneousFatVolume: float
    VertebralBoneVolume: float
    VertebralBoneDensity: float
    MuscleRadiodensity: float
    FatFreeVolume: float
    IntramuscularFat: float
    LiverFatContent: float
    AorticCalcification: float
    AorticCalcificationNumberOfPlaques: float
    VisceralFatIndex: float
    SubcutaneousFatIndex: float
    FatFreeIndex: float

class BodyComposition(TypedDict):
    SkeletalMuscleArea: float
    SkeletalMuscleIndex: float
    VisceralFatArea: float
    SubcutaneousFatArea: float
    VertebralBoneArea: float
    VertebralBoneDensity: float
    MuscleRadiodensity: float
    FatFreeArea: float
    IntramuscularFat: float
    LiverFatContent: float
    AorticCalcification: float
    AorticCalcificationNumberOfPlaques: float
    VisceralFatIndex: float
    SubcutaneousFatIndex: float
    FatFreeIndex: float

class VolumetricBodyCompositionAttribution(TypedDict):
    SkeletalMuscleVolume: Any
    SkeletalMuscleIndex: Any
    VisceralFatVolume: Any
    SubcutaneousFatVolume: Any
    VertebralBoneVolume: Any
    VertebralBoneDensity: Any
    MuscleRadiodensity: Any
    FatFreeVolume: Any
    IntramuscularFat: Any
    LiverFatContent: Any
    AorticCalcification: Any
    AorticCalcificationNumberOfPlaques: Any
    VisceralFatIndex: Any
    SubcutaneousFatIndex: Any
    FatFreeIndex: Any

class BodyCompositionAttribution(TypedDict):
    SkeletalMuscleArea: Any
    SkeletalMuscleIndex: Any
    VisceralFatArea: Any
    SubcutaneousFatArea: Any
    VertebralBoneArea: Any
    VertebralBoneDensity: Any
    MuscleRadiodensity: Any
    FatFreeArea: Any
    IntramuscularFat: Any
    LiverFatContent: Any
    AorticCalcification: Any
    AorticCalcificationNumberOfPlaques: Any
    VisceralFatIndex: Any
    SubcutaneousFatIndex: Any
    FatFreeIndex: Any


class XComposition:
    device: torch.device
    verbose: bool
    _model_type: Literal['early', 'intermediate', 'late']
    _metric_type: Literal['Volumetric', 'L3']
    model: nn.Module
    imaging_model: nn.Module | None
    clinical_model: nn.Module | None
    clinical_scaler: StandardScaler
    transform: Any
    target_scaler: StandardScaler
    measures: list[str]

    def __init__(self, model_type: Literal['early', 'intermediate', 'late'] = "intermediate", metric_type: Literal['Volumetric', 'L3'] = 'Volumetric', image_size = 512, verbose=True):
        super(XComposition, self).__init__()
        self.verbose = verbose
        self._model_type = model_type
        self._metric_type = metric_type
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        if verbose:
            print(f"Using device {device}.")

        if metric_type == "l3":
            self.measures = [
                'SkeletalMuscleArea',
                'SkeletalMuscleIndex',
                'VisceralFatArea',
                'SubcutaneousFatArea',
                'VertebralBoneArea',
                'VertebralBoneDensity',
                'MuscleRadiodensity',
                'FatFreeArea',
                'IntramuscularFat',
                'LiverFatContent',
                'AorticCalcification',
                'AorticCalcificationNumberOfPlaques',
                'VisceralFatIndex',
                'SubcutaneousFatIndex',
                'FatFreeIndex'
            ]
        else: 
            # volumetric
            self.measures = [
                'SkeletalMuscleVolume',
                'SkeletalMuscleIndex',
                'VisceralFatVolume',
                'SubcutaneousFatVolume',
                'VertebralBoneVolume',
                'VertebralBoneDensity',
                'MuscleRadiodensity',
                'FatFreeVolume',
                'IntramuscularFat',
                'LiverFatContent',
                'AorticCalcification',
                'AorticCalcificationNumberOfPlaques',
                'VisceralFatIndex',
                'SubcutaneousFatIndex',
                'FatFreeIndex'
            ]


        if model_type == "early":
            model = RegressionModel_EarlyFusion(image_size)
            model_file = 'early_orig_l3_test_v1_model.pth' if metric_type == 'L3' else 'early_orig_vol_test_v1_model.pth'
            with rpath(weights, model_file) as w_file:
                model.load_state_dict(torch.load(w_file, weights_only = True, map_location=self.device), strict = True)
        elif model_type == "late":
            imaging_model = RegressionModel(
                clin_dropout = 1,
                image_dropout = 0,
            )
            clinical_model = RegressionModel(
                clin_dropout = 0,
                image_dropout = 1,
            )
            model = LateFusionFC(
                len(self.measures), 
            )
            model_file = 'imaging_orig_l3_test_v1_model_final.pth' if metric_type == 'L3' else 'imaging_orig_vol_test_final_v3_model_final.pth'
            with rpath(weights, model_file) as w_file:
                imaging_model.load_state_dict(torch.load(w_file, weights_only = True, map_location=self.device), strict = True)
            model_file = 'clinical_orig_l3_test_v1_model_final.pth' if metric_type == 'L3' else 'clinical_orig_vol_test_v1_model_final.pth'
            with rpath(weights, model_file) as w_file:
                clinical_model.load_state_dict(torch.load(w_file, weights_only = True, map_location=self.device), strict = True)
            model_file = 'orig_multimodal_late_L3_model.pth' if metric_type == 'L3' else 'orig_multimodal_late_Vol_repeat_v2_model.pth'
            with rpath(weights, model_file) as w_file:
                model.load_state_dict(torch.load(w_file, weights_only = True, map_location=self.device), strict = True)
                
        elif model_type == "intermediate":
            model = RegressionModel(
                concatenate_modes = True,
            )
            model_file = 'multimodal_orig_l3_test_model_final.pth' if metric_type == 'L3' else 'multimodal_orig_vol_test_v1_1_model_final.pth'
            with rpath(weights, model_file) as w_file:
                model.load_state_dict(torch.load(w_file, weights_only = True, map_location=self.device), strict = True)
        else:
            raise ("model_type not identified. Select between 'early', 'intermediate', and 'late'.")
        
        self.model = model.to(device)
        self.model.eval()
        if model_type == 'late':
            self.imaging_model = imaging_model.to(device)
            self.imaging_model.eval()
            self.clinical_model = clinical_model.to(device)
            self.clinical_model.eval()
        

        self.clinical_scaler = StandardScaler()
        self.clinical_scaler.mean_ = np.array([67.45283019,  1.68292302, 77.7541812 ])
        self.clinical_scaler.var_ = np.array([2.98696633e+02, 3.54263147e-02, 4.25082758e+02])
        self.clinical_scaler.scale_ = np.sqrt(self.clinical_scaler.var_)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(image_size,image_size)),  # Resize the image to 256x256 pixels
            transforms.ToTensor(),  # Convert the image to a tensor
        ])

        self.target_scaler = StandardScaler()


        if metric_type == "l3":
            self.target_scaler.mean_ = np.array([
                125.14760495,  44.9400916 , 166.70394145, 239.5644466 ,
                18.86134172, 324.81253669,  19.02094337, 373.33098797,
                12.56724757,  20.036201  ,   9.03266657,  11.86930693,
                59.24750018,  87.89628178, 134.69129128
            ])
            self.target_scaler.var_ = np.array([
                1573.67583443,   250.95421763, 12596.27203542, 22403.15371029,
                24.22489071,  6990.95587807,   255.18254782, 11178.16943945,
                103.84622446,  4519.08389641,   208.84252241,   148.49381041,
                1553.51702834,  3562.33198932,  1889.18985391
            ])
        else:
            self.target_scaler.mean_ = np.array([
                2247.96707009,  801.19966078, 2843.76614039, 4518.7092236 ,
                345.54986121,  318.74833423,   19.21742501, 8051.35807585,
                222.51841185,   19.83477645,    8.99465132,   10.68520357,
                1002.66414096, 1647.32820014, 2878.88538925
            ])
            self.target_scaler.var_ = np.array([
                6.54055499e+05, 8.85446783e+04, 3.80591687e+06, 8.50303872e+06,
                6.19214262e+03, 6.04838072e+03, 2.24863933e+02, 5.56113196e+06,
                3.14102921e+04, 4.46123001e+03, 2.08878270e+02, 1.20851251e+02,
                4.41020868e+05, 1.27913024e+06, 7.54298075e+05
            ])
        
        self.target_scaler.scale_ = np.sqrt(self.target_scaler.var_)

    
    def _prepare_inputs(self, image, sex, age, height, weight):
        clinical_variables = (sex, age, height, weight)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        inputs = self.transform(image)
        clinical_vars = np.array([clinical_variables[0]] + list(self.clinical_scaler.transform([clinical_variables[1:4]])[0])).reshape(1,-1)
        if len(inputs.shape)==3:
            inputs = inputs.unsqueeze(0)
            clinical_vars = torch.tensor(clinical_vars, dtype = torch.float)
        return inputs, clinical_vars
    
    def _eval(self, clinical_vars, inputs):
        clinical_vars = clinical_vars.to(self.device)
        inputs = inputs.to(self.device)

        with torch.no_grad():  
            if self._model_type == 'late':
                imaging_outputs = self.imaging_model(inputs, clinical_vars)
                clinical_outputs = self.clinical_model(inputs, clinical_vars)
                inputs_concat = torch.cat([imaging_outputs, clinical_outputs], dim = 1)
                outputs = self.model(inputs_concat)
            else:
                outputs = self.model(inputs, clinical_vars)
        return self.target_scaler.inverse_transform(outputs.cpu()).squeeze()
        

    def __call__(self, image, sex: int, age: float, height: float, weight: float) -> BodyComposition | VolumetricBodyComposition:
        """
        computes the body composition metric for the given x-ray image and clinical variables
        :param image pixel_array x-ray image
        :param sex the sex of the patient 1=Male, 0=Female
        :param age the age of the patient in years
        :param height the height of the patient in meter
        :param weight the weight of the patient in kilograms
        :return the extracted body composition metrics
        """
        # Encode 'Sex' as a binary variable (1 for Male, 0 for Female)
        
        inputs, clinical_vars = self._prepare_inputs(image, sex, age, height, weight)
        
        outputs = self._eval(clinical_vars, inputs)

        return {bc: outputs[i] for i, bc in enumerate(self.measures)}   

    def explain(self, image, sex: int, age: float, height: float, weight: float) -> tuple[BodyComposition, BodyCompositionAttribution] | tuple[VolumetricBodyComposition, VolumetricBodyCompositionAttribution]:
        """
        computes the body composition metric for the given x-ray image and clinical variables and explains it
        :param image pixel_array x-ray image
        :param sex the sex of the patient 1=Male, 0=Female
        :param age the age of the patient in years
        :param height the height of the patient in meter
        :param weight the weight of the patient in kilograms
        :return a tuple of the extracted body composition metrics along with the attribution
        """
        from captum.attr import Occlusion
        inputs, clinical_vars = self._prepare_inputs(image, sex, age, height, weight)
        
        outputs = self._eval(clinical_vars, inputs)

        body_comp = {bc: outputs[i] for i, bc in enumerate(self.measures)} 
        
        # generate explainability
        occlusion = Occlusion(self.model)
        sliding_window_shapes = (1, 30, 30)  
        strides = (1, 20, 20)  
        baseline_image = torch.mean(inputs) * torch.ones_like(inputs)
        baseline_image = torch.zeros_like(inputs)
        
        baseline_clinical = torch.zeros_like(clinical_vars)
        attributions_oc: BodyCompositionAttribution = {}
        with torch.no_grad():
            for i, bc in enumerate(self.measures):
                attributions_oc[bc] = occlusion.attribute((inputs, clinical_vars), 
                                                target=i,  
                                                strides=(strides, 1), 
                                                sliding_window_shapes=(sliding_window_shapes, (1,)), 
                                                baselines=(baseline_image, baseline_clinical),
                                                show_progress=self.verbose)
        return body_comp, attributions_oc
