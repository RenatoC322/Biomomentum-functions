![](LOGO.png)
<p align="right">© Biomomentum Inc - All rights reserved</p>

# Biomomentum-functions
This repository contains multiple python scripts used for data analysis in mechanical testing with mach-1 testers from Biomomentum. To use
this library, it recommended to use conda environments.
## Installation
1. Install [Anaconda](https://www.anaconda.com/download/success) on your computer.
2. Open *Anaconda Powershell Prompt* by searching in the taskbar.
3. Create a new environment using the following command on an *Anaconda Powershell Prompt* (use python 3.11 for most stable release): 
> conda create --name env python=3.11
3. From the /dist directory of this repository, download the WHL file
4. Activate the newly created environment:
> conda activate env
5. Run the following command:
> pip install --upgrade biomomentum
6. Import the library into a Python script or notebook:
> Import biomomentum

## Testing
A jupyter notebook is provided for testing some functions. Follow these steps to try out the notebook on your computer.
1. Open *Anaconda Powershell Prompt* by searching in the taskbar.
2. Activate the newly created environment:
> conda activate env
3. Install **jupyter package** in the created environment using
> conda install jupyter
4. Install **matplotlib package** in the created environment using
> conda install matplotlib
5. Download and install [VsCode](https://code.visualstudio.com/download)
6. On the left tab of the software go to extensions and download Python and Jupyter.
7. Download the folder \examples to test the library.

## Documentation
The following section presents each function incorporated into the *biomomentum* Python library.

### Utils
#### sorted_alphanumeric
```python
def sorted_alphanumeric(files) -> list
```
Sorts alpha numerically files from directory.

Arguments:
- `files` *list* - Files from directory to sort.
  
Returns:
- *list* - Files from directory to sort.

#### get_super
```python
def get_super(x) -> str
```
Insert super script char on string

Arguments:
- `x` *str* - Character to super script.
  
Returns:
- *str* - Char in super script format.

#### inDict
```python
def inDict(dict, key) -> Bool
```
Checks if key is in dictionary.

Arguments:
- `dict` *Dict* - Dictionary to check keys.
- `key` *str* - Key to check if is in dictionary.
  
Returns:
- *Bool* - True if key is in dictionary, false otherwise.

#### select_data_file
```python
def select_data_file(file_extension = ".txt") -> str
```
Function that brings a pop up prompt to select the mach-1 file or MAP file.

Arguments:
- `file_extension` *str* - File extention from file either .txt (Mach-1 files) or .map (MAP files).
  
Returns:
- `filename` *str* - Directory of the selected file.

#### select_data_file_dir
```python
def select_data_file_dir(keyword = None, read_MAP = False) -> List
```
Function that brings a pop up prompt to select a folder contaning multiple mach-1 files or MAP files.

Arguments:
- `keyword` *str* - String Name of group of mach-1 files to load in folder.
- `read_MAP` *Bool* - Bool to indicate whether to select mach_1 files or MAP files.  
Returns:
- `files` *list* - Files of the mach-1 from the folder or MAP files.

### stats
#### ResidualStandardError
```python
def ResidualStandardError(x, xfit, predictors) -> float
```
Insert super script char on string

Arguments:
- `x` *np.array* - Independent variable values (assumed to be error-free).
- `xfit` *np.array* - Signal fit of xfit.
- `predictor` *int* - Number of predictors of the function.
  
Returns:
- `ser` *float* - Standard residual error.

#### rsquared
```python
def rsquared(Y, mse, poly_order) -> float
```
Extracts statistical R-squared.

Arguments:
- `Y` *np.array* - Signal Fitted .
- `mse` *np.array* - Mean Squared Error of the fit.
- `poly_order` *int* - Number of predictors of the function.
  
Returns:
- `Rsq_adj` *int* -  Adjusted R-squared.

### interpolation
#### linear_least_square
```python
def linear_least_square(x,y)
```
Least square algorithm.

Arguments:
- `x` *np.array* - independent variable (data units).
- `y` *np.array* - dependent variable (data units).
  
Returns:
- `A` *np.array* - Parameters of linear model (A[0] slope, A[1] intercept).
- `curveFit` *np.array* - Linear fit.
- `Rsq_adj` *float* - Adjusted R-squared.

#### interpolateMAP
```python
def interpolateMAP(subSurfaces, interpolate_to_bounds = False, smooth_data = False, threshold = 4, keyword = "")
```
Function to apply 2D linear interpolation into the data.

Arguments:
- `subSurfaces` *dict* - Dictionary of all the surfaces identified in the MAP file.
- `threshold` *float* - threshold standard deviation to control smoothing.
- `interpolate_to_bounds` *bool* - Flag to indicate whether to extrapolate values to surface bound.
- `threshold` *float* - threshold standard deviation to control smoothing.
- `keyword` *str* - Name given to the measurements in the MAP file.
  
Returns:
- `QP_2D` *list of np.array* -   List of 2D array of the interpolated values into the subSurface.
- `triangles` *list of scipy.Delaunay* - List of Triangles used for the interpolation (see [Delaunay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html#scipy.spatial.Delaunay)).
- `grid_X` *list of np.array* - List of 2D array of the X values used to construct the interpolation.
- `grid_Y` *list of np.array* - List of 2D array of the Y values used to construct the interpolation.

#### smoothMAP
```python
def smoothMAP(QP, triangles, threshold)
```
Function to smooth data for interpolation.

Arguments:
- `QP` *np.array* - independent variable (data units).
- `triangles` *list[list]* - list of lists, each sublist contains the indices of neighbors for each data point.
- `threshold` *np.array* - threshold standard deviation to control smoothing.
  
Returns:
- `smoothed_map` *np.array* - Smoothed data.
  
### ios
#### read_mach_1_file
```python
def read_mach_1_file(file_path: Union[str, Path], read_data: bool = True, data_headers: Optional[List[str]] = None, show_progress: bool = True) -> Dict[str, Any]
```
Parse a Mach-1 Motion .txt file into a nested dictionary structure.

Arguments:
- `file_path` *str* - .txt file name generated by Mach-1 Motion .
- `read_data` *int* - Integer indicating if the numerical data in the Mach-1 .txt file should be read. Default value is 1. 0: Read only the infos.
- `data_headers` *list* - List of specific data headers to load.
- `show_progress` *bool* - Flag to show progress bar on command prompt.
  
Returns:
- `result` *dict* - A mapping from function instance keys to a dictionary containing:
  - `<INFO>`: core info key-values1
  - `<FunctionName>`: metadata for each function
  - `<DATA>`: parsed measurement data (numpy arrays)
  
#### read_mach_1_files
```python
def read_mach_1_files(sources: Union[str, Path, List[Tuple[Path, str]]], read_data: bool = True, data_headers: Optional[List[str]] = None, pattern: str = "*.txt", show_progress: bool = True, use_multiprocessing: bool = False) -> Tuple[Dict[str, Any], Path]
```
Read Mach-1 .txt files into a dictionary using the refactored reader.

Arguments:
- `read_data` *int* - Integer indicating if the numerical data in the Mach-1 .txt file should be read. Default value is 1. 0: Read only the infos.
- `data_headers` *List* - List of specific data headers to load.
- `show_progress` *Bool* - Flag to show progress bar on command prompt.
- `use_multiprocessing` *Bool* - Flag to indicate whether to use multiprocessing to load mach-1 files.
  
Returns:
- *Tuple*:
  - Dict mapping each key to its parsed content.
  - Path of the processed directory (parent of first file).

#### get_subSurfaces
```python
def get_subSurfaces(MAP_file, keyword = "", scanFlag = False) -> dict
```
Function to separate surfaces from MAP in dictionary.

Arguments:
- `MAP_file` *str* - Map file directory.
- `keyword` *str* - Name given to the measurements in the MAP file.
- `scanFlag` *bool* - Indicate whether MAP file contains scan data
  
Returns:
- `subSurfaces` *dict* - Dictionary of all the surfaces identified in the MAP file.

### analysis_static
#### compliance_correction
```python
def compliance_correction(displacement, load, criteria = 1, interval = None) -> float
```
Extracts compliance correction factor for compression tests in mach-1.

Arguments:
- `displacement` *np.array* - Displacement of sample in mm.
- `load` *np.array* - Load of sample in N.
- `criteria` *float* - Criteria to select data.
- `interval` *list* - Interval to extract correction factor.
  
Returns:
- `correction_factor` *float* - Correction for compliance.

#### HayesElasticModel
```python
def HayesElasticModel(posZ, loadZ, gf_flag, maxStrain, R, v, Rsq_req, sampleThickness = None, origin_set=False, eqModulus=False, spherical_indenter=False)
```
Extracts Elastic Properties from Indentation.

Arguments:
- `posZ` *np.array* - Array Z-position (mm).
- `loadZ` *np.array* - Array Z-load (gf or N).
- `gf_flag` *Bool* - Indicate whether loadZ units is gf.
- `maxStrain` *float* - Radius of indenter in mm.
- `R` *flaot* - Poisson's ratio.
- `v` *float* - Interval to extract correction factor.
- `Rsq_req` *float* - Required fit R square value (usually 0.95).
- `sampleThickness` *float* - Sample thickness in mm.
- `origin_set` *bool* - Indicate whether signal starts at origin.
- `eqModulus` *bool* - Indicate whether signal only fits 2 points.
- `spherical_indenter` *bool* - Indicate wheter to use cylinder Hayes model or spherical model.
  
Returns:
- `G` *float* - Indentation Shear Modulus in MPa.
- `E` *float* - Indentation Elastic Modulus in MPa.
- `Fit` *np.array* - Fit for posZ and loadZ using Hayes spherical model 1972.
- `Rsq_adj` *float* - Adjusted R-squared for the fit.

#### relaxation_constant
```python
def relaxation_constant(data, time, startIdx) -> float
```
Extracts Relaxation Constant from time data.

Arguments:
- `data` *np.array* - Data to extract tau.
- `time` *np.array* - Time (s).
- `startIdx` *int* - Index to indicate where to start the analysis.
  
Returns:
- `tau` *float* - Time constant.

#### stressrelaxation_fit
```python
def stressrelaxation_fit(t, sz, r0, e0, vm, t0in = None, filemat = None
```
Function to fit stress relaxation curves and extract mechanical properties.

Arguments:
- `t` *np.array* - Data array (s).
- `sz` *np.array* - Stress array (MPa).
- `r0` *Bool* - Disk radius (mm).
- `e0` *float* - Disk stress.
- `vm` *flaot* - Poisson's ratio.
- `t0in` *float* - Compression time (s).
- `filemat` *float* - Matlab file (sveff_tab.mat).
  
Returns:
- `szfit` *np.array* - Stress array fit (MPa).
- `ef` *float* - Elastic Fibril Modulus (MPa).
- `k0` *float* - Hydraulic permeability (mm2/MPa.s).
- `e33` *float* - Elastic Equilibrium Modulus (MPa).
- `t0` *float* - Compression time (s).
- `S11` *float* - Elastic Modulus S11 (MPa).
- `szequ` *float* - Equilibrium stress (MPa).
- `K` *float* - Coefficient (MPa.s).
- `tau` *float* - Time constant (s).
- `em` *float* - Elastic Modulus of isotropic matrix (MPa).
- `nm` *float* - Poisson coefficient of isotropic matrix.
- `mse` *float* - Mean quadratic error .

#### get_cartilage_index
```python
def get_cartilage_index(Fz, Time) -> np.array
```
Function to extract cartilage index from signal

Arguments:
- `Fz` *np.array* - Normal force from automatic thickness indentation.
- `Time` *np.array* - Time (s).
  
Returns:
- `index_interest_mean` *np.array* - Begin and end of cartilage layer of surface.
  
### analysis_dynamic
#### FitSinusoid
```python
def FitSinusoid(time, freq, data_1, data_2 = None, method = "trf", lossType = "soft_l1", fscale = 0.001, trim_data = False)
```
Sinusoid fit on data using least-squares and extract dynamic modulus.

Arguments:
- `time` *np.array* - time vector (s).
- `freq` *float* - frequency guess (Hz).
- `data_1` *np.array* - data vector (data units) .
- `data_2` *np.array* - data vector (data units) to sync with data_1 (None by default).
- `method` *np.array* - method for the least-squares ([least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)).
- `lossType` *np.array* - loss for the method applied ([least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)).
- `fscale` *np.array* - weight for the loss function ([least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)).
- `trim_data` *bool* - bool to indicate whether to trim the data or not.
  
Returns:
- `Params` *structure* - If data_2 is not given then.
  - `Params[0]` *float* - Sin offset of data_1.
  - `Params[1]` *float* - Sin amplitude of data_1.
  - `Params[2]` *float* - Sin frequency of data_1.
  - `Params[3]` *float* - Sin phase of data_1.
- `Params` *structure* - If data_2 is given then.
  - `Params[2]` *float* - Sin frequency of data_1 and data_2.
  - `Params[4]` *float* - Sin offset of data_2.
  - `Params[5]` *float* - Sin amplitude of data_2.
  - `Params[6]` *float* - Sin phase of data_2.
- `data_sin ` *np.array* - Sin fit into data_1.
- `data_sin_2 ` *np.array* - Sin fit into data_2 (if passed in arguments).
- `data_extra` *np.array* - Data trimmed if flag passed (for plotting) for data_1.
- `data_extra_2` *np.array* - Data trimmed if flag passed (for plotting) for data_2 (if passed in arguments).
- `ser_data` *float* - Residual standard error of the fit for data_1.
- `ser_data_2` *float* - Residual standard error of the fit for data_2 (if passed in arguments).

