# Biomomentum-functions
This repository contains multiple python scripts used for data analysis in mechanical testing with mach-1 testers from Biomomentum. To use
this library, it recommended to use conda environments.
## Installation
1. Create a new environments using the following command on an Anaconda command prompt (use python 3.11 for most stable release): 
> conda create --name env python=3.11
2. From the /dist directory, download the WHL file
3. Activate the newly created environment:
> conda activate env
4. Copy the directory of the WHL file and run the following command:
> pip install /directory/.../.whl
5. Import the library into a Python script or notebook:
> Import biomomentum

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
get_super(x) -> str
```
Insert super script char on string

Arguments:
- `x` *str* - Character to super script.
  
Returns:
- *str* - Char in super script format.

#### ResidualStandardError
```python
ResidualStandardError(x, xfit, predictors) -> float
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
rsquared(Y, mse, poly_order) -> float
```
Extracts statistical R-squared.

Arguments:
- `Y` *np.array* - Signal Fitted .
- `mse` *np.array* - Mean Squared Error of the fit.
- `poly_order` *int* - Number of predictors of the function.
  
Returns:
- `Rsq_adj` *int* -  Adjusted R-squared.

#### linear_least_square
```python
linear_least_square(x,y)
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
interpolateMAP(subSurfaces, interpolate_to_bounds = False, smooth_data = False, threshold = 4, keyword = "")
```
Function to apply 2D linear interpolation into the data.

Arguments:
- `subSurfaces` *dict* - Dictionary of all the surfaces identified in the MAP file.
- `threshold` *float* - threshold standard deviation to control smoothing.
- `interpolate_to_bounds` *bool* - Flag to indicate whether to extrapolate values to surface bound.
- `threshold` *float* - threshold standard deviation to control smoothing.
- `keyword` *str* - Name given to the measurements in the MAP file.
  
Returns:
- `QP_2D` *np.array* - 2D array of the interpolated values into the subSurface.
- `triangles` *scipy.Delaunay* - Triangles used for the interpolation (see [Delaunay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html#scipy.spatial.Delaunay)).
- `grid_X` *np.array* - 2D array of the X values used to construct the interpolation.
- `grid_Y` *np.array* - 2D array of the Y values used to construct the interpolation.

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
  
### Structures
#### compute_stats

```python
def compute_stats(x: np.ndarray, y: np.ndarray) -> dict
```
Add function description.

Arguments:

- `x` *np.array* - Data series 1.
- `y` *np.array* - Data series 2.
  
Returns:
- `float` - The calculated RMS value

### Static_analysis
#### compute_stats

```python
def compute_stats(x: np.ndarray, y: np.ndarray) -> dict
```
Add function description.

Arguments:

- `x` *np.array* - Data series 1.
- `y` *np.array* - Data series 2.
  
Returns:
- `float` - The calculated RMS value

### Dynamic_analysis
#### compute_stats

```python
def compute_stats(x: np.ndarray, y: np.ndarray) -> dict
```
Add function description.

Arguments:

- `x` *np.array* - Data series 1.
- `y` *np.array* - Data series 2.
  
Returns:
- `float` - The calculated RMS value


