# Project Explanation — Multi-Modal Solar Power Forecasting

This document explains what each notebook does, every term used inside it, and why that term matters. Written in plain language so anyone — technical or not — can understand the project end to end.

---

## What This Project Does (Big Picture)

We have a **500 kW solar power plant** in Pune, India. Every hour we want to predict how much power it will produce in the **next hour**. To do this we use three different types of data at the same time:

1. **The last 24 hours of power output** (history tells us what's happening)
2. **Current weather numbers** — temperature, humidity, wind, cloud cover, sun angle
3. **A satellite image of the sky above the plant** — clouds block sunlight, so the image tells us what's coming

We combine all three using a deep learning model, evaluate it against traditional methods, and also build an alarm system that raises a flag when something is wrong with the plant.

---

## Notebook 1 — `01_weather_eda.ipynb`
### Weather Data Analysis

**Purpose:** Understand the raw weather data before feeding it to any model. Find patterns, check quality, and confirm that the variables behave as expected across seasons.

---

### Terms in this notebook

**ERA5**
ERA5 is a global weather dataset produced by ECMWF (a European weather centre). It is not measurements from a physical station — it is a computer model of the atmosphere run backwards over historical time to estimate what the weather was at any location. We use it because it gives consistent hourly data for every location on Earth, for free. Our data covers Pune (18.6°N, 73.8°E) for the full year 2023.

**GHI — Global Horizontal Irradiance (W/m²)**
The total solar energy falling on a flat horizontal surface per second, measured in Watts per square metre. This is the most direct driver of power output — more sunlight = more electricity. GHI is zero at night and peaks around 900–1000 W/m² at clear-sky noon.

**DNI — Direct Normal Irradiance**
The part of sunlight coming straight from the sun's disc (not scattered). Important for concentrating solar systems. Less relevant for flat-panel PV but included in the data.

**DHI — Diffuse Horizontal Irradiance**
Sunlight that has been scattered by the atmosphere and arrives from all directions rather than directly from the sun. On overcast days most of the irradiance is DHI.

**Cloud Cover Fraction**
A number between 0 and 1. 0 = completely clear sky. 1 = fully overcast. This is the single most important weather variable for solar forecasting because clouds block GHI directly. The satellite CNN branch is specifically designed to capture this signal visually.

**Zenith Angle (zenith_deg)**
The angle between the sun and the point directly overhead. 0° = sun is directly overhead (maximum irradiance). 90° = sun is on the horizon. Beyond 90° = night. Power output drops quickly above 70°. We use zenith angle as a feature so the model knows what time of day it is from a solar geometry perspective.

**Temperature (temp_air, °C)**
Air temperature affects solar panel efficiency. PV panels become less efficient as they get hotter — approximately −0.4% per °C above 25°C. So a hot summer day actually produces slightly less power than expected from irradiance alone.

**Relative Humidity (%)**
The amount of moisture in the air. High humidity increases atmospheric scattering which slightly reduces GHI. It also correlates with cloud formation. Less important than cloud cover but still a useful feature.

**Wind Speed (m/s)**
Wind cools the panels, which improves efficiency slightly. Also used in the satellite branch as a proxy for how fast clouds are moving across the sky.

**Pressure (pressure_pa) and Precipitation (precip_mm)**
Barometric pressure and rainfall. Used in correlation analysis. Rainfall is a strong indicator of cloud cover and is occasionally used for atmospheric correction.

**Monthly Boxplot**
A chart that shows the distribution of a variable for each month. The box shows where the middle 50% of values fall, the line inside is the median, and the whiskers show the range. Helps us see that monsoon months (Jun–Sep) have different weather than winter (Dec–Feb).

**Season Colours**
We use four meteorological seasons specific to India:
- **Winter** (Dec–Feb) — clear skies, low humidity, good solar resource
- **Pre-Monsoon** (Mar–May) — hot, hazy, increasing cloud
- **Monsoon** (Jun–Sep) — heavy cloud, rain, lowest solar output
- **Post-Monsoon** (Oct–Nov) — clouds clearing, transitioning back to winter

**Pearson Correlation (r)**
A number between −1 and +1 that measures how linearly related two variables are. r = +1 means they move together perfectly. r = −1 means when one goes up the other goes down perfectly. r ≈ 0 means no linear relationship. We use it to confirm, for example, that cloud cover and GHI are strongly negatively correlated (r ≈ −0.8).

**Diurnal Profile**
How a variable changes over the 24 hours of a typical day. GHI has a bell-curve diurnal profile — zero at night, rising after dawn, peaking at noon, falling to zero at dusk. Plotting this by season shows how the solar day gets longer in summer and shorter in winter.

**Daytime Hours Filter (`ghi > 10`)**
We filter to hours where GHI > 10 W/m² to exclude night-time rows. At night all irradiance variables are zero and including them would distort the correlation analysis.

---

## Notebook 2 — `02_power_eda.ipynb`
### Power Output Analysis

**Purpose:** Understand the power generation data — its distribution, seasonal patterns, and where the anomalies occur. This is the variable we are trying to predict.

---

### Terms in this notebook

**SCADA (Supervisory Control and Data Acquisition)**
The software system installed at every solar plant that records sensor readings every minute — including inverter output, string currents, temperatures, and irradiance. We aggregate its 1-minute readings to hourly averages to get one row per hour. This is our target variable: `power_kw`.

**500 kW Nameplate Capacity**
The maximum rated output of the plant. Our model clips predictions to never exceed 500 kW. In practice the plant rarely exceeds ~480 kW due to inverter losses and soiling.

**power_kw**
The actual AC power output from the plant in kilowatts, recorded by SCADA and averaged to hourly resolution. This is what we are predicting.

**Inverter**
The device that converts DC electricity (produced by solar panels) to AC electricity (used by the grid). When an inverter trips (fails), its output drops to zero even though the sun is shining.

**Anomaly Types**
The dataset contains three types of plant fault events injected at labelled timestamps:
- **`inverter_trip`** — one or more inverters shut down completely. Power drops to near zero. Easy to detect because the drop is large (100+ kW).
- **`string_fault`** — a string of panels goes offline (disconnected cable, failed bypass diode, shading). Power drops by 15–35%. Moderate drop, moderately detectable.
- **`mppt_underperf`** — the Maximum Power Point Tracker (the control algorithm inside the inverter) is operating sub-optimally, causing 5–15% lower output than expected. Very small drop, hardest to detect.

**`is_anomaly` flag**
A boolean column (True/False) marking whether an hour contains a fault event. Used as ground truth for evaluating the alert module in Notebook 7.

**Power Distribution Histogram**
A bar chart showing how often each power level occurs. We plot three versions:
- All 8760 hours — a large spike at 0 kW (night-time) plus a spread from 0–500 kW
- Daytime only — removes the night spike, shows realistic generation distribution
- Monthly mean — shows that monsoon months have the lowest average output

**Diurnal Power Profile**
Mean power at each hour of the day, by season. Shows the characteristic bell curve of solar generation peaking around noon. Monsoon months have a flatter, lower bell due to cloud cover. Winter months have a sharper peak.

**One-Week Anomaly Visualisation (Dec 4–10)**
A time-series plot of one week in the test set with anomaly events highlighted using different markers (X = inverter trip, square = string fault, triangle = MPPT). This shows visually how each fault type manifests as a different magnitude of deviation from normal output.

**Cloud Cover vs Power Scatter**
A scatter plot where each point is one daytime hour. X-axis = cloud cover, Y-axis = power output. Shows a clear negative correlation — the more cloud, the less power. This motivates including the satellite image as an input to the model.

**Zenith Angle vs Power Scatter**
Shows how power drops as the sun gets lower (higher zenith angle). This confirms that solar geometry is the dominant control on power output during clear-sky periods.

**70/15/15 Chronological Split**
We divide the data in time order (not randomly) to simulate real-world deployment:
- **Train: Jan–Aug (70%)** — the model learns from this data
- **Val: Sep–Oct (15%)** — used to tune the model and stop training early if it starts memorising
- **Test: Nov–Dec (15%)** — only used at the very end to report final performance

We must split chronologically (not randomly) because randomly mixing hours would let the model see "future" data during training, giving fake good results.

---

## Notebook 3 — `03_satellite_imagery.ipynb`
### Satellite Imagery Analysis

**Purpose:** Explore the INSAT-3D satellite patches, validate that they encode cloud information correctly, visualise them by season, and export them as individual image files.

---

### Terms in this notebook

**INSAT-3D**
India's geostationary weather satellite operated by ISRO. It captures images of India and the Indian Ocean every 15–30 minutes. We use the **visible channel** — essentially a black-and-white photograph of cloud tops as seen from space. We take a 64×64 pixel crop centred over the Pune plant site.

**Satellite Patch (64×64×3)**
A small image crop: 64 pixels wide, 64 pixels tall, 3 colour channels (Red, Green, Blue). Each hourly row has exactly one patch. The pixel values encode reflectance — bright white = thick cloud tops, dark = clear sky or land surface.

**`satellite_patches.npz`**
A compressed NumPy archive (`.npz` = NumPy zip) storing all 8760 patches as a single array of shape `(8760, 64, 64, 3)` with uint8 values (0–255). Loaded with `np.load()`.

**Key `'images'`**
The name we use when saving/loading the patch array inside the `.npz` file. `npz['images']` retrieves the full array. (Note: not `'patches'` — a common mistake.)

**Brightness**
The mean pixel value across all 64×64×3 values of a patch, divided by 255 to get a [0, 1] range. Brighter patches = more cloud or more sun reflectance. We use brightness as a sanity check: it should correlate positively with cloud cover fraction.

**Brightness Validation**
We compute Pearson r between patch brightness and ERA5 cloud cover (daytime hours). A strong positive correlation (r > 0.6) confirms the images are physically consistent with the weather data. We also expect brightness to correlate negatively with GHI (more cloud = brighter image but less irradiance).

**RGB Channels**
The three colour planes of each image — Red, Green, Blue. The visible channel from INSAT-3D is actually monochrome (greyscale), but we store it as a 3-channel image to match the CNN input format. We plot per-channel histograms by season to confirm consistency.

**4×5 Sample Grid (Seasons × Cloud Cover Bins)**
A grid of example patches arranged as 4 rows (seasons) × 5 columns (cloud cover bands: 0–0.2, 0.2–0.4, etc.). This is the best way to visually confirm that the patches look different in different weather conditions and across seasons.

**PNG Export (`data/satellite_images/YYYY/MM/`)**
We save every patch as an individual `.png` file organised into year/month subfolders. This creates a proper image dataset directory that can be loaded by standard image libraries (PIL, OpenCV, torchvision). The filename format is `YYYYmmddTHHMMSS.png` (ISO 8601 timestamp).

**PIL (Pillow)**
Python Imaging Library — the standard Python package for reading and writing image files. We use `Image.fromarray(patch).save(path)` to write each patch as a PNG.

---

## Notebook 4 — `04_dataset_pipeline.ipynb`
### Dataset Pipeline Validation

**Purpose:** Confirm that the PyTorch dataset correctly joins the three data sources, produces the right tensor shapes, and applies normalisation properly before any model training.

---

### Terms in this notebook

**PyTorch**
The deep learning framework used to build and train the model. It represents data as `Tensor` objects (multi-dimensional arrays that can run on GPU) and uses automatic differentiation to compute gradients for training.

**`build_datasets(csv_path, npz_path)`**
The main function in `dataset.py` that loads both data files, computes normalisation statistics, splits into train/val/test, and returns three `SolarMultiModalDataset` objects plus the normalisation stats and the full DataFrame.

**`SolarMultiModalDataset`**
A PyTorch `Dataset` subclass. It pre-computes all normalised arrays once at load time and returns one sample per `__getitem__` call. Each sample contains 5 items: `x_ts, x_met, x_img, y_norm, y_raw`.

**`DataLoader`**
A PyTorch utility that wraps a Dataset and serves data in batches, optionally shuffling and using multiple worker processes for parallel loading. We use `batch_size=32` for training.

**Batch**
One group of samples processed together. A batch of 32 means 32 hourly samples are stacked into tensors and passed through the model at once. Training in batches is much faster than one sample at a time.

**x_ts — Time Series Input, shape (batch, 24, 6)**
The 24-hour sliding window of past power and weather data fed to the LSTM encoder. For each of the 24 hours before the prediction time, we have 6 values: `power_kw, ghi, temp_air, humidity, hour_sin, hour_cos`. All z-score normalised.

**WINDOW = 24**
We always look back exactly 24 hours. This captures one full day of history, which is the most informative window for solar forecasting because solar patterns repeat daily.

**HORIZON = 1**
We predict exactly 1 hour into the future. A horizon of 1 is called "one-step-ahead" forecasting.

**x_met — Meteorological Input, shape (batch, 5)**
Current-hour weather scalars: `temp_air, humidity, wind_speed, cloud_cover, zenith_deg`. These are the "right now" weather conditions at prediction time. Z-score normalised.

**x_img — Image Input, shape (batch, 3, 64, 64)**
The satellite patch at the current hour. Shape is channel-first (3 channels first, then height, then width) as required by PyTorch CNN layers. Values scaled to [0, 1] float.

**y_norm — Normalised Target**
The power output at `t+1` (the hour we are predicting), converted to z-score units using the training set mean and standard deviation. The model predicts this normalised value; we convert back to kW for reporting.

**y_raw — Raw Target (kW)**
The actual power output at `t+1` in kilowatts. Used to compute RMSE and MAE in physical units.

**LSTM_FEATURES**
`['power_kw', 'ghi', 'temp_air', 'humidity']` — the four columns used in the LSTM sliding window. Hour information (sin/cos) is appended separately.

**MET_FEATURES**
`['temp_air', 'humidity', 'wind_speed', 'cloud_cover', 'zenith_deg']` — the five weather scalars for the Met encoder.

**hour_sin, hour_cos — Cyclical Hour Encoding**
The hour of day (0–23) cannot be fed directly as a number because hour 0 and hour 23 are adjacent in real time but far apart numerically. Instead we encode them as `sin(2π × hour/24)` and `cos(2π × hour/24)`. This places hours on a circle so the model sees midnight correctly as being "close" to 1 AM.

**Z-Score Normalisation**
We subtract the training set mean and divide by the training set standard deviation for every feature. This puts all features onto a similar scale (roughly −3 to +3) which helps the model learn faster. We fit the statistics **only on training rows** — applying training statistics to val and test rows ensures no information leakage.

**`norm_stats.json`**
A JSON file storing the mean and standard deviation for every feature, computed once from the training set. It is loaded at inference time to denormalise predictions back to kW.

**LSTM Window Heatmap**
A visual of one 24h×6 input window as a colour grid. Rows = the 6 features, columns = the 24 hours. Red = high z-score, blue = low z-score. Useful for checking that the window looks sensible (e.g., power should be zero at night hours).

---

## Notebook 5 — `05_model_training.ipynb`
### Model Training & Inference Results

**Purpose:** Show training dynamics (how the model improved over epochs), present the final test set predictions (scatter plot), and analyse the attention weights to understand which input modality the model relies on most.

---

### Terms in this notebook

**Multi-Modal Model**
A model that accepts multiple different types of input data simultaneously. Our model has three separate input streams (time series, weather scalars, satellite image) that are encoded independently and then merged. "Multi-modal" means "multiple types of data".

**LSTM Encoder — shape output R^128**
LSTM stands for Long Short-Term Memory. It is a type of recurrent neural network designed to learn patterns in sequences. We feed it the 24-hour window and it produces a 128-dimensional summary vector representing "what has been happening with power and weather over the past day". It has 2 layers and hidden size 128.

**How LSTM works (simple)**
Think of it as reading the 24 hours one by one. At each hour it updates its internal memory — it decides what to remember, what to forget, and what to output. After reading all 24 hours, the final memory state is our 128-number summary.

**Met (FC) Encoder — shape output R^64**
A simple feed-forward encoder for the 5 weather scalars. FC = Fully Connected. It uses two linear layers: `5 → 128 → 64`, with Batch Normalisation and ReLU activation in between. Produces a 64-number summary of current weather conditions.

**CNN Encoder — shape output R^128**
CNN = Convolutional Neural Network. Designed for images. It uses three convolutional blocks, each extracting increasingly abstract features from the satellite image. Filter counts: 32 → 64 → 128. Ends with Global Average Pooling (GAP) to produce a 128-number summary of the satellite image.

**Convolutional Block (ConvBlock)**
Each block contains: Conv2d (3×3 filter, slides over the image) → BatchNorm2d → ReLU → AvgPool2d (2×2, halves the image size). After 3 blocks, the 64×64 image becomes 8×8 before GAP flattens it.

**Global Average Pooling (GAP)**
Takes the spatial average of each feature map, converting a tensor of shape (batch, 128, 8, 8) to (batch, 128, 1, 1) then to (batch, 128). This makes the CNN output a fixed-size vector regardless of input image size.

**Early Fusion (ConcatFusion)**
Simply concatenate the three encoder outputs: [z_ts (128) ; z_met (64) ; z_img (128)] = z_fused of size 320. "Early" because fusion happens right after encoding. Simple and effective.

**Attention Fusion (AttnFusion)**
Instead of simple concatenation, each encoder output is projected to 128 dimensions, then a learned weight (alpha) is computed via softmax over all three. The final vector is the weighted sum. This allows the model to dynamically decide "how much should I trust the time series vs the satellite image for this specific hour?"

**Softmax**
A function that takes a vector of numbers and converts them to probabilities that sum to 1. Used for attention weights so α_ts + α_met + α_img = 1.0 always.

**Attention Weights (alpha: ts=0.783, met=0.070, img=0.148)**
The learned importance the model assigns to each modality on average over the test set. ts=0.783 means the LSTM time series is the most informative (78.3% weight). img=0.148 means satellite imagery contributes 14.8%. met=0.070 means weather scalars contribute only 7%. This matches physical intuition: recent power history is the strongest predictor.

**Regression Head**
The final prediction layers that take the fused vector and output a single number (predicted power). Architecture: `FC(d→256) → ReLU → Dropout(0.2) → FC(256→128) → ReLU → Dropout(0.2) → Linear(1)`. "Regression" because we are predicting a continuous number, not a category.

**Dropout (p=0.2)**
Randomly sets 20% of neuron outputs to zero during training. This forces the network to not rely too heavily on any single neuron, which reduces overfitting (memorising the training data instead of learning general patterns).

**ReLU (Rectified Linear Unit)**
The most common activation function in deep learning. ReLU(x) = max(0, x). It introduces non-linearity so the network can learn complex patterns. Without it, stacking linear layers would still only produce a linear model.

**Batch Normalisation (BN)**
Normalises the output of a layer across the batch during training. Stabilises training, allows higher learning rates, and acts as mild regularisation. Applied after each FC layer in the Met encoder.

**419,905 Parameters**
The total number of learnable numbers (weights and biases) in the full model. This is relatively small for a deep learning model, which is why training takes only ~12 seconds on a GPU.

**Training Loop (train.py)**
The code that runs forward pass (prediction) → loss computation → backward pass (gradient computation) → weight update, repeated for each batch of each epoch.

**MSE Loss (Mean Squared Error)**
The training objective. For each prediction we compute `(predicted − actual)²`, then average over the batch. Squaring penalises large errors more than small ones, pushing the model to avoid big mistakes.

**Adam Optimiser (lr=0.001, weight_decay=1e-4)**
Adam is an adaptive learning rate optimiser — it adjusts the step size for each parameter individually based on its gradient history. lr = learning rate = how big a step to take per update. weight_decay = L2 regularisation = gentle penalty on large weights to prevent overfitting.

**Cosine Annealing (LR Schedule)**
The learning rate starts at 0.001 and gradually reduces following a cosine curve over 100 epochs, ending near 0.00001. This allows large steps early (explore quickly) and small steps later (fine-tune carefully).

**Linear Warmup (5 epochs)**
For the first 5 epochs the learning rate ramps up from 0.0001 to 0.001 linearly. This prevents the model from making large, destabilising updates on the very first batches when weights are random.

**Early Stopping (patience=12)**
If the validation RMSE does not improve for 12 consecutive epochs, training stops and the best checkpoint is restored. This prevents overfitting — after a point the model starts memorising training data and the validation error increases.

**Gradient Clipping (norm=1.0)**
Before each weight update, if the gradient vector has magnitude > 1.0 it is rescaled to exactly 1.0. This prevents "exploding gradients" — a training instability where gradients become extremely large and cause wild weight updates.

**CNN Freeze (first 10 epochs)**
The CNN encoder's first two convolutional blocks are frozen (weights not updated) for the first 10 epochs. This gives the LSTM and Met encoders time to reach reasonable values before the CNN parameters start changing too.

**Epoch**
One complete pass through the entire training dataset. We train for up to 100 epochs with early stopping.

**Checkpoint (`best.pt`)**
A saved snapshot of the model weights at the epoch with the lowest validation RMSE. PyTorch saves this as a `.pt` (PyTorch) file. At evaluation time we load this checkpoint instead of the final epoch weights.

**Predicted vs Actual Scatter (Fig 5)**
Each point represents one test-set hour. X = actual power, Y = predicted power. Points on the diagonal line (y=x) are perfect predictions. The colour shows absolute error — green = small error, red = large error. A good model has points tightly clustered around the diagonal with R²=0.968.

**R² (R-squared, Coefficient of Determination)**
Measures how much of the variance in power output the model explains. R²=1.0 means perfect predictions. R²=0 means the model is no better than always predicting the mean. R²=0.968 means the model explains 96.8% of the variance — excellent.

**RMSE (Root Mean Square Error, kW)**
The square root of the average squared prediction error. RMSE = 5.89 kW means on average predictions are off by about 5.89 kW. Lower is better. Expressed in the same units as power (kW).

**MAE (Mean Absolute Error, kW)**
The average of the absolute errors. MAE = 4.63 kW means predictions are off by 4.63 kW on average. Less sensitive to large outliers than RMSE. MAE ≤ RMSE always.

**nRMSE (Normalised RMSE, %)**
RMSE divided by the plant's nameplate capacity (500 kW), expressed as a percentage. nRMSE = 1.18% makes it easy to compare across plants of different sizes. A value below 2% is considered good.

---

## Notebook 6 — `06_baselines_comparison.ipynb`
### Baseline Models Comparison — Fig 2 & Fig 3

**Purpose:** Compare all models — traditional baselines, single-modality deep learning, and the multi-modal proposed model — to show that adding more modalities and more sophisticated fusion improves forecasting accuracy.

---

### Terms in this notebook

**Baseline Model**
A simpler model used as a reference point. If our complex model cannot beat a simple baseline, it is not worth using. Baselines here are ARIMA and SVR — well-established methods that do not use deep learning.

**ARIMA(2,1,2)**
ARIMA = AutoRegressive Integrated Moving Average. A classical statistical time series model.
- **AR(2)** — uses the last 2 values of the series to predict the next one (autoregressive)
- **I(1)** — differences the series once to make it stationary (removes trends)
- **MA(2)** — uses the last 2 prediction errors to correct the current forecast (moving average)
It only uses the power history, no weather or satellite data. RMSE = 21.40 kW — worst among all models.

**Stationarity**
A time series is stationary if its mean and variance do not change over time. ARIMA requires stationarity. The "I(1)" differencing (subtracting consecutive values) removes the daily trend and makes the series stationary.

**SVR — Support Vector Regression (RBF kernel)**
A machine learning model that finds the best-fit function by minimising prediction errors within a tolerance band (ε-insensitive zone). The RBF (Radial Basis Function) kernel maps the features into a higher-dimensional space to capture non-linear relationships. C=100 is the regularisation parameter — higher C means the model tries harder to fit every training point.
Uses lag features (past power values) + current weather scalars. RMSE = 16.80 kW — better than ARIMA but worse than all deep learning models.

**Lag Features**
For SVR we manually create features: `power_kw` at t−1, t−2, t−24, t−48. These give the model information about recent and yesterday's power, mimicking what the LSTM learns automatically.

**StandardScaler**
Sklearn's z-score normaliser. Subtracts mean and divides by std, same as our z-score normalisation. Fitted on training data only and applied to val/test.

**Late-Fusion Ensemble**
Instead of feeding all inputs to one model, we train three separate single-modality models (LSTM only, Met only, CNN only) and average their predictions at the end. "Late" = fusion happens after each model has made its own prediction. We use **inverse-RMSE weighting**: models with lower RMSE get higher weight in the average. RMSE = 9.12 kW — better than any single-modality model.

**Single-Modality Deep Learning**
Three variants of the model where only one input is used:
- **TS only (LSTM)** — only the 24h power window. RMSE = 11.27 kW
- **Met only (FC)** — only the 5 weather scalars. RMSE = 14.83 kW
- **Img only (CNN)** — only the satellite patch. RMSE = 13.05 kW

The ordering TS > Img > Met shows that time-series history is most informative, satellite second, and instantaneous weather scalars third — because the LSTM already implicitly captures weather through the past power signal.

**Modality Combination Ablation**
Testing all 7 possible combinations of the three inputs (TS, Met, Img, TS+Met, TS+Img, Met+Img, All three) to see which combinations help. Key findings:
- Adding any modality to TS-only always helps
- TS+Img beats TS+Met (satellite adds more information than weather scalars alone)
- All three together is best

**Fig 2 — RMSE Bar Chart**
Horizontal or vertical bars showing RMSE for every model, ordered from worst (ARIMA) to best (AttnFusion). Colour-coded by category: grey = traditional, blue = single-modal DL, green = multi-modal DL.

**Fig 3 — MAE and R² Side-by-Side**
Two bar charts: left shows MAE for all models, right shows R². Confirms the same ranking as Fig 2 from two different angles.

**Fig 6 — Ablation Bars (7 Modality Combinations)**
Shows RMSE for each of the 7 input combinations, colour-coded by how many modalities are used: 1 = blue, 2 = orange, 3 = green. Illustrates the benefit of adding more input types.

---

## Notebook 7 — `07_alert_module.ipynb`
### Anomaly Alert Module

**Purpose:** Evaluate the fault detection system that runs alongside the forecasting model. When the model's prediction diverges significantly from actual output, it should raise an alarm so engineers can investigate.

---

### Terms in this notebook

**Alert Module**
A rule-based + ML hybrid system that flags anomalous hours. It runs after the model makes a prediction, compares prediction to actual, and decides whether to raise an alarm.

**Residual**
The difference between actual and predicted power: `residual = actual − predicted`. A large residual means something unexpected happened. The alert module watches residuals.

**Threshold Detector (τ = 0.15)**
The simplest detector. Raises an alert if the normalised residual exceeds 15%:
`|actual − predicted| / 500 kW > 0.15`
This means: "if we're off by more than 75 kW (15% of 500 kW), raise a flag." Simple and interpretable. High precision (0.95) but lower recall (0.72) — it catches most big anomalies but misses subtle ones.

**Isolation Forest**
A machine learning anomaly detection algorithm. It randomly splits the feature space with decision trees. Anomalous points require fewer splits to isolate (they are "alone" in the feature space). Features used: residual, 3-hour moving average of residual, and the 5 met scalars. `n_estimators=200, contamination=0.05`. Lower precision (0.83) but higher recall (0.85) than the threshold — it catches more faults including subtle ones but also raises more false alarms.

**OR-Fusion**
The final alert decision: raise an alarm if **either** detector fires. `Alert = Threshold OR IsolationForest`. This maximises recall (catches the most faults) at the cost of some precision (more false alarms). The OR-fusion achieves the best F1 = 0.894.

**Precision**
Of all the hours we flagged as anomalous, what fraction were truly anomalous? Precision = TP / (TP + FP). High precision = few false alarms. OR-Fusion precision = 0.910.

**Recall**
Of all the truly anomalous hours, what fraction did we catch? Recall = TP / (TP + FN). High recall = few missed faults. OR-Fusion recall = 0.880.

**F1 Score**
The harmonic mean of precision and recall: `F1 = 2 × (P × R) / (P + R)`. Balances both. OR-Fusion F1 = 0.894 — the best single-number summary of detector quality.

**FPR (False Positive Rate)**
Of all the truly normal hours, what fraction did we incorrectly flag? FPR = FP / (FP + TN). Lower is better. OR-Fusion FPR = 0.089 — we falsely alarm on 8.9% of normal hours.

**TP, FP, FN, TN**
- TP (True Positive) — correctly detected fault
- FP (False Positive) — falsely raised alarm on a normal hour
- FN (False Negative) — missed a real fault
- TN (True Negative) — correctly identified a normal hour

**Per-Type Recall**
How well each anomaly type is detected by OR-Fusion:
- **inverter_trip** — 100% recall. Power drops to near-zero, residual is 200+ kW. Unmissable.
- **string_fault** — 92% recall. 15–35% drop, residual ~56 kW. Mostly caught.
- **mppt_underperf** — 75% recall. Only 5–15% drop, residual ~14 kW. Small but detectable.

**MPPT (Maximum Power Point Tracker)**
The control algorithm inside the inverter that continuously adjusts the operating voltage to extract the most power from the panels. When it performs poorly, the plant generates 5–15% less than it should. Hard to detect because the power deviation is small.

**Fig 7 — Precision/Recall/F1 Grouped Bar Chart**
Three grouped bars (one group per metric) with three colours (one per detector). Visually shows that OR-Fusion achieves the best balance across all three metrics.

**Fig 8 — Diurnal RMSE Profile**
RMSE plotted by hour of day for two models: LSTM-only (TS only) and AttnFusion. Both have low RMSE at night (power is zero, easy to predict) and peak RMSE around noon (cloud variability is highest). AttnFusion's line is consistently below LSTM-only, with the largest gap at 10–15h — the window when the satellite imagery of incoming clouds is most valuable.

**Moving Average (MA3)**
The average of the last 3 residuals. Used as an input to the Isolation Forest. A sequence of three increasingly large residuals is a stronger anomaly signal than a single spike.

---

## Summary Table — All Files

| File | Type | Purpose |
|---|---|---|
| `pune_500kw_hourly.csv` | CSV | Hourly SCADA power + weather + anomaly labels (8760 rows) |
| `weather_pune_2023.csv` | CSV | Hourly ERA5 weather variables for Pune 2023 |
| `satellite_patches.npz` | NPZ | 8760 satellite patches, shape (8760,64,64,3), key='images' |
| `satellite_images/` | Folder | Same patches as individual PNG files, organised by YYYY/MM/ |
| `norm_stats.json` | JSON | Z-score mean/std for every feature, fitted on train only |
| `checkpoints/best.pt` | PT | Saved model weights (best validation RMSE) |
| `checkpoints/history.csv` | CSV | Per-epoch train loss, val loss, val RMSE |
| `checkpoints/ablation/*.pt` | PT | Saved weights for all 8 ablation configurations |
| `checkpoints/ablation/*.metrics.json` | JSON | Test RMSE, MAE, R², nRMSE for each configuration |
| `checkpoints/ablation/results.csv` | CSV | Aggregated metrics for all 8 ablation runs |
| `checkpoints/ablation/alert_metrics.json` | JSON | Precision, Recall, F1, FPR for all 3 alert detectors |
| `checkpoints/ablation/baselines_results.json` | JSON | ARIMA, SVR, Late-Fusion metrics |
| `baselines.py` | Python | ARIMA, SVR, Late-Fusion implementation |
| `model.py` | Python | Full model architecture (encoders + fusion + head) |
| `train.py` | Python | Training loop |
| `dataset.py` | Python | PyTorch Dataset and build_datasets() |
| `alert.py` | Python | Threshold + IsolationForest + OR-fusion evaluation |
| `ablation.py` | Python | Runs all 8 training configurations automatically |
| `evaluate.py` | Python | Generates paper figures from checkpoints |
