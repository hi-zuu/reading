### 2023 FUXI

FuXi: a cascade machine learning forecasting system for 15-day global weather forecast (**换成引用**)

#### problems of weather prediction?

- Uncertainty in weather forecasts is inevitable due to the limited resolution, approximation of physical processes in parameterizations, errors in initial conditions (and boundary conditions for regional models), and the chaotic nature of the atmosphere. Additionally, the degree of uncertainty and the magnitude of errors in weather forecasts increases as forecast lead time.
- To conclude, significant progress have been achieved in ML based weather forecasting, particularly in 10-day forecasts where the ML models have outperformed ECMWF HRES. However, further breakthroughs are necessary to address the issues related to iterative accumulated errors and enhance the accuracy of forecasts for longer lead times.

#### category

 A **cascaded ML weather forecasting *system*** that provides **15-day global forecasts** at **a temporal resolution of 6 hours** and **a spatial resolution of 0.25°**.

#### method

- autoregressive
- cube embedding, U-Transformer(**to be supplemented**), and a fully connected (FC) layer
- pre-train, fine-tune： 预训练时输入 t-1、t 时刻的天气情况，预测 t+1 时刻天气情况形成基础模型。微调出三个模型：FuXi-Short、FuXi-Medium 以及 FuXi-Long，三个模型分别预测第 0 到 5 天 (0-20 time steps)、第 5 到 10 天、第 10 到 15 天的天气情况。三者之间，FuXi-Medium 使用 FuXi-Short 的最后两步输出作为输入，FuXi-Long 与 FuXi-Medium 的关系也是这样。
- ensemble forecasting: 50-member ensemble (1+ 49).incorporate **random Perlin noise** to perturb the initial conditions. **Monte Carlo dropout** to perturb the model parameters

#### data

- **introduction**: A subset of ECMWF ERA5 reanalysis dataset covers 39 years and includes 70 variables, which comprise 5 upper-air atmospheric variables (geopotential, temperature, u component of wind, v component of wind , and relative humidity) at 13 pressure levels(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, and 1000hPa) and 5 surface variables(T2M, 10-meter u wind component , 10-meter v wind component , MSL, and Total Precipitation).  It has a spatial resolution of 0.25° (721 × 1440 latitude-longitude grid points) and a temporal resolution of 6 hours.
- **spliting method**:  Following previous studies in splitting the data into training, validation, and testing set12,17, the training set consists of 54020 (54020 = 365×4×37, similarly, 2920 = 365×4×2, and 1460 = 365×4) samples spanning from 1979 to 2015. The validation set contains 2920 samples corresponding to the years 2016 and 2017, while out-of-sample testing is performed using 1460 samples from 2018.
- **input dimensions**: 2×70×721×1440, where 2,70,721, and 1440 represent the two preceding time steps (t−1 and t), the total number of input variables, latitude (H) and longitude (W) grid points, respectively.

#### loss function

The loss function used is the latitude-weighted L1 loss, which is defined as follows:

![image-20251011193355117](.\picture\image-20251011193355117.png)

where C, H, and W are the number of channels and the number of grid points in latitude and longitude direction, respectively. c, i, and j are the indices for variables, latitude and longitude coordinates, respectively. ^ Xtþ1 c; i; j and Xtþ1 c; i; j are predicted and ground truth for some variable and locations (latitude and longitude coordinates) at time step of t+1. ai represents the weight at latitude i and the value of ai decreases as latitude increases. The L1 loss is averaged over all the grid points and variables.

#### metrics(*to be supplemented*)

two metrics to assess  forecast performance:

![image-20251011183204665](.\picture\image-20251011183204665.png)

two metrics to evaluate the quality of ensemble forecasts:

![image-20251011183233517](.\picture\image-20251011183233517.png)

![image-20251011183300020](.\picture\image-20251011183300020.png)  

#### other info

#### ![image-20251011193545605](.\picture\image-20251011193545605.png)

#### FUXI need to do（solve?）

- ML models have outperformed HRES  in 10-day forecasts with a spatial resolution of 0.25. ML models still accumulate forecast errors for longer effective forecasts which can not be solved by implementing autoregressive multi-time step loss (this is a single model).
- The next significant goals are to achieve comparable performance to ECMWF ensemble, of which the ensemble mean (EM) often has greater skill than the deterministic forecasts for longer lead times, and to increase the forecast lead time beyond 10 days.

#### I need to do

- read ***Rasp, S. et al. Weatherbench: a benchmark data set for data-driven weather forecasting.***

- read ***Weyn, J. A., Durran, D. R. & Caruana, R. Improving data-driven global weather prediction using deep convolutional neural networks on a cubed sphere.*** (multi-time-step loss function)

- what is cube embedding

#### thought

从时间序列上看，假设天气数据集有 1、2、3、4、5 这五天。训练时会出现样本（x = 1； y = 2，3）以及（x = 2；y = 3，4）。那么对于不同的输入 x = 1 和 x = 2，输出会有部分相同（即 y = 3 会在两种输入对应的输出中各出现一次）。这样在深度学习中是否合理？



### 2025 FUXI

A data-to-forecast machine learning system for global weather (**换成引用**)

#### category

a weather forecast system that integrate data assimilation (DA) and machine learning.

#### method

#### data

#### metrics

#### I need to do

what is DA？what  is background in DA？



### FUXI-DA(论文作图以及评估可以参考这篇文章)

#### category

a  DL based DA framework.

This framework is designed for the **direct assimilation of satellite observations** to produce the **analysis** that **optimize the forecast performance of DL-based weather forecasting models**.

#### what is DA ？what is background ?

The initial fields is commonly referred to as the ‘analysis’. **DA system generate the analysis** which is considered the most accurate estimate of the current atmospheric state, obtained by combining **short-term forecasts, referred to as the ‘background’**, from NWP models with observational data.

DA **establishes intricate relationships** between the **background field** and the vast amount of multi-source **observation data** within **limited operational time windows**.(a formula to demonstrate? **to be supplement**)

DA can provide initial conditions. Both numerical weather prediction (NWP) systems and the recently developed deep learning-based (DL-based) weather forecasting models rely on accurate initial conditions.

#### current DA‘s problem

 One primary challenge is that **the large volume of observations is challenging to fully exploit** in traditional DA systems.

*reason*:

- thinning process
- a significant portion of satellite observations are assimilated under clear-sky conditions, neglecting valuable cloud and precipitation data.

Another challenge lies in the **high computational cost** of widely adopted DA methods, such as four-dimensional variational (4D-Var), ensemble Kalman filter (EnKF), and ensemble-variational (En-Var) methods. It needs time to generate the analysis. So, NWP centers initiate assimilation and forecasting operations early to ensure meteorological support. Consequently, operational products are unable to assimilate the observations collected during the period from the start of product creation to delivery22.

#### data

**label**: ERA5 reanalysis data includes variables same as 2023 FUXI. (70channels)

**background**:  6-h forecast fields generated by the DL-based weather forecasting model FuXi. (70channels)

**observation data**: Fengyun-4B satellite. 

![image-20251019150446209](.\picture\image-20251019150446209.png)

Additionally, considering that satellite observation biases usually vary with scan angle and geographic location52,54,55, the longitude, latitude, satellite zenith angle, and observation time of each observation are encoded as additional channels. 

Finally, the observation input consists of a total of 15 channels, including 8 brightness tem perature channels and 7 spatiotemporal encoding channels. 

#### method

![image-20251016184946630](.\picture\image-20251016184946630.png)



**model**: 

![image-20251019153809243](.\picture\image-20251019153809243.png) 

why crop?  风云4B只能观测到这么大的范围

**Loss function**:

![image-20251019154942084](.\picture\image-20251019154942084.png)



#### evaluation

**Evaluation method**:

![image-20251019142617085](.\picture\image-20251019142617085.png)



**assimilation performance**: To evaluate assimilation performance, three experimental configurations were designed: EXP_ASSI, EXP_CORR, and EXP_CTRL.

![image-20251016175949398](.\picture\image-20251016175949398.png)



**consistency with prior knowledge of atmospheric physics**: The first run involved using a 6-h forecast and the original observations as inputs. For the second run, we introduced a perturbation at a specific point within the original observa tions. By analyzing the differences in the outputs from these two runs, we assessed the analysis increment attributable to the perturbation.



**** FUXI-DA plan to do

To further enhance FuXi-DA’s feature extraction and integration capabilities, we plan to implement state-of-the-art training strategies, such as contrastive learning techniques to align disparate features50, and explore advanced architectures like Transformers to achieve feature fusion51. Additionally, it is essential to develop an algorithm within DL-based assimilation systems to detect and correct sudden observational biases, such as those resulting from satellite instrument degradation or contamination during extreme events52, thereby mitigating their impact on the assimilation process.

