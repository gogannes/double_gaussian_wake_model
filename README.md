# Double Gaussian Wake Model

## Project
This project is the implementation of the Double Gaussian Wake model for wind turbines published in the article:

Schreiber, J., Balbaa, A., and Bottasso, C. L.: Brief communication: A double-Gaussian wake model, Wind Energ. Sci., 5,
237–244, https://doi.org/10.5194/wes-5-237-2020, 2020.

## About the wake model
The analytical wind turbine wake model assumes a double-Gaussian velocity distribution.
The choice of a double-Gaussian shape function is motivated by the behavior of the near-wake region that is observed in
numerical simulations and experimental measurements.
The method is based on the conservation of momentum principle, while stream-tube theory is used to determine the wake
expansion at the tube outlet.

The model parameters used in this repository have been identified for a scaled wind turbine model (TUM-G1). For 
different wind turbines, it is recommended to re-identify parameters, as they depend on turbine characteristics.

## How to install the wake model
This repository can be installed by downloading the source code (not via the PyPI package manager).

Download the source code:
```
git clone https://github.com/gogannes/double_gaussian_wake_model
```
And install the requirements:
```
pip install -r requirements.txt
 ```

## How to use the wake model
See `demo.py`. 

## License
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree
