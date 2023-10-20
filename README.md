# Least Sqaures Rectangle Fit

This is the official implementation of the paper: [Optimised Least Squares Approach for Accurate Polygon and Ellipse Fitting](https://arxiv.org/abs/2307.06528).

If you use this code or any part of it in your research, please cite our paper as follows:

Y. Quan, S. Chen, Optimised Least Squares Approach for Accurate Polygon and Ellipse Fitting. arXiv:2307.06528 [cs.CV], 2023. https://arxiv.org/abs/2307.06528

For any questions or issues, please contact yimingquan@lsu.edu.cn or open an issue on GitHub. Thank you for your interest in our work!

------------------------------------

This code is for fitting a rectangle to a set of data points using the least squares(LS) method. The rectfit function takes a matrix of data points as input and returns the parameters of the fit rectangle, the values of parameters during LS adjustment, and the root mean square error (RMSE) of the fit. 

The parameters of the fit rectangle 

xc, yc： x abd y coordinates of centre
mx, my:  edge lengths
alpha: orientation

Here is an example of using the code with some sample data:
```
import numpy as np
import matplotlib.pyplot as plt
from rectfit import rectfit

data = np.mat([[9.1641, 7.5196], [7.9641, 9.5981],
[6.8094, 8.9314], [3.3453, 6.9314],
[1.3692, 5.0207], [2.3692, 3.2887],
[3.3692, 1.5566], [8.0774, 2.7353],
[9.8094, 3.7353], [10.0641, 5.9608]])

est, vs, rmse = rectfit(data, iter=10)
print('parameters (xc, yc, mx, my, alpha) of fit rectangle is', est.T[0])
```
