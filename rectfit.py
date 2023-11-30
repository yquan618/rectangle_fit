"""
This is the official implementation of the paper: [Optimised Least Squares Approach for Accurate Polygon and Ellipse Fitting](https://arxiv.org/abs/2307.06528)
If you use this code or any part of it in your research, please cite our paper:
Y. Quan, S. Chen, Optimised Least Squares Approach for Accurate Polygon and Ellipse Fitting. arXiv:2307.06528 [cs.CV], 2023. https://arxiv.org/abs/2307.06528
"""

import numpy as np
from numpy import cos, sin, tan, pi, mat, zeros, array
from numpy.linalg import norm
from matplotlib import pyplot as plt

p1, p2, p3, p4 = 15.625, 5.24609375, 8.3998, 0.2786875

def f_step(x):
    out = zeros(x.shape)
    out[x >= 0] = 1.0
    return out

def est_phi(theta, mxmy):
    theta = np.mod(theta, 2 * pi)
    num = len(theta)
    reshape_flag = False
    if num > 1:
        theta = theta.reshape((1, -1))
        reshape_flag = True
    h = mxmy[1]**2 / mxmy[0]**2
    t = np.tan(theta)**2
    k = h/(h+t)
    q1 = p1 * (k - 0.5) ** 3 - p2 * (k - 0.5)
    q2 = p3 * np.sqrt((k ** 2 - k) * (k - k**2 - p4))
    v = -2.5 * k + 1.75 - (np.cbrt(q1 + q2) + np.cbrt(q1 - q2))
    phi = np.sign(tan(theta)) * np.arccos(np.sqrt(v)) + pi * f_step(-cos(theta)) * np.sign(sin(theta))
    if reshape_flag:
        phi = phi.reshape((-1, 1))
    return phi

def get_pxy(phi):
    t0 = 17 + 0.11 / (cos(4 * phi) + 1.08) - 0.22 / (cos(8 * phi) + 2)
    t1 = 4 * cos(2 * phi)
    px = cos(phi) * (t0 - t1) / 26
    py = sin(phi) * (t0 + t1) / 26
    return px, py

def get_L_pxy(phi, v, data):
    [xc, yc, mx, my, alpha] = v[:,0]
    cosa = cos(alpha)
    sina = sin(alpha)
    Q = np.array([[cosa, sina], [-sina, cosa]])  # Q = R^-1
    M = np.diag((mx, my))
    X_c = mat([xc, yc]).T
    px, py = get_pxy(phi)
    pxy = mat(np.hstack((px, py))).T
    L = (Q*(data.T - X_c) - M*pxy).T.reshape(1, -1).T
    return L, px, py

def GaussNewton(get_L_pxy, get_J, data, v0, iter=20, epsilon=1E-6):
    V = v0
    num = len(data)
    Vs = zeros((5, iter))
    for iloop in range(iter):
        [xc, yc, mx, my, alpha] = V[:5, 0]
        theta = array(np.arctan2((data[:, 1] - yc), (data[:, 0] - xc)) - alpha)
        phi = est_phi(theta, [mx, my])
        L, px, py = get_L_pxy(phi, V, data)
        sse = np.sum(np.square(L))
        A = get_J(V, data, phi, px, py)
        dV = -(A.T * A).I * A.T * L
        if norm(dV) / (norm(V)) < epsilon:
            break
        Vs[:, iloop] = V.T
        V += dV
        V = np.array(V)
        rmse = np.sqrt(sse / num)
        print("iter = %d,dV norm= " % iloop, norm(dV), ', rmse=', rmse)
    if iloop < iter:
        Vs = Vs[:, :iloop]
    sse = np.sum(np.square(L))
    return V, Vs, sse

def get_J(V, data, phi, px, py):
    [xc, yc, mx, my, alpha] = V[:, 0]
    num = len(phi)
    dxy = data - mat([xc, yc])
    x = data[:, 0]
    y = data[:, 1]
    theta = np.array(np.arctan2((y - yc), (x - xc)) - alpha)
    h = my**2 / mx**2
    t = tan(theta)**2
    k = h/(h+t)
    q1 = p1 * (k - 0.5) ** 3 - p2 * (k - 0.5)
    q2 = p3 * np.sqrt((k ** 2 - k) * (k - k ** 2 - p4))
    v = -2.5 * k + 1.75 - (np.cbrt(q1 + q2) + np.cbrt(q1 - q2))
    v[v == 1.0] = 1 - 1E-12 #avoid inf number in calculation of dphi_k
    v[v == 0.0] = 1E-12 
    kk = (k - 1) * k
    kk[kk == 0] = -1E-12 #avoid inf number in calculation of dq2_dk
    dq1_dk = 3 * p1 * (k - 0.5) ** 2 - p2
    dq2_dk = -p3 * (2 * k - 1) * (p4 + 2 * kk) / (2 * np.sqrt(-kk * (p4 + kk)))
    q1p2_m2_3 = 1 / np.cbrt((q1 + q2) ** 2)
    q1m2_m2_3 = 1 / np.cbrt((q1 - q2) ** 2)
    dphi_k = np.sign(tan(theta)) * (7.5 + q1p2_m2_3 * (dq1_dk + dq2_dk) + q1m2_m2_3 * (dq1_dk - dq2_dk)) / (
                6 * np.sqrt(v * (1 - v)))
    dk_theta = -2 * h * tan(theta) * (t + 1) / (h + t) ** 2
    dphi_theta = dphi_k * dk_theta
    q3 = 2 * h * t / (t + h) ** 2
    dk_mx = -q3 / mx
    dk_my = q3 / my
    dphi_mx = dphi_k * dk_mx
    dphi_my = dphi_k * dk_my
    dp_alpha = -dphi_theta
    dp_theta = -dp_alpha
    norm2 = norm(dxy, axis=1) ** 2
    norm2 = norm2[np.newaxis, :]
    dtheta_xc = dxy[:, 1] / norm2.T
    dtheta_yc = -dxy[:, 0] / norm2.T
    cosa, sina = cos(alpha), sin(alpha)
    cosp, sinp = cos(phi), sin(phi)
    cos2p, cos4p = cos(2*phi), cos(4*phi)
    sin2p, sin4p = sin(2*phi), sin(4*phi)
    w = cos4p+1.08
    mx_dpx_phi = mx * sinp*(2.4*cos2p-1.8)/5.2 + 0.022*(4*cosp*sin4p-sinp*w)/w**2/5.2
    my_dpy_phi = my * cosp*(2.4*cos2p+1.8)/5.2 + 0.022*(4*sinp*sin4p+cosp*w)/w**2/5.2
    AB12 = np.zeros((num*2, 2))
    AB12[0::2, 0] = -cosa - (mx_dpx_phi * dp_theta * np.array(dtheta_xc)).T
    AB12[1::2, 0] =  sina - (my_dpy_phi * dp_theta * np.array(dtheta_xc)).T
    AB12[0::2, 1] = -sina - (mx_dpx_phi * dp_theta * np.array(dtheta_yc)).T
    AB12[1::2, 1] = -cosa - (my_dpy_phi * dp_theta * np.array(dtheta_yc)).T
    AB34 = np.zeros((num*2, 2))
    AB34[0::2, 0] = -(px + mx_dpx_phi * dphi_mx).T
    AB34[1::2, 0] = -(my_dpy_phi * dphi_mx).T
    AB34[0::2, 1] = -(mx_dpx_phi * dphi_my).T
    AB34[1::2, 1] = -(py + my_dpy_phi * dphi_my).T
    AB56 = (np.array([[-sina, cosa], [-cosa, -sina]])*(dxy.T)).T.reshape(1,-1).T
    AB56[0::2, 0] -= mx_dpx_phi * dp_alpha
    AB56[1::2, 0] -= my_dpy_phi * dp_alpha
    return np.hstack((AB12, AB34, AB56))

def gen_rect(V, phi_range=np.arange(0, 360, 1), noise=False):
    [xc, yc, mx, my, alpha] = V[:, 0]
    T = mat([xc, yc]).T
    cosa = cos(alpha)
    sina = sin(alpha)
    Q = np.array([[cosa, -sina], [sina, cosa]])
    phi = np.deg2rad(phi_range)
    px, py = get_pxy(phi)
    xy = mat(np.vstack((mx * px, my * py)))
    XY = T + Q * xy
    if noise:
        XY += np.random.randn(XY.shape[0], XY.shape[1])
    return XY.T

def rectfit(data, iter=10):
    num = len(data)
    data = mat(data)
    xy_ = data.mean(axis=0)
    data_c = data - xy_
    rmsd = norm(data_c)/np.sqrt(num)
    data_cn = data_c/rmsd
    data = data_cn
    V = np.zeros((5, 1))
    [xc, yc] = [0.0, 0.0]
    [mx, my] = [1, 1]
    _, _, Vh = np.linalg.svd(data)
    alpha = np.mod(np.arctan2(Vh[0,1], Vh[0,0]), pi)
    V[:5, 0] = [xc, yc, mx, my, alpha]
    V, Vs, sse = GaussNewton(get_L_pxy, get_J, data, V, iter=iter, epsilon=1E-6)
    rmse = np.sqrt(sse / num)
    print('rmse=', rmse)
    V[:2] = V[:2]*rmsd + xy_.T
    V[2:4] *= rmsd
    Vs[:2, :] = Vs[:2, :]*rmsd + xy_.T
    Vs[2:4, :] *= rmsd
    rmse *= rmsd
    return V, Vs, rmse

if __name__ == "__main__":
    data = array([[9.1641, 7.5196],
                   [7.9641, 9.5981],
                   [6.8094, 8.9314],
                   [3.3453, 6.9314],
                   [1.3692, 5.0207],
                   [2.3692, 3.2887],
                   [3.3692, 1.5566],
                   [8.0774, 2.7353],
                   [9.8094, 3.7353],
                   [10.0641, 5.9608]])
    estQ, vsQ, rmseQ = rectfit(data, iter=10)
    print('parameters (xc, yc, mx, my, alpha) of fit rectangle is', estQ.T[0])
    plt.figure()
    Q_fit = gen_rect(estQ, phi_range=np.arange(0, 360, 1))
    plt.plot(Q_fit[:, 0], Q_fit[:, 1], 'k-.', label='rectangle fit')
    plt.plot(data[:, 0], data[:, 1], 'r+', label='Points', markersize=8)
    plt.gca().set_aspect(1)
    plt.legend()
    font_img = {'weight': 'normal', 'size': 12}
    plt.xlabel('${X}$', font_img)
    plt.ylabel('${Y}$', font_img)
    plt.show()
