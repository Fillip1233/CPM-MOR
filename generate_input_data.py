import numpy as np
import random
import math
import matplotlib.pyplot as plt

def generate_input_data(port_num, t_all, dt, data_length, seed):
    #     PULSE(VL VH TD TR TF PW PER)

    vl_range = (1.748e-5, 4.139e-5)
    vh_range = (0.036, 0.121)
    # td_range = (0, 1.2e-9)
    td_range = (0, 1e-9)
    tr_range = (1e-10, 3e-10)
    tf_range = (1e-10, 3e-10) 
    pw_range = (1e-10, 3e-10)
    # tr = 1e-10
    # tf = 1e-10
    # pw = 1e-11
    # per_range = (2e-9, 3e-9)
    per = 2e-9
    random.seed(seed)

    IS = []
    for _ in range(port_num):
        vl = random.uniform(*vl_range)
        vh = random.uniform(*vh_range)
        td = random.uniform(*td_range)
        tr = random.uniform(*tr_range)
        # tr = 1e-10
        tf = random.uniform(*tf_range)
        # tf = 1e-10
        pw = random.uniform(*pw_range)
        # pw = 1e-11
        # per = random.uniform(*per_range)
        IS.append([vl, vh, td, tr, tf, pw, per])

    ui = np.zeros((port_num, data_length))
    for i in range(0, port_num):
        t = 0
        count = 0
        while t < t_all and count < data_length:
            is_ = IS[i]
            vl, vh, td, tr, tf, pw, per = is_[0], is_[1], is_[2], is_[3], is_[4], is_[5], is_[6]
            t1 = t - np.multiply(per, math.floor(t / per))
            if t1 < td:
                ui[i, count] = vl
            else:
                if t1 < td + tr:
                    ui[i, count] = vl + np.dot(((vh - vl) / (tr)), (t1 - td))
                else:
                    if t1 < td + tr + pw:
                        ui[i, count] = vh
                    else:
                        if t1 < td + tr + pw + tf:
                            ui[i, count] = vh + np.dot(((vl - vh) / (tf)), (t1 - (td + tr + pw)))
                        else:
                            ui[i, count] = vl
            t += dt
            count += 1
    return ui

if __name__ == '__main__':
    np.random.seed(0)
    t0 = 0
    t_all = 2e-09
    dt = 1e-11
    port_num = 10
    data_length = 200
    Ui =[]
    for i in range(1000):
        print("Generating {}th input data".format(i))
        ui = generate_input_data(port_num = port_num, t_all=t_all, dt = dt, data_length = data_length,seed=i)
        Ui.append(ui)
    Ui = np.array(Ui)
    np.save('Uin_10port_1per.npy', Ui)
    # ui = generate_input_data(port_num = port_num, t_all=t_all, dt = dt,seed=1)
    # time = np.linspace(0, 2e-9, 200)
    # for i in range(port_num):    
    #     plt.plot(time, ui[i, :], 'b-o')
    #     plt.title("Waveform Visualization")
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Amplitude")
    #     plt.grid(True)
    #     plt.show()