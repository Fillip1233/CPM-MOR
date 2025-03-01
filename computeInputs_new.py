import operator
import numpy as np
import math


def computeInputs_new(VsourcesAll=None, IsourcesAll=None, t=None, func=None, *args, **kwargs):
    # compute vector of current source and voltage source
    # values at t t.  All sources are sinusoids.
    rowVs = VsourcesAll.shape[0]
    rowIs = IsourcesAll.shape[0]
    uv = np.zeros((rowVs, 1))
    ui = np.zeros((rowIs, 1))
    # uv = []
    # ui = []
    # uv = np.array(uv)
    # ui = np.array(ui)
    if operator.eq(func, 'sin'):
        if len(t) == 1:
            # sourcesAll = [node1, node2, freq, phase, offset, amp]
            if not (np.all(VsourcesAll == 0)) == 1:
                # # uv = voffset + vamp.*sin(2*pi*vfreq*0+vphi);
                uv = VsourcesAll[:, 4] + VsourcesAll[:, 5]*np.sin(
                    2 * np.pi*VsourcesAll[:, 2]*t + VsourcesAll[:, 3])
            if not (np.all(IsourcesAll == 0)) == 1:
                #   # ui = ioffset + iamp.*sin(2*pi*ifreq*0+iphi);
                ui = IsourcesAll[:, 4] + IsourcesAll[:, 5]*np.sin(
                    2 * np.pi* IsourcesAll[:, 2]*t + IsourcesAll[:, 3])
        else:
            t = t[:]
            t = t.T
            NV = np.size(VsourcesAll, 0)
            NI = np.size(IsourcesAll, 0)
            NT = len(t)
            if not (np.all(VsourcesAll == 0)) == 1:
                voff = np.tile(VsourcesAll[:4], (1, NT))
                vpha = np.tile(VsourcesAll[:, 3], (1, NT))
                vamp = np.tile(VsourcesAll[:, 5], (1, NT))
                vfrq = np.tile(VsourcesAll[:, 2], (1, NT))
                tv = np.tile(t, (NV, 1))
                uv = voff + np.multiply(vamp, np.sin(np.multiply(np.dot(np.dot(2, np.pi), vfrq), tv) + vpha))
            if not (np.all(IsourcesAll == 0)) == 1:
                ti = np.tile(t, (NI, 1))
                ioff = np.tile(IsourcesAll[:, 4], (1, NT))
                ipha = np.tile(IsourcesAll[:, 3], (1, NT))
                iamp = np.tile(IsourcesAll[:, 5], (1, NT))
                ifrq = np.tile(IsourcesAll[:, 2], (1, NT))
                ui = ioff + iamp*np.sin(2*np.pi*ifrq*ti) + ipha

    else:
        if operator.eq(func, 'pulse'):
            #     # PULSE(V1 V2 TD TR TF PW PER)
            #     t = t - per * int(t / per)
            #         if t < td:
            #             return v1
            #         elif t < td + tr:
            #             return v1 + ((v2 - v1) / (tr)) * (t - td)
            #         elif t < td + tr + pw:
            #             return v2
            #         elif t < td + tr + pw + tf:
            #             return v2 + ((v1 - v2) / (tf)) * (t - (td + tr + pw))
            #         else:
            #             return v1
            if not (np.all(VsourcesAll == 0)) == 1:
                nv = VsourcesAll.shape[0]
                uv = np.zeros((nv, 1))
                for i in range(0, nv):
                    vs = VsourcesAll[i, :]
                    v1, v2, td, tr, tf, pw, per = vs[0], vs[1], vs[2], vs[3], vs[4], vs[5], vs[6]
                    t = t - np.multiply(per, math.floor(t / per))
                    if t < td:
                        uv[i] = v1
                    else:
                        if t < td + tr:
                            uv[i] = v1 + np.dot(((v2 - v1) / tr), (t - td))
                        else:
                            if t < td + tr + pw:
                                uv[i] = v2
                            else:
                                if t < td + tr + pw + tf:
                                    uv[i] = v2 + np.dot(((v1 - v2) / tf), (t - (td + tr + pw)))
                                else:
                                    uv[i] = v1
            if not (np.all(IsourcesAll == 0)) == 1:
                ni = IsourcesAll.shape[0]
                ui = np.zeros((ni, 1))
                for i in range(0, ni):
                    is_ = IsourcesAll[i, :]
                    v1, v2, td, tr, tf, pw, per = is_[0], is_[1], is_[2], is_[3], is_[4], is_[5], is_[6]
                    t = t - np.multiply(per, math.floor(t / per)) #将t限制在一个周期内/zhenjie
                    if t < td:
                        ui[i] = v1
                    else:
                        if t < td + tr:
                            ui[i] = v1 + np.dot(((v2 - v1) / (tr)), (t - td))
                        else:
                            if t < td + tr + pw:
                                ui[i] = v2
                            else:
                                if t < td + tr + pw + tf:
                                    ui[i] = v2 + np.dot(((v1 - v2) / (tf)), (t - (td + tr + pw)))
                                else:
                                    ui[i] = v1
    return ui, uv
    # uv = sparse(uv); ui = sparse(ui);
