from numpy import newaxis as na

import numpy as np
from py_wake.deficit_models import DeficitModel
from py_wake.superposition_models import SquaredSum, LinearSum, MaxSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind


class AreaOverlappingFactor():
    def __init__(self, k=.1):
        self.k = k

    def overlapping_area_factor(self, dw_ijlk, cw_ijlk, D_src_il, D_dst_ijl):
        """Calculate overlapping factor

        Parameters
        ----------
        dw_jl : array_like
            down wind distance [m]
        cw_jl : array_like
            cross wind distance [m]
        D_src_l : array_like
            Diameter of source turbines [m]
        D_dst_jl : array_like or None
            Diameter of destination turbines [m]. If None destination is assumed to be a point

        Returns
        -------
        A_ol_factor_jl : array_like
            area overlaping factor
        """
        wake_radius_ijlk = (self.k * dw_ijlk + D_src_il[:, na, :, na] / 2)
        if D_dst_ijl is None:
            return wake_radius_ijlk > cw_ijlk  # If None destination is assumed to be a point
        else:
            return self.cal_overlapping_area_factor(wake_radius_ijlk,
                                                    (D_dst_ijl[..., na] / 2),
                                                    np.abs(cw_ijlk))

    def cal_overlapping_area_factor(self, R1, R2, d):
        """ Calculate the overlapping area of two circles with radius R1 and
        R2, centers distanced d.

        called by: def overlapping_area_factor()

        The calculation formula can be found in Eq. (A1) of :
        [Ref] Feng J, Shen WZ, Solving the wind farm layout optimization
        problem using Random search algorithm, Reneable Energy 78 (2015)
        182-192
        Note that however there are typos in Equation (A1), '2' before alpha
        and beta should be 1.

        Parameters
        ----------
        R1: array:float
            Radius of the first circle [m]

        R2: array:float
            Radius of the second circle [m]

        d: array:float
            Distance between two centers [m]

        Returns
        -------
        A_ol: array:float
            Overlapping area [m^2]
            A_ol_f: array:float, Overlapping factor
        """
        # treat all input as array
        R1, R2, d = [np.asarray(a) for a in [R1, R2, d]]
        if R2.shape != R1.shape:
            R2 = np.zeros_like(R1) + R2
        A_ol_f = np.zeros_like(R1)
        p = (R1 + R2 + d) / 2.0

        # make sure R_big >= R_small
        Rmax = np.where(R1 < R2, R2, R1)
        Rmin = np.where(R1 < R2, R1, R2)

        # full wake cases
        index_fullwake = (d <= (Rmax - Rmin))
        A_ol_f[index_fullwake] = 1

        # partial wake cases
        mask = (d > (Rmax - Rmin)) & (d < (Rmin + Rmax))

        # in somecases cos_alpha or cos_beta can be larger than 1 or less than
        # -1.0, cause problem to arccos(), resulting nan values, here fix this
        # issue.
        def arccos_lim(x):
            return np.arccos(np.maximum(np.minimum(x, 1), -1))

        alpha = arccos_lim((Rmax[mask] ** 2.0 + d[mask] ** 2 - Rmin[mask] ** 2) /
                           (2.0 * Rmax[mask] * d[mask]))

        beta = arccos_lim((Rmin[mask] ** 2.0 + d[mask] ** 2 - Rmax[mask] ** 2) /
                          (2.0 * Rmin[mask] * d[mask]))

        A_triangle = np.sqrt(p[mask] * (p[mask] - Rmin[mask]) *
                             (p[mask] - Rmax[mask]) * (p[mask] - d[mask]))

        A_ol_f[mask] = (alpha * Rmax[mask] ** 2 + beta * Rmin[mask] ** 2 -
                        2.0 * A_triangle) / (R2[mask] ** 2 * np.pi)

        return A_ol_f  # A_ol_f: array:float, Overlapping factor


class NOJDeficit(DeficitModel, AreaOverlappingFactor):
    args4deficit = ['WS_ilk', 'D_src_il', 'D_dst_ijl', 'dw_ijlk', 'cw_ijlk', 'ct_ilk']
    # 字面：参数for赤字，译为deficit计算参数
    # 这些参数的计算都不在noj内完成，所以noj内定义的计算的主题，对应的参数的计算还是依赖wind_farm_modek.py和engineering_model.py

    def __init__(self, k=.1):
        AreaOverlappingFactor.__init__(self, k)

    def _calc_layout_terms(self, WS_ilk, D_src_il, D_dst_ijl, dw_ijlk, cw_ijlk, **_):  # 计算分布关系
        R_src_il = D_src_il / 2
        term_denominator_ijlk = (1 + self.k * dw_ijlk / R_src_il[:, na, :, na]) ** 2
        term_denominator_ijlk += (term_denominator_ijlk == 0)
        # term_denominator_ijlk+False = term_denominator_ijlk，不等于0的等于本身，等于0的变为1
        A_ol_factor_ijlk = self.overlapping_area_factor(dw_ijlk, cw_ijlk, D_src_il, D_dst_ijl)

        with np.warnings.catch_warnings():  # 通过警告过滤器进行控制是否发出警告消息。
            np.warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
            self.layout_factor_ijlk = WS_ilk[:, na] * (dw_ijlk > 0) * (A_ol_factor_ijlk / term_denominator_ijlk)
            #  (dw_ijlk > 0)，1*False=0，1*True=1

    def calc_deficit(self, WS_ilk, D_src_il, D_dst_ijl, dw_ijlk, cw_ijlk, ct_ilk, **_):
        if not self.deficit_initalized:  # deficit_models.py, deficit_initalized = False
            self._calc_layout_terms(WS_ilk, D_src_il, D_dst_ijl, dw_ijlk, cw_ijlk)
        ct_ilk = np.minimum(ct_ilk, 1)  # treat ct_ilk for np.sqrt()，防止出现ct大于1的情况
        # np.minimum 取对应位置上的较小值．
        # >>> np.maximum([1, 2, 3, 4, 5], 2)
        # array([2, 2, 3, 4, 5])
        term_numerator_ilk = (1 - np.sqrt(1 - ct_ilk))
        return term_numerator_ilk[:, na] * self.layout_factor_ijlk  # 计算出风速衰减值，m/s
        # (1 - np.sqrt(1 - ct_ilk))* WS_ilk[:, na] *  (A_ol_factor_ijlk / (1 + self.k * dw_ijlk / R_src_il[:, na, :, na]) ** 2)


class NOJ(PropagateDownwind):
    def __init__(self, site, windTurbines, k=.1, superpositionModel=SquaredSum(),
                 deflectionModel=None, turbulenceModel=None):
        """
        Parameters
        ----------
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        k : float, default 0.1
            wake expansion factor
        superpositionModel : SuperpositionModel, default SquaredSum
            Model defining how deficits sum up
        blockage_deficitModel : DeficitModel, default None
            Model describing the blockage(upstream) deficit
        deflectionModel : DeflectionModel, default None
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel, default None
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=NOJDeficit(k),
                                   superpositionModel=superpositionModel,
                                   deflectionModel=deflectionModel,
                                   turbulenceModel=turbulenceModel)


if __name__ == '__main__':
    from py_wake.examples.data.iea37._iea37 import IEA37Site
    from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
    import matplotlib.pyplot as plt

    # setup site, turbines and wind farm model
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wf_model_ss = NOJ(site, windTurbines, k=0.05, superpositionModel=SquaredSum())
    print(wf_model_ss)
    wf_model_ls = NOJ(site, windTurbines, k=0.05, superpositionModel=LinearSum())
    print(wf_model_ls)
    wf_model_ms = NOJ(site, windTurbines, k=0.05, superpositionModel=MaxSum())
    print(wf_model_ms)

    # run wind farm simulation
    sim_res_ss = wf_model_ss(x, y)
    # return SimulationResult(self, localWind=localWind,
    #                         x_i=x, y_i=y, h_i=h, type_i=type, yaw_ilk=yaw_ilk,
    #                         wd=wd, ws=ws,
    #                         WS_eff_ilk=WS_eff_ilk, TI_eff_ilk=TI_eff_ilk,
    #                         power_ilk=power_ilk, ct_ilk=ct_ilk)
    sim_res_ls = wf_model_ls(x, y)
    sim_res_ms = wf_model_ms(x, y)

    # calculate AEP
    aep_ss = sim_res_ss.aep()
    aep_ls = sim_res_ls.aep()
    aep_ms = sim_res_ms.aep()
    print(aep_ss, aep_ls, aep_ms)
    # plot wake map
    for sim_res, aep in [['sim_res_ss', 'aep_ss'], ['sim_res_ls', 'aep_ls'], ['sim_res_ms', 'aep_ms']]:
        flow_map = locals()[sim_res].flow_map(wd=30, ws=9.8)
        flow_map.plot_wake_map()
        flow_map.plot_windturbines()
        plt.title('%s AEP: %.2f GWh' % (aep, locals()[aep]))
        plt.show()
