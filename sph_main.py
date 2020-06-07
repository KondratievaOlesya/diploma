import numpy as np
import cv2
from pysph.solver.application import Application

from pysph.base.kernels import CubicSpline, WendlandQuintic

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.wc.basic import (
    MomentumEquation, TaitEOS, TaitEOSHGCorrection
)
from pysph.sph.wc.basic import (ContinuityEquationDeltaSPH,
                                MomentumEquationDeltaSPH)
from pysph.sph.basic_equations import \
    (ContinuityEquation, SummationDensity, XSPHCorrection)
from pysph.sph.wc.viscosity import LaminarViscosity
from pysph.sph.rigid_body import (
    BodyForce, RigidBodyCollision, RigidBodyMoments,
    RigidBodyMotion, AkinciRigidFluidCoupling,
    RK2StepRigidBody, RigidBodyForceGPUGems,
    PressureRigidBody, NumberDensity, SummationDensityBoundary,
    LiuFluidForce
)

import particles_generator as pg

dx = 1 * 1e-1
rho0 = 1000
gamma = 7.0
dim = 3
hdx = 1.2
h0 = dx * hdx
hdx = hdx
gx = 0
gy = 0
gz = -9.81
alpha = 0.1
beta = 0.0
delta = 0.1
nu = 3


class Mudflow(Application):
    def create_particles(self):
        particles = pg.get_test6_5()
        return particles

    def create_solver(self):
        kernel = CubicSpline(dim=3)

        integrator = EPECIntegrator(fluid=WCSPHStep(), surface=WCSPHStep(),
                                    obstacle=RK2StepRigidBody())

        c0 = 10 * np.sqrt(10 * 9.81 * 1)
        dt = 0.125 * dx * hdx / (c0 * 1.1) / 2.
        print("DT: %s" % dt)
        tf = 6
        solver = Solver(
            kernel=kernel,
            dim=3,
            integrator=integrator,
            dt=dt,
            tf=tf,
            adaptive_timestep=False, )

        return solver

    def create_equations(self):
        c0 = 10 * np.sqrt(10 * 9.81 * 1)
        co = c0
        wood_rho = 1000
        solid_rho = 1000
        ro = rho0
        equations = [
            Group(equations=[
                BodyForce(dest='obstacle', sources=None, gz=-9.81),
            ], real=False),
            Group(equations=[
                ContinuityEquation(dest='fluid',
                                   sources=['fluid', 'surface', 'obstacle']),
                ContinuityEquation(dest='surface',
                                   sources=['surface', 'fluid', 'obstacle'])
            ]),

            # Tait equation of state
            Group(equations=[
                TaitEOSHGCorrection(dest='fluid', sources=None, rho0=ro,
                                    c0=co, gamma=7.0),
                TaitEOSHGCorrection(dest='surface', sources=None, rho0=ro,
                                    c0=co, gamma=7.0),
            ], real=False),
            Group(equations=[
                MomentumEquation(dest='fluid', sources=['fluid', 'surface'],
                                 alpha=alpha, beta=0.0, c0=co,
                                 gz=-9.81),
                AkinciRigidFluidCoupling(dest='fluid', sources=['obstacle']),
                # LiuFluidForce(
                #     dest='fluid',
                #     sources=['obstacle'], ),
                LaminarViscosity(
                    dest='fluid', sources=['fluid'], nu=nu
                ),
                XSPHCorrection(dest='fluid', sources=['fluid', 'surface']),
            ]),
            Group(equations=[
                RigidBodyCollision(dest='obstacle', sources=['surface', 'obstacle'],
                                   kn=1e5)
            ]),
            Group(equations=[RigidBodyMoments(dest='obstacle', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='obstacle', sources=None)]),
        ]

        return equations


if __name__ == '__main__':
    app = Mudflow()
    app.run()
