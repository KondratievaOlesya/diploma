import numpy as np
import cv2
import pyximport
from pysph.solver.utils import load
pyximport.install()
from pysph.base.utils import get_particle_array
from pysph.base.utils import (get_particle_array_wcsph,
                              get_particle_array_rigid_body)
import ctypes
from pysph.tools.geometry_stl import get_stl_surface
def get_test_particles():
    _dx = 4
    dx = _dx * 1e-2
    dx = dx
    hdx = 1.2
    ro = 1000
    solid_rho = 800
    m = 1000 * dx * dx * dx
    co = 2 * np.sqrt(2 * 9.81 * 150 * 1e-3)
    alpha = 0.1
    size = 1
    x = []
    y = []
    z = []
    # Tank
    # first wall
    for j in np.arange(0, size, dx):
        for i in np.arange(0, size, dx):
            x.append(0)
            y.append(i)
            z.append(j)
    # second wall
    for j in np.arange(0, size, dx):
        for i in np.arange(0, size, dx):
            x.append(i)
            y.append(0)
            z.append(j)

    # third wall
    for j in np.arange(0, size, dx):
        for i in np.arange(0, size, dx):
            x.append(i)
            y.append(size)
            z.append(j)

    # fourth wall
    for j in np.arange(0, size, dx):
        for i in np.arange(0, size, dx):
            x.append(size)
            y.append(i)
            z.append(j)
    # bottom
    for j in np.arange(0, size, dx):
        for i in np.arange(0, size, dx):
            x.append(i)
            y.append(j)
            z.append(0)
    # Incline surface
    start = size * 0.4
    finish = size
    xs = []; ys = []; zs = []
    for j in np.arange(start, finish, dx):
        for i in np.arange(0, size, dx):
            xs.append(j)
            ys.append(i)
            zs.append(j - start)
    # Fluid
    start = size * 0.6
    finish = size
    xf = []; yf = []; zf = []
    start_height = size * 0.2
    for layer in range(1, 6):
        for j in np.arange(start, finish, dx):
            for i in np.arange(0, size, dx):
                xf.append(j)
                yf.append(i)
                zf.append(start_height + j - start + layer * dx)
    # Obstacle
    xx, yy, zz = np.mgrid[
                 size * 0.3: size * 0.4:dx,
                 size * 0.2:size * 0.8:dx,
                 dx:size * 0.5:dx
                 ]
    xo = xx.ravel(); yo = yy.ravel(); zo = zz.ravel()
    x_solid = x + xs
    y_solid = y + ys
    z_solid = z + zs

    m = ro * dx * dx * dx
    rho = ro
    h = hdx * dx
    fluid = get_particle_array_wcsph(x=xf, y=yf, z=zf, h=h, m=m, rho=rho,
                                     name="fluid")
    m = 1000 * dx ** 3
    rho = 1000
    rad_s = dx / 2.
    h = hdx * dx
    V = dx ** 3
    surface = get_particle_array_wcsph(x=x_solid, y=y_solid, z=z_solid, h=h, m=m, rho=rho,
                                       rad_s=rad_s, V=V, name="surface")
    rho = 2300
    m = rho * (dx / 2.) ** 3
    rad_s = dx / 4.
    V = (dx / 2.) ** 3
    cs = 0.0
    obstacle = get_particle_array_rigid_body(x=xo, y=yo, z=zo, h=h, m=m,
                                             rho=rho, rad_s=rad_s, V=V, cs=cs, rho0=rho,
                                             name="obstacle")
    obstacle.total_mass[0] = np.sum(m)
    obstacle.add_property('cs')
    obstacle.add_property('arho')
    obstacle.set_lb_props(list(obstacle.properties.keys()))
    obstacle.set_output_arrays(
        ['x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'fz',
         'rho', 'm', 'h', 'p', 'tag', 'pid', 'gid']
    )
    return [fluid, surface, obstacle]


def get_test1():
    size = 200
    layers = 10
    dx = 0.04
    img = cv2.imread("groznyi.png", 0)
    img = cv2.resize(img, (size, size))
    # surface
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    x = xx.ravel()
    x = x * dx
    _x = x
    for i in range(0, layers - 1):
        x = np.concatenate([x, _x])

    y = yy.ravel()
    y = y * dx
    _y = y
    for i in range(0, layers - 1):
        y = np.concatenate([y, _y])

    z = img.ravel()
    z = z * dx
    _z = z
    for i in range(0, layers - 1):
        z = np.concatenate([z, (_z - dx * (i + 1))])
    # fluid
    layers = 10
    x_start = int(size * 0.2)
    x_finish = int(size * 0.6)
    y_start = int(size * 0.1)
    y_finish = int(size * 0.2)
    xf = []; yf = []; zf = []
    for mult in range(1, layers + 1):
        for i in range(x_start, x_finish):
            for j in range(y_start, y_finish):
                zf.append(img[i, j] * dx + (mult * dx))
                xf.append(i * dx)
                yf.append(j * dx)
    #obstacle
    xo = []; yo = []; zo = []
    z_row = img.ravel()
    delimiter = np.amax(z_row)*0.09
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    x_row = xx.ravel()
    y_row = yy.ravel()
    for i in range(0, len(z_row)):
        if z_row[i] >= delimiter and y_row[i] >= 0.3 * size:
            zo.append(z_row[i] + 1)
            xo.append(x_row[i])
            yo.append(y_row[i])
    xs = []; ys = []; zs = []
    for i in range(0, len(x)):
        if not(z[i] >= (delimiter * dx) and y[i] >= 0.3 * size * dx):
            xs.append(x[i])
            ys.append(y[i])
            zs.append(z[i])
    xo = np.array(xo) * dx
    yo = np.array(yo) * dx
    zo = np.array(zo) * dx
    # set particles
    ro = 1000
    m = ro * dx * dx * dx
    rho = ro
    hdx = 1.2
    h = hdx * dx
    fluid = get_particle_array_wcsph(x=xf, y=yf, z=zf, h=h, m=m, rho=rho,
                                     name="fluid")
    m = 1000 * dx ** 3
    rho = 1000
    rad_s = dx / 2.
    h = hdx * dx
    V = dx ** 3
    surface = get_particle_array_wcsph(x=xs, y=ys, z=zs, h=h, m=m, rho=rho,
                                       rad_s=rad_s, V=V, name="surface")
    rho = 2300
    m = rho * (dx / 2.) ** 3
    rad_s = dx / 4.
    V = (dx / 2.) ** 3
    cs = 0.0
    obstacle = get_particle_array_rigid_body(x=xo, y=yo, z=zo, h=h, m=m,
                                             rho=rho, rad_s=rad_s, V=V, cs=cs, rho0=rho,
                                             name="obstacle")
    return [fluid, surface, obstacle]

def get_test2():
    size = 200
    layers = 10
    dx = 0.04
    hdx = 1.2
    h = hdx * dx
    x, y, z = get_stl_surface('123.stl', 1, 1)
    z_row = z
    x = x*dx
    _x = x
    for i in range(0, layers - 1):
        x = np.concatenate([x, _x])

    y = y * dx
    _y = y
    for i in range(0, layers - 1):
        y = np.concatenate([y, _y])

    z = z * dx
    _z = z
    for i in range(0, layers - 1):
        z = np.concatenate([z, (_z - dx * (i + 1))])
    # fluid
    layers = 10
    x_start = int(size * 0.2)
    x_finish = int(size * 0.6)
    y_start = int(size * 0.1)
    y_finish = int(size * 0.2)
    xf = []; yf = []; zf = []
    for mult in range(1, layers + 1):
        for i in range(x_start, x_finish):
            for j in range(y_start, y_finish):
                zf.append(z[i] * dx + (mult * dx))
                xf.append(i * dx)
                yf.append(j * dx)
    # obstacle
    xo = []; yo = []; zo = []
    z_row = z_row
    delimiter = np.amax(z_row)
    xx, yy = np.mgrid[0:np.sqrt(len(x)), 0:np.sqrt(len(y))]
    x_row = xx.ravel()
    y_row = yy.ravel()
    for i in range(0, len(z_row)):
        if z_row[i] >= delimiter and y_row[i] >= 0.3 * size:
            zo.append(z_row[i] + 1)
            xo.append(x_row[i])
            yo.append(y_row[i])
    xs = []; ys = []; zs = []
    for i in range(0, len(x)):
        if not (z[i] >= (delimiter * dx) and y[i] >= 0.3 * size * dx):
            xs.append(x[i])
            ys.append(y[i])
            zs.append(z[i])
    xo = np.array(xo) * dx
    yo = np.array(yo) * dx
    zo = np.array(zo) * dx
    # set particles
    ro = 1000
    m = ro * dx * dx * dx
    rho = ro
    hdx = 1.2
    h = hdx * dx
    fluid = get_particle_array_wcsph(x=xf, y=yf, z=zf, h=h, m=m, rho=rho,
                                     name="fluid")
    m = 1000 * dx ** 3
    rho = 1000
    rad_s = dx / 2.
    h = hdx * dx
    V = dx ** 3
    surface = get_particle_array_wcsph(x=xs, y=ys, z=zs, h=h, m=m, rho=rho,
                                       rad_s=rad_s, V=V, name="surface")
    rho = 2300
    m = rho * (dx / 2.) ** 3
    rad_s = dx / 4.
    V = (dx / 2.) ** 3
    cs = 0.0
    obstacle = get_particle_array_rigid_body(x=xo, y=yo, z=zo, h=h, m=m,
                                             rho=rho, rad_s=rad_s, V=V, cs=cs, rho0=rho,
                                             name="obstacle")
    return [fluid, surface, obstacle]

def get_test3():
    dx = 0.04
    hdx = 1.2
    h = hdx * dx
    x, y, z = get_stl_surface('surface.stl', 1, 1)
    xf, yf, zf = get_stl_surface('fluid.stl', 1, 1)
    xo, yo, zo = get_stl_surface('obstacle.stl', 1, 1)
    ro = 1000
    x = x*dx; y = y*dx; z = z*dx
    xf = xf*dx; yf=yf*dx; zf=zf*dx
    xo = xo*dx; yo=yo*dx;zo=zo*dx
    inter_with_idx = indices = np.arange(zo.shape[0])[np.in1d(zo, z)]
    xo = np.delete(xo, inter_with_idx)
    yo = np.delete(yo, inter_with_idx)
    zo = np.delete(zo, inter_with_idx)
    m = ro * dx * dx * dx
    rho = ro
    hdx = 1.2
    h = hdx * dx
    fluid = get_particle_array_wcsph(x=xf, y=yf, z=zf, h=h, m=m, rho=rho,
                                     name="fluid")
    m = 1000 * dx ** 3
    rho = 1000
    rad_s = dx / 2.
    h = hdx * dx
    V = dx ** 3
    surface = get_particle_array_wcsph(x=x, y=y, z=z, h=h, m=m, rho=rho,
                                       rad_s=rad_s, V=V, name="surface")
    rho = 2300
    m = rho * (dx / 2.) ** 3
    rad_s = dx / 4.
    V = (dx / 2.) ** 3
    cs = 0.0
    obstacle = get_particle_array_rigid_body(x=xo, y=yo, z=zo, h=h, m=m,
                                             rho=rho, rad_s=rad_s, V=V, cs=cs, rho0=rho,
                                             name="obstacle")
    return [fluid, surface, obstacle]

def get_test4():
    dx = 0.04
    hdx = 1.2
    h = hdx * dx
    x, y, z = get_stl_surface('surface_a.stl', 1, 1)
    xf, yf, zf = get_stl_surface('fluid_a.stl', 1, 1)
    xo, yo, zo = get_stl_surface('obstacle_a.stl', 1, 1)
    ro = 1000
    x = x*dx; y = y*dx; z = z*dx
    xf = xf*dx; yf=yf*dx; zf=zf*dx
    xo = xo*dx; yo=yo*dx;zo=zo*dx
    inter_with_idx = indices = np.arange(zo.shape[0])[np.in1d(zo, z)]
    xo = np.delete(xo, inter_with_idx)
    yo = np.delete(yo, inter_with_idx)
    zo = np.delete(zo, inter_with_idx)
    m = ro * dx * dx * dx
    rho = ro
    hdx = 1.2
    h = hdx * dx
    fluid = get_particle_array_wcsph(x=xf, y=yf, z=zf, h=h, m=m, rho=rho,
                                     name="fluid")
    m = 1000 * dx ** 3
    rho = 1000
    rad_s = dx / 2.
    h = hdx * dx
    V = dx ** 3
    surface = get_particle_array_wcsph(x=x, y=y, z=z, h=h, m=m, rho=rho,
                                       rad_s=rad_s, V=V, name="surface")
    rho = 2300
    m = rho * (dx / 2.) ** 3
    rad_s = dx / 4.
    V = (dx / 2.) ** 3
    cs = 0.0
    obstacle = get_particle_array_rigid_body(x=xo, y=yo, z=zo, h=h, m=m,
                                             rho=rho, rad_s=rad_s, V=V, cs=cs, rho0=rho,
                                             name="obstacle")
    return [fluid, surface, obstacle]

def get_test5():
    dx = 0.04
    hdx = 1.2
    h = hdx * dx
    x, y, z = get_stl_surface('surface_plane.stl', 1, 1)
    xf, yf, zf = get_stl_surface('fluid_plane.stl', 1, 1)
    xo, yo, zo = get_stl_surface('obstacle_plane.stl', 1, 1)
    ro = 1000
    x = x*dx; y = y*dx; z = z*dx
    xf = xf*dx; yf=yf*dx; zf=zf*dx
    xo = xo*dx; yo=yo*dx;zo=zo*dx
    inter_with_idx = indices = np.arange(zo.shape[0])[np.in1d(zo, z)]
    xo = np.delete(xo, inter_with_idx)
    yo = np.delete(yo, inter_with_idx)
    zo = np.delete(zo, inter_with_idx)
    m = ro * dx * dx * dx
    rho = ro
    hdx = 1.2
    h = hdx * dx
    fluid = get_particle_array_wcsph(x=xf, y=yf, z=zf, h=h, m=m, rho=rho,
                                     name="fluid")
    m = 1000 * dx ** 3
    rho = 1000
    rad_s = dx / 2.
    h = hdx * dx
    V = dx ** 3
    surface = get_particle_array_wcsph(x=x, y=y, z=z, h=h, m=m, rho=rho,
                                       rad_s=rad_s, V=V, name="surface")
    rho = 2300
    m = rho * (dx / 2.) ** 3
    rad_s = dx / 4.
    V = (dx / 2.) ** 3
    cs = 0.0
    obstacle = get_particle_array_rigid_body(x=xo, y=yo, z=zo, h=h, m=m,
                                             rho=rho, rad_s=rad_s, V=V, cs=cs, rho0=rho,
                                             name="obstacle")
    return [fluid, surface, obstacle]

def get_test6():
    dx = 0.1
    hdx = 1.2
    h = hdx * dx
    x, y, z = get_stl_surface('stl/surface_tiny.stl', dx, h)
    min_z = np.min(z)
    xf, yf, zf = get_stl_surface('stl/fluid_tiny.stl', dx, h)
    body_ids = np.array([], dtype=int)
    xo = np.array([]); yo = np.array([]); zo = np.array([])
    for i in range(1, 20):
        if i == 13:
            continue
        filename = 'stl/obstacle_tiny/' + str(i) + '.stl'
        x0, y0, z0 = get_stl_surface(filename, dx, h)
        z0 = z0 + 1.8*dx
        indx_to_delete = np.nonzero(zo <= min_z+1.8*dx)
        x0 = np.delete(x0, indx_to_delete)
        y0 = np.delete(y0, indx_to_delete)
        z0 = np.delete(z0, indx_to_delete)

        xo = np.concatenate((xo, x0))
        yo = np.concatenate((yo, y0))
        zo = np.concatenate((zo, z0))
        if i > 13:
            body_id = np.ones_like(x0, dtype=int) * int(i - 2)
        else:
            body_id = np.ones_like(x0, dtype=int) * int(i - 1)
        for j in body_id:
            body_ids = np.append(body_ids, j)
    ro = 1000
    m = ro * dx * dx * dx
    rho = ro
    hdx = 1.2
    h = hdx * dx
    fluid = get_particle_array_wcsph(x=xf, y=yf, z=zf, h=h, m=m, rho=rho,
                                     name="fluid")
    m = 1000 * dx ** 3
    rho = 1000
    rad_s = dx / 2.
    h = hdx * dx
    V = dx ** 3
    surface = get_particle_array_wcsph(x=x, y=y, z=z, h=h, m=m, rho=rho,
                                       rad_s=rad_s, V=V, name="surface")
    rho = 2300
    m = rho * (dx / 2.) ** 3
    rad_s = dx / 4.
    V = (dx / 2.) ** 3
    cs = 0.0
    obstacle = get_particle_array_rigid_body(x=xo, y=yo, z=zo, h=h, m=m,
                                             rho=rho, rad_s=rad_s, V=V, cs=cs, rho0=rho, body_id=body_ids,
                                             name="obstacle")
    return [fluid, surface, obstacle]

def get_test6_5():
    dx = 0.1
    hdx = 1.2
    h = hdx * dx
    x, y, z = get_stl_surface('stl/surface_arshan.stl', dx, h)
    min_z = np.min(z)
    xf, yf, zf = get_stl_surface('stl/fluid_arshan.stl', dx, h)
    zf = zf
    body_ids = np.array([], dtype=int)
    xo = np.array([]); yo = np.array([]); zo = np.array([])
    for i in range(1, 19):
        filename = 'stl/obstacle_arshan/' + str(i) + '.stl'
        x0, y0, z0 = get_stl_surface(filename, dx, h)
        z0 = z0 + dx
        indx_to_delete = np.nonzero(zo <= min_z+dx)
        x0 = np.delete(x0, indx_to_delete)
        y0 = np.delete(y0, indx_to_delete)
        z0 = np.delete(z0, indx_to_delete)

        xo = np.concatenate((xo, x0))
        yo = np.concatenate((yo, y0))
        zo = np.concatenate((zo, z0))
        body_id = np.ones_like(x0, dtype=int) * int(i - 1)
        for j in body_id:
            body_ids = np.append(body_ids, j)
    ro = 1000
    m = ro * dx * dx * dx
    rho = ro
    hdx = 1.2
    h = hdx * dx
    fluid = get_particle_array_wcsph(x=xf, y=yf, z=zf, h=h, m=m, rho=rho,
                                     name="fluid")

    rho = 1000
    m = rho * dx ** 3
    h = hdx * dx
    rad_s = dx / 2.
    h = hdx * dx
    V = dx ** 3
    surface = get_particle_array_wcsph(x=x, y=y, z=z, h=h, m=m, rho=rho, rad_s=rad_s, V=V,
                                      name="surface")
    rho = 1000
    m = rho * (dx) ** 3
    rad_s = dx / 4.
    V = (dx / 2.) ** 3
    cs = 0.0
    obstacle = get_particle_array_rigid_body(x=xo, y=yo, z=zo, h=h, m=m,
                                             rho=rho, rho0=rho, body_id=body_ids, rad_s=rad_s, V=V, cs=cs,
                                             name="obstacle")
    obstacle.total_mass[0] = np.sum(m)
    obstacle.add_property('cs')
    obstacle.add_property('arho')
    obstacle.set_lb_props(list(obstacle.properties.keys()))
    obstacle.set_output_arrays(
        ['x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'fz',
         'rho', 'm', 'h', 'p', 'tag', 'pid', 'gid']
    )
    fluid.set_lb_props(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'gid',
                                           'x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'rho0'])

    # fluid.set_lb_props(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'gid',
    #                    'x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'rho0'])
    fluid.set_lb_props(list(fluid.properties.keys()))

    # surface.set_lb_props(['x', 'y', 'z', 'rho', 'h', 'm', 'gid', 'rho0'])
    # obstacle.set_lb_props(['x', 'y', 'z', 'rho', 'h', 'm', 'gid', 'rho0'])
    surface.set_lb_props(list(surface.properties.keys()))

    # surface and obstacle particles can do with a reduced list of properties
    # to be saved to disk since they are fixed
    surface.set_output_arrays(['x', 'y', 'z', 'rho', 'm', 'h', 'p', 'tag', 'pid', 'gid'])
    surface.add_property('V')
    surface.add_property('fx')
    surface.add_property('fy')
    surface.add_property('fz')
    return [fluid, surface, obstacle]

def get_test7():
    dx = 0.1
    hdx = 1.2
    h = hdx * dx
    x, y, z = get_stl_surface('stl/surface_small.stl', dx, h)
    xf, yf, zf = get_stl_surface('stl/fluid_small.stl', dx, h)
    body_ids = np.array([])
    xo = np.array([])
    yo = np.array([])
    zo = np.array([])
    for i in range(1, 11):
        filename = 'stl/obstacle_small/' + str(i) + '.stl'
        x0, y0, z0 = get_stl_surface(filename, dx, h)
        xo = np.concatenate((xo, x0))
        yo = np.concatenate((yo, y0))
        zo = np.concatenate((zo, z0))
        body_id = np.ones_like(x0, dtype=int) * (i - 1)
        body_ids = np.concatenate((body_ids, body_id))
    ro = 1000
    m = ro * dx * dx * dx
    rho = ro
    hdx = 1.2
    h = hdx * dx
    fluid = get_particle_array(x=xf, y=yf, z=zf,
                                     name="fluid")
    m = 1000 * dx ** 3
    rho = 1000
    rad_s = dx / 2.
    h = hdx * dx
    V = dx ** 3
    surface = get_particle_array(x=x, y=y, z=z, name="surface")
    rho = 2300
    m = rho * (dx / 2.) ** 3
    rad_s = dx / 4.
    V = (dx / 2.) ** 3
    cs = 0.0
    obstacle = get_particle_array(x=xo, y=yo, z=zo,
                                             name="obstacle")
    return [fluid, surface, obstacle]

def restart_from_file(name):
    dx = 0.1
    hdx = 1.2
    h = hdx * dx
    file = load(name)

    particles = file['arrays']
    x, y, z = [particles['surface'].x, particles['surface'].y, particles['surface'].z]
    xf, yf, zf = [particles['fluid'].x, particles['fluid'].y, particles['fluid'].z]
    xo, yo, zo = [particles['obstacle'].x, particles['obstacle'].y, particles['obstacle'].z]
    zo = zo + 2 * dx
    body_ids = particles['obstacle'].body_id
    ro = 1000
    m = ro * dx * dx * dx
    rho = ro
    hdx = 1.2
    h = hdx * dx
    fluid = get_particle_array_wcsph(x=xf, y=yf, z=zf, h=h, m=m, rho=rho,
                                     name="fluid")

    rho = 1000
    m = rho * (dx ** 3)
    h = hdx * dx
    surface = get_particle_array_wcsph(x=x, y=y, z=z, h=h, m=m, rho=rho,
                                       name="surface")
    rho = 1000
    m = rho * (dx ** 3)
    obstacle = get_particle_array_rigid_body(x=xo, y=yo, z=zo, h=h, m=m,
                                             rho=rho, rho0=rho, body_id=body_ids,
                                             name="obstacle")
    obstacle.total_mass[0] = np.sum(m)
    obstacle.add_property('cs')
    obstacle.add_property('arho')
    obstacle.set_lb_props(list(obstacle.properties.keys()))
    obstacle.set_output_arrays(
        ['x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'fz',
         'rho', 'm', 'h', 'p', 'tag', 'pid', 'gid']
    )
    fluid.set_lb_props(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'gid',
                        'x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'rho0'])

    # fluid.set_lb_props(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'gid',
    #                    'x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'rho0'])
    fluid.set_lb_props(list(fluid.properties.keys()))

    # surface.set_lb_props(['x', 'y', 'z', 'rho', 'h', 'm', 'gid', 'rho0'])
    # obstacle.set_lb_props(['x', 'y', 'z', 'rho', 'h', 'm', 'gid', 'rho0'])
    surface.set_lb_props(list(surface.properties.keys()))

    # surface and obstacle particles can do with a reduced list of properties
    # to be saved to disk since they are fixed
    surface.set_output_arrays(['x', 'y', 'z', 'rho', 'm', 'h', 'p', 'tag', 'pid', 'gid'])
    surface.add_property('V')
    surface.add_property('fx')
    surface.add_property('fy')
    surface.add_property('fz')
    return [fluid, surface, obstacle]

def get_test8():
    dx = 0.1
    hdx = 1.2
    h = hdx * dx
    x, y, z = get_stl_surface('stl/surface_arshan.stl', dx, h)
    min_z = np.min(z)
    xf, yf, zf = get_stl_surface('stl/fluid_arshan.stl', dx, h)
    zf = zf
    body_ids = np.array([], dtype=int)
    xo = np.array([]); yo = np.array([]); zo = np.array([])
    for i in range(1, 18):
        filename = 'stl/obstacle_arshan/' + str(i) + '.stl'
        x0, y0, z0 = get_stl_surface(filename, dx, h)
        z0 = z0 + 3*dx
        indx_to_delete = np.nonzero(zo <= min_z+3*dx)
        x0 = np.delete(x0, indx_to_delete)
        y0 = np.delete(y0, indx_to_delete)
        z0 = np.delete(z0, indx_to_delete)

        xo = np.concatenate((xo, x0))
        yo = np.concatenate((yo, y0))
        zo = np.concatenate((zo, z0))
        body_id = np.ones_like(x0, dtype=int) * int(i - 1)
        for j in body_id:
            body_ids = np.append(body_ids, j)
    xw, yw, zw = get_stl_surface('stl/obstacle_arshan/18.stl', dx, h)
    ro = 1000
    m = ro * dx * dx * dx
    rho = ro
    hdx = 1.2
    h = hdx * dx
    fluid = get_particle_array_wcsph(x=xf, y=yf, z=zf, h=h, m=m, rho=rho,
                                     name="fluid")

    rho = 1000
    m = rho * dx ** 3
    h = hdx * dx
    m = 1000 * dx ** 3
    rad_s = np.ones_like(x) * 2 / 2. * 1e-3
    surface = get_particle_array_wcsph(x=x, y=y, z=z, h=h, m=m, rho=rho, rad_s=rad_s,
                                      name="surface")
    rho = 1000
    m = rho * (dx) ** 3
    rad_s = np.ones_like(xo) * dx / 2. * 1e-3
    cs = np.zeros_like(xo)
    obstacle = get_particle_array_rigid_body(x=xo, y=yo, z=zo, h=h, m=m, rad_s=rad_s,
                                             rho=rho, rho0=rho, body_id=body_ids, cs=cs,
                                             name="obstacle")
    cs = np.zeros_like(xw)
    rad_s = np.ones_like(xw) * dx / 2. * 1e-3
    wood = get_particle_array_rigid_body(x=xw, y=yw, z=zw, h=h, m=m,
                                         rho=rho, rho0=rho, cs=cs, rad_s = rad_s,
                                         name="wood")
    obstacle.total_mass[0] = np.sum(m)
    obstacle.add_property('cs')
    obstacle.add_property('arho')
    obstacle.set_lb_props(list(obstacle.properties.keys()))
    obstacle.set_output_arrays(
        ['x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'fz',
         'rho', 'm', 'h', 'p', 'tag', 'pid', 'gid']
    )
    fluid.set_lb_props(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'gid',
                                           'x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'rho0'])

    # fluid.set_lb_props(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'gid',
    #                    'x0', 'y0', 'z0', 'u0', 'v0', 'w0', 'rho0'])
    fluid.set_lb_props(list(fluid.properties.keys()))

    # surface.set_lb_props(['x', 'y', 'z', 'rho', 'h', 'm', 'gid', 'rho0'])
    # obstacle.set_lb_props(['x', 'y', 'z', 'rho', 'h', 'm', 'gid', 'rho0'])
    surface.set_lb_props(list(surface.properties.keys()))

    # surface and obstacle particles can do with a reduced list of properties
    # to be saved to disk since they are fixed
    surface.set_output_arrays(['x', 'y', 'z', 'rho', 'm', 'h', 'p', 'tag', 'pid', 'gid'])
    surface.add_property('V')
    surface.add_property('fx')
    surface.add_property('fy')
    surface.add_property('fz')
    return [fluid, surface, obstacle, wood]