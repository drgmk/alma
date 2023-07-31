# attempts to speed up imaging
# code is the same as in alma.image, but uses tf

import tensorflow as tf

# this will disable tf.function (not clear it helps here)
# tf.config.run_functions_eagerly(True)

# need float64 to play nicely with python calls to dens function
dt = tf.dtypes.float64

# use this to see whether calls use CPU/GPU
# tf.debugging.set_log_device_placement(True)


@tf.function
def dens(r, az, el, p_dens):
    r_ecc = p_dens[0] * (1. - p_dens[1]**2) / (1. + p_dens[1]*tf.math.cos(az))
    out = tf.math.exp(-0.5*((r-r_ecc)/p_dens[2])**2)/2.5066282746/p_dens[2] * \
          tf.math.exp(-0.5*( el/p_dens[3] )**2) * \
          (1. - p_dens[1]*tf.math.cos(az))
    return out


def img(p_, x_, yarray_, zarray_, arcsec_pix_, axisym,
        p_dens_, docube=False):

    p = tf.constant(p_, dtype=dt)
    x = tf.constant(x_, dtype=dt)
    yarray = tf.constant(yarray_, dtype=dt)
    zarray = tf.constant(zarray_, dtype=dt)
    arcsec_pix = tf.constant(arcsec_pix_, dtype=dt)
    p_dens = tf.constant(p_dens_, dtype=dt)

    @tf.function
    def dens(r, az, el, p):
        return dens_(r, az, el, p)

    deg2rad = tf.constant(0.017453292519943295, dtype=dt)
    piby2 = tf.constant(1.5707963267948966, dtype=dt)

    x0, y0, pos, anom, inc, tot = p
    yarray = yarray - y0 / arcsec_pix

    # zyz rotation, a.c.w so use -ve angles for cube -> model
    c0 = tf.math.cos(deg2rad*-pos)
    s0 = tf.math.sin(deg2rad*-pos)
    c1 = tf.math.cos(deg2rad*-inc)
    s1 = tf.math.sin(deg2rad*-inc)
    c2 = tf.math.cos(deg2rad*-anom-piby2)  # to get from N to x
    s2 = tf.math.sin(deg2rad*-anom-piby2)
    c0c1c2 = c0 * c1 * c2
    c0c1s2 = c0 * c1 * s2
    c0s1 = c0 * s1
    s0s2 = s0 * s2
    s0c1 = s0 * c1
    s0c2 = s0 * c2
    c0c1c2_s0s2 = c0c1c2 - s0s2
    c0c1s2_s0c2 = c0c1s2 + s0c2

    # x-independent parts of the coordinate transformation
    trans1 = -(s0c1*c2 + c0*s2)*yarray + s1*c2*zarray
    trans2 = (-s0c1*s2 + c0*c2)*yarray + s1*s2*zarray
    trans3 = s0*s1*yarray + c1*zarray

    if not docube:
        return (tot * image(x0, x, c0c1c2_s0s2, c0c1s2_s0c2, c0s1,
                           trans1, trans2, trans3, arcsec_pix, axisym, p_dens)).numpy()

    x = x - x0 / arcsec_pix

    xsh = x.shape
    trsh = trans1.shape
    x = tf.broadcast_to(x, trsh+xsh)
    trans1_ = tf.broadcast_to(trans1, xsh+trsh)
    trans1_ = tf.transpose(trans1_, [1, 2, 0])
    trans2_ = tf.broadcast_to(trans2, xsh+trsh)
    trans2_ = tf.transpose(trans2_, [1, 2, 0])
    trans3_ = tf.broadcast_to(trans3, xsh+trsh)
    trans3_ = tf.transpose(trans3_, [1, 2, 0])

    x3 = c0c1c2_s0s2*x + trans1_
    y3 = c0c1s2_s0c2*x + trans2_
    z3 = -c0s1*x + trans3_

    if docube:
        return (tot * cube(x3, y3, z3, arcsec_pix, axisym, p_dens)).numpy()


# @tf.function
def cube(x3, y3, z3, arcsec_pix, axisym, p_dens):

    rxy2 = x3**2 + y3**2
    rxy = tf.math.sqrt(rxy2)
    r = tf.math.sqrt(rxy2 + z3**2) * arcsec_pix
    if axisym:
        az = tf.constant(0.0, dtype=dt)
    else:
        az = tf.math.atan2(y3, x3)
    el = tf.math.atan2(z3, rxy)

    cube_ = dens(r, az, el, p_dens)
    # r_ecc = p_dens[0] * (1. - p_dens[1]**2) / (1. + p_dens[1]*tf.math.cos(az))
    # cube_ = tf.math.exp(-0.5*((r-r_ecc)/p_dens[2])**2)/2.5066282746/p_dens[2] * \
    #     tf.math.exp(-0.5*( el/p_dens[3] )**2) * \
    #     (1. - p_dens[1]*tf.math.cos(az))
    cube_ = cube_ / r

    cube_ = tf.transpose(cube_, [0, 2, 1])
    return cube_ / tf.math.reduce_sum(cube_)


@tf.function
def image(x0, x, c0c1c2_s0s2, c0c1s2_s0c2, c0s1, trans1, trans2, trans3, arcsec_pix, axisym, p_dens):

    out_list = tf.TensorArray(dtype=dt, size=x.shape[0])
    for i in tf.range(x.shape[0]):

        x1 = x[i] - x0 / arcsec_pix

        # x,y,z locations in original model coords
        x3 = c0c1c2_s0s2*x1 + trans1
        y3 = c0c1s2_s0c2*x1 + trans2
        z3 = -c0s1*x1 + trans3

        rxy2 = x3**2 + y3**2
        rxy = tf.sqrt(rxy2)
        r = tf.sqrt(rxy2 + z3**2) * arcsec_pix
        # these two lines are as expensive as dens below
        if axisym:
            az = tf.constant(0.0, dtype=dt)
        else:
            az = tf.math.atan2(y3, x3)
        el = tf.math.atan2(z3, rxy)

        # the density in this y,z layer
        layer_ = dens(r, az, el, p_dens)
        # r_ecc = p_dens[0] * (1. - p_dens[1]**2) / (1. + p_dens[1]*tf.math.cos(az))
        # layer_ = tf.math.exp(-0.5*((r-r_ecc)/p_dens[2])**2)/2.5066282746/p_dens[2] * \
        #          tf.math.exp(-0.5*( el/p_dens[3] )**2) * \
        #          (1. - p_dens[1]*tf.math.cos(az))
        layer_ = layer_ / r

        # put this in the image
        out_list = out_list.write(i, tf.reduce_sum(layer_, axis=1))

    image_ = tf.transpose(out_list.stack())
    image_ = image_ / tf.reduce_sum(image_)
    return image_
