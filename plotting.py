import matplotlib.pyplot as plt
import numpy as np
from model import renav

from astropy.coordinates import ICRS, ITRS, SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u

from math import pi
import scipy

from model import xyz_to_pixel


def rotateVector(vect, axis, theta):
    """ Rotate vector around the given axis by a given angle.
        From WesternMeteorPyLib (Denis Vida, MIT License)"""

    rot_M = scipy.linalg.expm(np.cross(np.eye(3), axis/np.linalg.norm(axis, 2)*theta))

    return np.dot(rot_M, vect)


def rotation_correction(vel_vec, traj_start):
    traj_x = traj_start[:, 0]
    traj_y = traj_start[:, 1]
    traj_z = traj_start[:, 2]

    lat_geocentric = np.arctan2(traj_z, np.sqrt(traj_x**2 + traj_y**2))
    lon = np.arctan2(traj_y, np.sqrt(traj_x**2 + traj_z**2))

    v_rot = 2*np.pi*np.linalg.norm(traj_start, 2, axis=1)*np.cos(lat_geocentric)/86164.09053  # m/s
    print(v_rot.shape)
    print(lon.shape)
    x = v_rot * -np.sin(np.radians(lon))
    y = v_rot * np.cos(np.radians(lon))
    z = np.zeros(len(x))

    rot_vel_vec = np.array([x, y, z]).T
    return vel_vec + rot_vel_vec


def zenith_attraction(vel_vec, traj_start, datetime):

    init_alts = np.linalg.norm(traj_start, 2, axis=1)

    # correct for gravitational acceleration
    vels = np.linalg.norm(vel_vec, 2, axis=1)*1000
    vels_g = np.sqrt(vels**2 - (2*6.67408*5.9722e13)/(init_alts*1000))
    # put vels in m for now

    time = Time(datetime)
    c = SkyCoord(x=-vel_vec[:, 0], y=-vel_vec[:, 1], z=-vel_vec[:, 2], frame='itrs', obstime=time)

    loc = EarthLocation.from_geocentric(x=traj_start[:, 0]*u.km, y=traj_start[:, 1]*u.km, z=traj_start[:, 2]*u.km)

    altaz = c.transform_to(AltAz(obstime=time, location=loc))
    zenith = pi/2 - np.radians(altaz.alt.value)

    delta_zenith = 2*np.arctan2((vels - vels_g)*np.tan(zenith/2), vels + vels_g)

    zenith_corrected = zenith + np.abs(delta_zenith)
    alt = np.degrees(pi/2 - zenith_corrected)
    altaz_corrected = SkyCoord(az=altaz.az.value*u.deg, alt=alt*u.deg, frame=AltAz, obstime=time, location=loc)
    c_corrected = altaz_corrected.transform_to(ITRS())
    # x_corrected = c_corrected.x
    # y_corrected = c_corrected.x
    dirs_corrected = np.array([c_corrected.x.value, c_corrected.y.value, c_corrected.z.value]).T
    dirs_corrected /= np.linalg.norm(dirs_corrected, 2, axis=1)[np.newaxis, :].T
    vel_vec_corrected = -dirs_corrected * vels_g[np.newaxis, :].T/1000
    # TODO: why negative here?

    return vel_vec_corrected

    # altaz.alt.value = np.degrees(pi/2 - zenith_corrected)


def helio_correction(eclons, eclats, vels, datetime):

    xm = -vels*np.cos(eclons)*np.cos(eclats)
    ym = -vels*np.sin(eclons)*np.cos(eclats)
    zm = -vels*np.sin(eclats)

    jd = Time(datetime, scale='tai').jd

    from jplephem.spk import SPK
    JPL_EPHEM_FILE = 'de430.bsp'
    jpl_data = SPK.open(JPL_EPHEM_FILE)

    # Calculate the position and the velocity of the Earth-Moon barycentre system
    # with respect to the Solar System Barycentre
    position_bary, velocity_bary = jpl_data[0, 3].compute_and_differentiate(jd)
    # Calculate the position of the Sun with respect to the Solar System Barycentre
    position_sun, velocity_sun = jpl_data[0, 10].compute_and_differentiate(jd)
    # Calculate the position and the velocity of the Earth with respect to the Earth-Moon barycentre
    position_earth, velocity_earth = jpl_data[3, 399].compute_and_differentiate(jd)

    # Calculate the velocity of the Earth in km/s
    velocity = (velocity_bary + velocity_earth)/86400.0

    J2000_OBLIQUITY = np.radians(23.4392911111)
    # Rotate the velocity vector to the ecliptic reference frame (from the Earth equator reference frame)
    earth_vel = rotateVector(velocity, np.array([1, 0, 0]), -J2000_OBLIQUITY)

    # Calculate the heliocentric velocity vector magnitude
    v_h = np.linalg.norm(np.array(earth_vel) + np.array([xm, ym, zm]).T, 2, axis=1)

    # Calculate the corrected meteoroid velocity vector
    xm_c = (xm + earth_vel[0])/v_h
    ym_c = (ym + earth_vel[1])/v_h
    zm_c = (zm + earth_vel[2])/v_h

    # Calculate corrected radiant in ecliptic coordinates
    # NOTE: 180 deg had to be added to L and B had to be negative arcsin to get the right results
    eclon_corrected = (np.arctan2(ym_c, xm_c) + np.pi) % (2*np.pi)
    eclat_corrected = -np.arcsin(zm_c)

    return eclon_corrected, eclat_corrected, v_h


def get_sample(pos, n=1):

    if n == 1:
        size = None
    else:
        size = n
    chain = np.random.choice(pos.chain, size=size)
    draw = np.random.choice(pos.draw, size=size)

    return chain, draw


def extract(variable, chains, draws):
    arr = variable.values
    flat = arr.reshape(-1, *arr.shape[2:])
    return flat[chains*arr.shape[0] + draws]


def plot_prf(datas, pos, n=100, prob=True, microns=True):
    plt.close()

    fig, ax = plt.subplots(1, 1)
    if microns:
        x_plot = np.linspace(-15, 10, 100)
    else:
        x_plot = np.linspace(0, 2, 100)
    for i in range(n):
        chain, draw = get_sample(pos)
        c = pos.c.values[chain, draw]
        k = pos.k.values[chain, draw]
        w = pos.w.values[chain, draw]

        for j, d in enumerate(datas):
            if microns:
                beta = c[j] - k*np.clip(x_plot-w+10, 0, np.inf)
            else:
                beta = c[j] - k*np.clip(x_plot-w, 0, np.inf)
            scaler = (1/(1+np.exp(-beta)))  # ** power
            label = None
            # label_active = None
            if i == 0:
                label = d['detectedBy']
                # label_active = label + ' (activated)'
            plt.plot(x_plot, scaler, color=['lightblue', 'red'][j], alpha=1, linewidth=0.5, label=label)

    if microns:
        plt.xlabel(r'Distance (microns) to pixel edge on one axis')
    else:
        plt.xlabel(r'Distance (pixels) to pixel center on one axis')
    if prob:
        plt.ylabel('Probability of event')
    else:
        plt.ylabel('Proportion of energy transmitted')
    plt.xlim(min(x_plot), max(x_plot))
    if microns:
        boundary = 0
    else:
        boundary = 0.5
    plt.axvline(boundary, color='white', label='Pixel edge', linewidth=1, linestyle='--')
    # if microns:
    #     for pitch in [20, 30]:
    #         plt.axvline(0-pitch/2,label=f'{pitch}$\mu m$ pixel',linewidth=1,linestyle='--')
    plt.legend(frameon=False)
    plt.ylim(0, 1)

    return fig, ax


def plot_prf_3d(datas, pos, prob=True, microns=True):
    plt.close()
    chain, draw = get_sample(pos)
    c = pos.c.values[chain, draw]
    k = pos.k.values[chain, draw]
    w = pos.w.values[chain, draw]

    if microns:
        x_plot = np.linspace(-20, 20, 50)
        y_plot = np.linspace(-20, 20, 50)
    else:
        x_plot = np.linspace(-0.8, 0.8, 50)
        y_plot = np.linspace(-0.8, 0.8, 50)
    xx, yy = np.meshgrid(x_plot, y_plot)

    if microns:
        pixel_pitch_x = datas[0]['p_pitch_x'][0]
        pixel_pitch_y = datas[0]['p_pitch_y'][0]
        d_x = np.clip(np.abs(xx)-(w+pixel_pitch_x/2-10), 0, np.inf)**2
        d_y = np.clip(np.abs(yy)-(w+pixel_pitch_y/2-10), 0, np.inf)**2
    else:
        d_x = np.clip(np.abs(xx)-w, 0, np.inf)**2
        d_y = np.clip(np.abs(yy)-w, 0, np.inf)**2

    beta = c[0] - k*np.sqrt((d_x+d_y))
    p = 1/(1+np.exp(-beta))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_proj_type('ortho')
    ax.set_zlim(0, 1)
    fig.tight_layout()
    ax.plot_surface(xx, yy, p, vmin=np.min(p), cmap='inferno', rstride=1, cstride=1, alpha=1)
    zs = [0]
    for i, z in enumerate(zs):
        if microns:
            bx = pixel_pitch_x/2
            by = pixel_pitch_y/2
            plt.plot([-bx, bx, bx, -bx, -bx], [-by, -by, by, by, -by], [z, z, z, z, z],
                     color='white', zorder=5, label=('Pixel boundary' if i == 0 else None), linestyle='--')
        else:
            plt.plot([-0.5, 0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5, -0.5], [z, z, z, z, z],
                     color='white', zorder=5, label=('Pixel boundary' if i == 0 else None), linestyle='--')

    import matplotlib.cm as cm
    m = cm.ScalarMappable(cmap=cm.inferno)
    m.set_array(p)
    plt.colorbar(m, label='Probability of event', ax=ax)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    ax.set(zticklabels=[])
    plt.legend(frameon=False)
    return fig, ax


def plot_traj(datas, pos):
    plt.close()
    plt.rcParams['grid.color'] = "darkgray"
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['axes.linewidth'] = 0.5

    fig = plt.figure(figsize=(6, 6), facecolor='black')
    fig.tight_layout()
    ax = fig.add_subplot(projection='3d')
    ax.set_proj_type('ortho')
    ax.set_facecolor('black')
    ax.tick_params(colors='gray')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    for i in range(100):
        chain, draw = get_sample(pos)
        a = pos.a[chain, draw, :].values
        r = pos.r[chain, draw].values
        lam = pos.lam[chain, draw].values
        phi = pos.phi[chain, draw].values
        b = [r*np.cos(phi)*np.cos(lam), r*np.sin(phi)*np.cos(lam), r*np.sin(lam)]
        decel = pos.decel[chain, draw].values

        if chain != 2:
            continue
        for j, d in enumerate(datas):
            t_shift = d['t_g_shift']
            traj = np.outer(a, t_shift-(decel*t_shift**2)/2).T + b
            sat_vec = d['sat_vec']
            x, y, z = renav(traj, sat_vec, 14, 6)
            ax.plot(x[:], y[:], z[:], color='xkcd:salmon', alpha=0.1, linewidth=0.5)

            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='pink', alpha=1, linewidth=0.5)
    for i, d in enumerate(datas):
        xyz = d['xyz']
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], alpha=0.5, c=d['t_shift'], marker='.', zorder=10)

    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    from matplotlib.ticker import LinearLocator
    ax.zaxis.set_major_locator(LinearLocator(3))

    plt.title('Trajectories in ECEF', color='white')
    return fig, ax


def plot_sensorplane(datas, pos=None, reg=None):
    for j, d in enumerate(datas):
        plt.close()
        plt.rcParams['grid.color'] = "darkgray"
        plt.rcParams['grid.linewidth'] = 0.5
        plt.rcParams['axes.linewidth'] = 0.5
        fig = plt.figure(figsize=(2, 2), facecolor='black')
        fig.tight_layout()
        ax = fig.add_subplot()
        ax.set_facecolor('black')
        ax.tick_params(colors='gray')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        sat_vec = d['sat_vec']

        plt.style.use('dark_background')

        on = np.sum(d['p_active'], axis=1) > 0
        plt.scatter(d['p_x'][~on], d['p_y'][~on], color='red', label='no events', s=5)
        # plt.scatter(d['p_x'][on],d['p_y'][on],color='blue',label='events',s=5)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')

        x, y = xyz_to_pixel(d['xyz'], sat_vec, reg=reg)
        ax.scatter(x, y, alpha=0.5, c=d['t_shift'], marker='.', zorder=10, edgecolor='white', s=80)

        if pos is not None:
            for i in range(100):
                chain, draw = get_sample(pos)
                a = pos.a[chain, draw, :].values
                r = pos.r[chain, draw].values
                lam = pos.lam[chain, draw].values
                phi = pos.phi[chain, draw].values
                b = [r*np.cos(phi)*np.cos(lam), r*np.sin(phi)*np.cos(lam), r*np.sin(lam)]
                decel = pos.decel[chain, draw].values
                t_shift = d['t_g_shift']
                traj = np.outer(a, t_shift-(decel*t_shift**2)/2).T + b

                x, y = xyz_to_pixel(traj, sat_vec, reg=reg)
                ax.scatter(x[:], y[:], c=d['t_g_shift'], alpha=0.2, linewidth=0.5, linestyle='-', s=5, marker='.')

        ax.axis('equal')
        ax.invert_yaxis()
        # limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xy'])
        # ax.set_box_aspect(np.ptp(limits, axis = 1))
        # from matplotlib.ticker import AutoLocator, LinearLocator

        plt.title(f'Approximate {d["detectedBy"]} sensor plane')
        # plt.savefig(f'plots/traj-{d["detectedBy"]}.png', bbox_inches='tight')
        plt.show()


def geocentric_to_ecliptic(ras, decs, obliq=np.radians(23.44)):
    eclons = np.arctan2(np.sin(obliq)*np.sin(decs) + np.sin(ras)*np.cos(decs)*np.cos(obliq), np.cos(ras)*np.cos(decs))
    eclats = np.arcsin(np.cos(obliq)*np.sin(decs) - np.sin(ras)*np.cos(decs)*np.sin(obliq))
    return eclons, eclats


def plot_results(datas, pos, n=1000, truth=None, filters=['max-vel', 'min-vel', 'falling']):
    assert truth is not None
    # truth = {'ra': leo['ra'], 'dec': leo['dec'], 'vel': leo['Vg']}
    # truth = {'ra': 271.9, 'dec': 4.4, 'vel': 60.97}

    # from model import rot_vel_vec
    # rot_vel = rot_vel_vec(datas[0]['latitude'],datas[0]['longitude'])
    # vec = a + rot_vel

    dt = datas[0]['datetime']

    chains, draws = get_sample(pos, n=n)

    # get the posterior samples
    a = extract(pos.a, chains, draws)
    r = extract(pos.r, chains, draws)
    lam = extract(pos.lam, chains, draws)
    phi = extract(pos.phi, chains, draws)
    b = np.array([r*np.cos(phi)*np.cos(lam), r*np.sin(phi)*np.cos(lam), r*np.sin(lam)]).T
    decel = extract(pos.decel, chains, draws)

    # compute ECEF trajectory data
    t_shift = datas[0]['t_g_shift']
    time_deceled = t_shift-(np.outer(decel, t_shift**2))/2
    traj = np.zeros((n, len(t_shift), 3))
    for i in range(n):
        traj[i, :] = (np.outer(a[i, :], time_deceled[i, :])).T + b[i, :]
    traj_start = traj[:, 0, :]
    traj_end = traj[:, -1, :]

    # get a rotation-corrected velocity vector
    vec = rotation_correction(a, traj_start)
    vels = np.linalg.norm(vec, axis=1)

    # compute altitudes
    x = traj_start[:, 0]*u.km
    y = traj_start[:, 1]*u.km
    z = traj_start[:, 2]*u.km
    alts_start = EarthLocation.from_geocentric(x=x, y=y, z=z).height.value
    x = traj_end[:, 0]*u.km
    y = traj_end[:, 1]*u.km
    z = traj_end[:, 2]*u.km
    alts_end = EarthLocation.from_geocentric(x=x, y=y, z=z).height.value
    alts = []
    for i in range(n):
        x = traj[i, :, 0]*u.km
        y = traj[i, :, 1]*u.km
        z = traj[i, :, 2]*u.km
        alts.append(EarthLocation.from_geocentric(x=x, y=y, z=z).height.value)
    alts = np.array(alts)

    # compute ra/dec
    c = SkyCoord(x=-vec[:, 0], y=-vec[:, 1], z=-vec[:, 2], frame='itrs', obstime=Time(dt))
    itrs = c.transform_to(ICRS)
    decs = itrs.dec.value
    ras = itrs.ra.value

    # we haven't applied any filters yet, so all posterior samples are good
    good = np.ones(n, dtype=bool)

    # plt.close()
    # plt.scatter(ras,decs,c=vels, alpha=1,s=3)
    # plt.colorbar(label='Geocentric velocity [km/s]')
    # plt.scatter(truth['ra'],truth['dec'],color='red')
    # plt.xlabel('Right ascension (°)')
    # plt.ylabel('Declination (°)')
    # plt.savefig('plots/radec.png', bbox_inches='tight')
    # plt.show()

    # correct velocity vector for zenith attraction (effect from Earth's gravity pulling it in)
    vec = zenith_attraction(vec, traj_start, dt)
    vels_postgrav = vels
    vels = np.linalg.norm(vec, axis=1)
    # compute corrected ra/dec
    c = SkyCoord(x=-vec[:, 0], y=-vec[:, 1], z=-vec[:, 2], frame='itrs', obstime=Time(dt))
    itrs = c.transform_to(ICRS)
    decs = itrs.dec.value
    ras = itrs.ra.value

    # compute ecliptic longitude and latitude
    eclons, eclats = geocentric_to_ecliptic(np.radians(ras), np.radians(decs))
    truth_eclon, truth_eclat = geocentric_to_ecliptic(np.radians(truth['ra']), np.radians(truth['dec']))

    # put in sun-centered coordinates
    from astropy.coordinates import get_sun
    c = get_sun(Time(dt))
    sollon = c.ra.deg
    eclon_corrected, eclat_corrected, v_h = helio_correction(eclons, eclats, vels, dt)

    truth_eclons = np.array([truth_eclon])
    truth_eclats = np.array([truth_eclat])
    truth_vels = np.array([truth['vel']])
    truth_eclon_c, truth_eclat_c, truth_vel_c = helio_correction(truth_eclons, truth_eclats, truth_vels, dt)

    # apply various filters
    if 'max-vel' in filters:
        good *= v_h <= 42.1
    if 'min-vel' in filters:
        good *= vels_postgrav >= 11
    if 'falling' in filters:
        good *= alts_start > alts_end

    plt.close()
    plt.scatter(ras[good], decs[good], c=vels[good], alpha=1, s=3)
    plt.colorbar(label=r'Geocentric velocity [km/s]')
    plt.scatter(truth['ra'], truth['dec'], color='red')
    plt.xlabel('Right ascension (°)')
    plt.ylabel('Declination (°)')
    # plt.savefig('plots/radec.png', bbox_inches='tight')
    plt.show()

    # plt.close()
    # plt.scatter(np.degrees(eclons),np.degrees(eclats),c=vels, alpha=1,s=3)
    # plt.colorbar(label='Geocentric velocity [km/s]')
    # plt.scatter(np.degrees(truth_eclon),np.degrees(truth_eclat),color='red')
    # plt.xlabel('Ecliptic longitude (°)')
    # plt.ylabel('Ecliptic latitude (°)')
    # plt.show()

    plt.close()
    plt.scatter((np.degrees(eclons[good])-sollon) % 360, np.degrees(eclats[good]), c=vels[good], alpha=1, s=3)
    plt.colorbar(label='Geocentric velocity [km/s]')
    plt.scatter((np.degrees(truth_eclon)-truth['sollon']) % 360, np.degrees(truth_eclat), color='red')
    plt.xlabel('Sun-centered ecliptic longitude (°)')
    plt.ylabel('Ecliptic latitude (°)')
    plt.show()

    plt.close()
    plt.scatter((np.degrees(eclon_corrected[good])-sollon) % 360, np.degrees(eclat_corrected[good]),
                c=vels[good], alpha=1, s=3)
    plt.colorbar(label='Geocentric velocity [km/s]')
    plt.scatter((np.degrees(truth_eclon_c)-truth['sollon']) % 360, np.degrees(truth_eclat_c), color='red')
    plt.xlabel('Sun-centered corrected ecliptic longitude (°)')
    plt.ylabel('Corrected ecliptic latitude (°)')
    plt.show()

    plt.close()
    plt.scatter((np.degrees(eclon_corrected[good])-sollon) % 360, np.degrees(eclat_corrected[good]),
                c=v_h[good], alpha=1, s=3)
    plt.colorbar(label='Heliocentric velocity [km/s]')
    plt.scatter((np.degrees(truth_eclon_c)-truth['sollon']) % 360, np.degrees(truth_eclat_c), color='red')
    plt.xlabel('Sun-centered corrected ecliptic longitude (°)')
    plt.ylabel('Corrected ecliptic latitude (°)')
    plt.show()

    plt.close()
    plt.hist(v_h[good], color='white', bins=50)
    plt.xlabel('Heliocentric velocity (km/s)')
    plt.axvline(42.1, color='blue')
    plt.axvline(truth_vel_c, color='red')
    # plt.savefig('plots/vel.png', bbox_inches='tight')
    plt.show()

    plt.close()
    plt.hist(vels[good], color='white', bins=50)
    plt.xlabel('Velocity (km/s)')
    plt.axvline(float(truth['vel']), color='red')
    # plt.savefig('plots/vel.png', bbox_inches='tight')
    plt.show()

    plt.close()
    plt.scatter(alts_start[good], alts_end[good], c=vels[good], s=3)
    plt.xlabel('Initial altitude (km)')
    plt.ylabel('Final altitude (km)')
    # plt.savefig('plots/alt.png', bbox_inches='tight')
    plt.show()
    plt.close()

    import matplotlib as mpl
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=min(vels), vmax=max(vels))

    for i, a in enumerate(alts[good]):
        t_shift = datas[0]['t_g_shift']
        color = mpl.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(vels[good][i])
        plt.plot(t_shift, a, color=color, linewidth=0.2)
    plt.ylabel('Altitude (km)')
    plt.xlabel('Time')
    # plt.savefig('plots/time-alt.png', bbox_inches='tight')
    plt.show()


def plot_lc(datas, pos, reg, good_chain=None):

    plt.close()
    for d_idx, d in enumerate(datas):
        on = np.sum(d['p_active'], axis=1) > 0
        plt.style.use('dark_background')
        # color = ['blue' if o else 'red' for o in on]
        for i in range(np.sum(on)):
            plt.scatter(d['p_x'][on][i], d['p_y'][on][i], label='events')
        plt.scatter(d['p_x'][~on], d['p_y'][~on], color='red', label='no events')

        n = 100
        chains, draws = get_sample(pos, n=n)
        # if chain != good_chain:
        #     continue

        # get the posterior samples
        a = extract(pos.a, chains, draws)
        r = extract(pos.r, chains, draws)
        lam = extract(pos.lam, chains, draws)
        phi = extract(pos.phi, chains, draws)
        b = np.array([r*np.cos(phi)*np.cos(lam), r*np.sin(phi)*np.cos(lam), r*np.sin(lam)]).T
        decel = extract(pos.decel, chains, draws)

        # compute ECEF trajectory data
        t_shift = datas[0]['t_g_shift']
        time_deceled = t_shift-(np.outer(decel, t_shift**2))/2
        traj = np.zeros((n, len(t_shift), 3))
        for i in range(n):
            traj[i, :] = (np.outer(a[i, :], time_deceled[i, :])).T + b[i, :]

        for i in range(n):
            x, y = xyz_to_pixel(traj[i, :], d['sat_vec'], reg=reg)
            plt.scatter(x[:], y[:], c=d['t_g_shift'], alpha=0.2, linewidth=0.5, linestyle='-', s=5, marker='.')

        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title('Pixels modeled')
        ax = plt.gca()
        ax.axis('equal')
        ax.invert_yaxis()
        plt.show()

        for e in d['p_energy']:
            # nonzero = e != 0
            if np.sum(e) > 0:
                plt.scatter(d['t_g'], e, s=1)

        # if chain != good_chain:
        #     continue

        # t = d['t_g_shift']
        flux = extract(pos.flux, chains, draws)
        # bg_subtract = pos.bg_subtract[chain,draw].values
        k = extract(pos.k, chains, draws)
        w = extract(pos.w, chains, draws)
        c = extract(pos.c, chains, draws)[:, d_idx]

        # error = extract(pos[f'pixel_error{d_idx}'], chains, draws)

        p_x = d['p_x']
        p_y = d['p_y']

        for i in range(n):
            plt.plot(d['t_g'], flux[i, :], alpha=0.3, linewidth=0.2, color='white')

            traj_pixel_x, traj_pixel_y = xyz_to_pixel(traj[i, :], d['sat_vec'], reg=reg)

            # sigma = pos[f'sigma{d_idx}'][chain,draw].values
            # print(flux.shape)

            plt.gca().set_prop_cycle(None)
            for j, e in enumerate(d['p_energy']):

                if np.sum(e) > 0:
                    # if i == 0:
                    #     plt.plot(datas[0]['t_g'], e)
                    pixel_x = p_x[j]
                    pixel_y = p_y[j]

                    pixel_pitch_x = d['p_pitch_x'][j]
                    pixel_pitch_y = d['p_pitch_y'][j]

                    # d_x = np.clip(np.abs(pixel_x-traj_pixel_x)-w, 0, np.inf)
                    # d_y = np.clip(np.abs(pixel_y-traj_pixel_y)-w, 0, np.inf)

                    d_x = np.clip(np.abs(pixel_x-traj_pixel_x)-(w[i]+pixel_pitch_x/2-10), 0, np.inf)
                    d_y = np.clip(np.abs(pixel_y-traj_pixel_y)-(w[i]+pixel_pitch_y/2-10), 0, np.inf)

                    beta = c[i] - k[i]*np.sqrt((d_x**2+d_y**2))
                    scaler = (1/(1+np.exp(-beta)))  # ** power

                    pixel_actual = scaler*flux[i, :]

                    mean = pixel_actual  # - np.cumsum(pixel_actual)*bg_subtract
                    # noise_floor = 2
                    # mean_floored = mean#**2 * (mean < noise_floor)/2 + mean * (mean >= noise_floor)
                    plt.plot(d['t_g'], mean, linewidth=0.2, alpha=0.5)
        # plt.ylim(0,150)
        plt.show()

    # PLOT WITH SIGMA

    # group_timestamps =  np.array([dt.timestamp() for dt in bdf.loc[220,:]['group_time']])
    # plt.plot(group_timestamps, bdf.loc[220,:]['group_energy'],color='blue',linestyle=':')
