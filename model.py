import pymc as pm
import pytensor.tensor as tt
import pytensor
from pytensor import pp
from pytensor.printing import debugprint
from math import pi
from pymc.sampling_jax import sample_numpyro_nuts
from pymc.sampling.jax import sample_blackjax_nuts
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

from astropy.coordinates import ICRS, ITRS, SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u


def rot_vel_vec(lat,lon):
    vel = np.cos(np.radians(lat)) * 0.465
    x = vel * -np.sin(np.radians(lon))
    y = vel * np.cos(np.radians(lon))
    z = 0
    return np.array([x,y,z])


from astropy.coordinates import get_sun
from bolides.astro_utils import get_solarhour
def earth_vel_vec(dt):
    solarhour = get_solarhour(datetime=dt, lon=0)
    deg = (solarhour * 360/24)%360
    #x = 29.78*np.sin(np.radians(deg))
    y = 29.78*np.cos(np.radians(deg))
    z = 0
    
    c = get_sun(Time(dt))
    sollon = c.ra.deg
    sollon_adjust = np.sin(sollon)
    z = -np.cos(np.radians(sollon)) * 29.78 * np.sin(np.radians(23.4))

    x = np.sqrt(29.78**2 - z**2) * np.sin(np.radians(deg))
    y = np.sqrt(29.78**2 - z**2) * np.cos(np.radians(deg))
    return np.array([x,y,z])


# get ECEF coords of satellite
def get_sat_vec(detectedBy):
    from bolides import GOES_E_LON, GOES_W_LON
    sat_lon = None
    
    if detectedBy in ['G17','G18']:
        sat_lon = GOES_W_LON
    elif detectedBy in ['G16']:
        sat_lon = GOES_E_LON
    else:
        print(f'ERROR: {detectedBy} not a valid satellite')

    sat = EarthLocation.from_geodetic(lon=sat_lon*u.degree, lat=0*u.degree, height=35786*u.km)
    
    sat_vec = np.array([v for v in sat.value])
    return sat_vec


# get Earth-Centered Earth-Fixed coordinates from lat lon on GLM ellipsoid.
def glmlatlon_to_ecef(lat,lon):
    e1 = 14
    p1 = 6
    
    eer = 6378.137
    eff = 3.35281e-3
    
    re1 = eer + e1
    rp1 = (1-eff)*eer + p1
    ff1 = (re1-rp1)/re1
    rlat1prime = np.arctan((1-ff1)**2*np.tan(np.radians(lat)))
    
    rmag = re1*(1-ff1)/np.sqrt(1-ff1*(2-ff1)*np.cos(rlat1prime)**2)
    
    x = rmag*np.cos(rlat1prime)*np.cos(np.radians(lon))
    y = rmag*np.cos(rlat1prime)*np.sin(np.radians(lon))
    z = rmag*np.sin(rlat1prime)
    return x, y, z


# project xyz coords to a new ellipsoid.
def renav(xyz, sat_vec, e2, p2, coerce=False):

    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    
    Rx, Ry, Rz = sat_vec

    dx = x - Rx
    dy = y - Ry
    dz = z - Rz
    
    unorm = np.sqrt(dx**2 + dy**2 + dz**2)
    
    udx = dx/unorm
    udy = dy/unorm
    udz = dz/unorm

    eer = 6378.137
    eff = 3.35281e-3
    
    re2 = eer + e2
    rp2 = (1-eff)*eer + p2
    ff2 = (re2-rp2)/re2

    Q1 = udx**2 + udy**2 + (udz**2)/((1-ff2)**2)
    Q2 = 2*(Rx*udx + Ry*udy)
    Q3 = Rx**2 + Ry**2 - re2**2 
    Q4 = Q2**2 - 4*Q1*Q3
    D = (-Q2 - np.sqrt(Q4))/(2*Q1)
    
    x_proj = Rx + D*udx
    y_proj = Ry + D*udy
    z_proj = Rz + D*udz

    if coerce:
        # coerce bad values to 0
        x_proj = (~(tt.isnan(x_proj)|tt.isinf(x_proj)))*x_proj
        y_proj = (~(tt.isnan(y_proj)|tt.isinf(y_proj)))*y_proj
        z_proj = (~(tt.isnan(z_proj)|tt.isinf(z_proj)))*z_proj
    
    return x_proj, y_proj, z_proj #np.vstack((x_proj, y_proj, z_proj)).T

# obtain mappings from azimuth+elevation to pixel coords
def get_regs(nav_filepath, sat_lon):
    nav = pd.read_csv(nav_filepath)
    del nav['meanLightTime']
    del nav['latStd']
    del nav['lonStd']
    del nav['timeStd']
    sat_vec = EarthLocation.from_geodetic(lon=sat_lon*u.degree, lat=0*u.degree, height=35786*u.km)
    sat_vec = np.array([v for v in sat_vec.value])


    nav = nav[~np.isnan(nav['meanLatitude'])]
    pixel_xs = []
    pixel_ys = []
    elevs = []
    azis = []
    lats = []
    lons = []
    lookz = []
    lookxy_norm = []
    z = []
    for i in range(10000):
        row = nav.loc[np.random.choice(nav.index),:]
        pixel_xs.append(row['x'])
        pixel_ys.append(row['y'])

        lats.append(row['meanLatitude'])
        lons.append(row['meanLongitude'])
        xyz = glmlatlon_to_ecef(row['meanLatitude'], row['meanLongitude'])
        
        look = xyz-sat_vec

        elev = np.degrees(np.arctan(np.dot(look,[0,0,1])/np.linalg.norm(look*np.array([1,1,0]))))
        elevs.append(elev)

        lookxy = look * np.array([1,1,0])
        lookxy_proj = -sat_vec * np.dot(lookxy, -sat_vec) / np.linalg.norm(sat_vec)**2
        opposite = lookxy_proj-lookxy
        azi = -np.degrees(np.arctan(np.linalg.norm(opposite)/np.linalg.norm(lookxy_proj)))*np.sign(np.dot(opposite,np.array([1,0,0])))
        azis.append(azi)
        
        xy = np.array([xyz[0],xyz[1],0])

    lat = np.array(lats)
    lon = np.array(lons)
    elevs = np.array(elevs)
    azis = np.array(azis)

    def felev(elev):
        return np.vstack((elev,elev**3)).T
    reg_elev = LinearRegression().fit(felev(elevs),pixel_ys)
    def fazi(azi):
        return np.vstack((azi,azi**3)).T
    reg_azi = LinearRegression().fit(fazi(azis),pixel_xs)

    return {'azi': reg_azi, 'elev': reg_elev}

def xyz_to_pixel(xyz, sat_vec, reg):
    look = xyz-sat_vec

    elev = np.degrees(np.arctan(np.dot(look,[0,0,1])/np.linalg.norm(look*np.array([1,1,0]),axis=1)))

    lookxy = look * np.array([1,1,0])

    lookxy_proj = np.outer(-sat_vec, np.dot(lookxy, -sat_vec) / np.linalg.norm(sat_vec)**2).T
    opposite = lookxy_proj-lookxy
    azi = -np.degrees(np.arctan(np.linalg.norm(opposite,axis=1)/np.linalg.norm(lookxy_proj,axis=1)))*np.sign(np.dot(opposite,np.array([1,0,0])))

    x = reg['azi'].intercept_ + reg['azi'].coef_[0]*azi + reg['azi'].coef_[1]*azi**3 
    y = reg['elev'].intercept_ + reg['elev'].coef_[0]*elev + reg['elev'].coef_[1]*elev**3

    return x,y

def xyz_to_pixel_tensor(xyz, sat_vec, reg):
    look = xyz-sat_vec
    
    xy_select = tt.as_tensor_variable([1,1,0])
    x_select = tt.as_tensor_variable([1,0,0])
    z_select = tt.as_tensor_variable([0,0,1])
    t_sat_vec = tt.as_tensor_variable(sat_vec)
    
    elev = np.arctan(tt.dot(look,z_select)/((look*xy_select).norm(2,axis=1)))*180/pi

    lookxy = look * xy_select

    lookxy_proj = tt.outer(-t_sat_vec, tt.dot(lookxy, -sat_vec) / np.linalg.norm(sat_vec)**2).T
    opposite = lookxy_proj-lookxy
    azi = -np.arctan(opposite.norm(2,axis=1)/lookxy_proj.norm(2,axis=1))*tt.sign(tt.dot(opposite,x_select))*180/pi

    x = reg['azi'].intercept_ + reg['azi'].coef_[0]*azi + reg['azi'].coef_[1]*azi**3 
    y = reg['elev'].intercept_ + reg['elev'].coef_[0]*elev + reg['elev'].coef_[1]*elev**3

    return x,y


def pixelate(t, xyz, energy, sat_vec, reg, corner=True):
    p_xy = []
    for p in np.array(xyz_to_pixel(xyz, sat_vec, reg=reg)).T:
        if any([np.linalg.norm(p-p2)<0.2 for p2 in p_xy]):
            continue
        else:
            p_xy.append(p)
    p_xy = np.array(p_xy)
    p_xy_full = list(p_xy.copy())

    for p in p_xy:
        dx = np.array([1,0])
        dy = np.array([0,1])
        left = p - dx
        right = p + dx
        top = p + dy
        bottom = p - dy
        extra_pixels = [left,right,top,bottom]
        if corner:
            tl = p - dx + dy
            tr = p + dx + dy
            bl = p - dx - dy
            br = p + dx - dy
            extra_pixels = extra_pixels + [tl,tr,bl,br]

        for extra_p in extra_pixels:
            if any([np.linalg.norm(extra_p-p2)<0.3 for p2 in p_xy_full]):
                continue
            else:
                p_xy_full.append(extra_p)
    p_xy = p_xy_full
    p_xy = np.array(p_xy)

    # make timestamps unique
    # for each timestamp, there is a corresponding list of pixels and energies
    t_g = []
    xyz_g = []
    energy_g = []
    for i,time in enumerate(t):
        if all([np.abs(time-tp)>0.001 for tp in t_g]):
            t_g.append(time)
            xyz_g.append([])
            energy_g.append([])
            idx = len(t_g)-1
        else:
            idx = np.argmin([np.abs(time-tp) for tp in t_g])
        xyz_g[idx].append(xyz[i])
        energy_g[idx].append(energy[i])

    t_g = np.array(t_g)

    p_active = np.zeros((len(p_xy),len(t_g)))
    p_energy = np.zeros((len(p_xy),len(t_g)))
    for i,xyzg in enumerate(xyz_g):
        for j,p in enumerate(np.array(xyz_to_pixel(xyzg, sat_vec, reg=reg)).T):
            #print(xyzg)
            #print(np.array(xyz_to_pixel(xyzg, sat_vec, reg=reg)))
            idx = np.argmin([np.linalg.norm(p-p2) for p2 in p_xy])
            p_active[idx,i] = 1
            #print(j,len(xyzg), len(energy_g[i]))
            p_energy[idx,i] = energy_g[i][j]

    #p_active = np.zeros((len(p_xy),len(xyz)))
    #p_energy = np.zeros((len(p_xy),len(xyz)))
    #for i,p in enumerate(np.array(xyz_to_pixel(xyz, sat_vec, reg=reg)).T):
    #    idx = np.argmin([np.linalg.norm(p-p2) for p2 in p_xy])
    #    p_active[idx,i] = 1
    #    p_energy[idx,i] = energy[i]
    
    return t_g, p_xy, p_active, p_energy


def more_pixel_data(p_active):
    p_before = np.array([[0]+[np.sum(p[:(i-1)])>0 for i in range(1,len(p))] for p in p_active])
    p_prev = np.roll(p_active,1,axis=1)
    p_prev[:,0] = 0
    return p_before, p_prev


def get_data(rows, reg):

    datas = []
    for r in rows:
        lats = r.event_lat
        lons = r.event_lon

        # convert to ECEF reference frame cartesian coordinates
        x, y, z = glmlatlon_to_ecef(lats, lons)

        t = np.array([dt.timestamp() for dt in r.event_time])

        # compute satellite longitudes
        from bolides import GOES_E_LON, GOES_W_LON
        if r.detectedBy in ['G17','G18']:
            sat_lon = GOES_W_LON
        elif r.detectedBy in ['G16']:
            sat_lon = GOES_E_LON
        else:
            print(f'ERROR: {d.detectedBy} not a valid satellite')

        # compute satellite location in ECEF reference frame
        sat = EarthLocation.from_geodetic(lon=sat_lon*u.degree, lat=0*u.degree, height=35786*u.km)
        sat_vec = np.array([v for v in sat.value])

        xyz = np.vstack((x,y,z)).T
        energy = r.event_energy
        data = {'t': t, 'xyz': xyz, 'energy': energy, 'sat_vec': sat_vec,
                'latitude': r.latitude, 'longitude': r.longitude,
                'detectedBy': r.detectedBy, 'datetime': r.datetime}


        t_g, p_xy, p_active, p_energy = pixelate(t, xyz, energy, sat_vec, reg=reg)
        p_before, p_prev = more_pixel_data(p_active)
        data['t_g'] = t_g
        data['p_xy'] = p_xy
        data['p_active'] = p_active
        data['p_energy'] = p_energy
        data['p_before'] = p_before
        data['p_prev'] = p_prev
        data['p_x'] = p_xy[:,0]
        data['p_y'] = p_xy[:,1]
        datas.append(data)

    t0 = min([min(d['t']) for d in datas])
    for d in datas:
        d['t_shift'] = d['t']-t0
        d['t_g_shift'] = d['t_g']-t0
    return datas


def run_model(rows, mcmc_kwargs, lc, mcmc_type='hmc'):
    from bolides.constants import GOES_E_LON
    reg = get_regs('G16_nav_LUT.csv', GOES_E_LON)
    print('mapping obtained')
    datas = get_data(rows, reg=reg)
    print('data created')
    if mcmc_type == 'hmc':
        datas, idata = fit(datas, reg=reg, lc=lc, mcmc_kwargs=mcmc_kwargs)
    elif mcmc_type == 'abc':
        datas, idata = fit_abc(datas, reg=reg, lc=lc, mcmc_kwargs=mcmc_kwargs)

    return datas, idata, reg




def fit_abc(datas, reg, lc=False, mcmc_kwargs={}):

    def simulate(rng, a, b, k, w, c, decel=0, flux=None, pixel_error=None, dmax=None, size=None):

        simulated_energies = []

        for i,d in enumerate(datas):
            t_shift = d['t_g_shift']
            traj = np.outer(a, (t_shift-(decel*t_shift**2)/2)).T + b
            sat_vec = d['sat_vec']

            p_active = d['p_active']
            p_energy = d['p_energy']

            p_x = d['p_x']
            p_y = d['p_y']

            traj_pixel_x, traj_pixel_y = xyz_to_pixel_tensor(traj, sat_vec, reg=reg)

            for j in range(len(p_active)):

                pixel_x = p_x[j]
                pixel_y = p_y[j]

                # distances are distances from the edge of the pixel.
                # The edge is rectangular with a distance of w from the centroid.
                # w should end up being roughly 0.5 or less.
                d_x = tt.clip(np.abs(pixel_x-traj_pixel_x)-w, 0, np.inf)
                d_y = tt.clip(np.abs(pixel_y-traj_pixel_y)-w, 0, np.inf)

                beta = c - k*np.sqrt((d_x**2+d_y**2))# + before*p_before[j]
                scaler = (1/(1+np.exp(-beta)))# ** power
                if lc:
                    pixel_actual = scaler*flux
                    mean = pixel_actual
                    noise_floor = 2

                    b = 0
                    e = [mean[0]]

                    M = 1/16
                    for k in range(len(d['t_g'])-1):
                        #bnext = M*mean[k+1] + (1-M)*b[k]

                        #b_{k+1} - b_{k}
                        Delta = M*mean[k+1] + (1-M-1)*b
                        Delta_clip = tt.clip(Delta, -dmax, dmax)
                        bnext = b + Delta_clip
                        e.append(mean[k+1]-bnext)
                        b = bnext
                        # Delta = -M/(1-M) * mean[k]
                    pixel_energy = np.random.normal(loc=e, scale=pixel_error)
                    simulated_energies.append(pixel_energy)
                else:
                    from scipy.stats import bernoulli
                    pixel_active = bernoulli.rvs(scaler)
        return np.array(simulated_energies)
    lats = [d['latitude'] for d in datas]
    lons = [d['longitude'] for d in datas]
    min_lat = min(lats)-4
    max_lat = max(lats)+4

    min_lon = min(lons)-4
    max_lon = max(lons)+4

    with pm.Model() as model:
        vel_dir = pm.Normal('direction', size=3)
        speed   = pm.Uniform('speed', upper=72, lower=-72)
        a = pm.Deterministic('a', speed * vel_dir/vel_dir.norm(2))
        
        r = pm.Uniform('r', lower=6357, upper=6500)
        lam = pm.Uniform('lam', lower=np.radians(min_lat), upper=np.radians(max_lat))
        phi = pm.Uniform('phi', lower=np.radians(min_lon), upper=np.radians(max_lon))
        b = [r*np.cos(phi)*np.cos(lam),r*np.sin(phi)*np.cos(lam),r*np.sin(lam)]

        decel = pm.Uniform('decel', lower=0, upper=0.1) # km/s^2??

        k = pm.Exponential('k', 1)
        w = pm.Exponential('w', 2)
        c = pm.Normal('c', 0, 10, size=len(datas))

        if lc:
            flux = pm.LogNormal('flux', sigma=1, size=len(datas[0]['t_g']))
            pixel_error = pm.Exponential(f'pixel_error', 0.01)
            dmax = pm.Exponential('dmax', 1)
        else:
            flux = None
            pixel_error = None
            dmax = None

        sim = pm.Simulator("sim", simulate,
                params=(a, b, k, w, c[0], decel, flux, pixel_error, dmax),
                epsilon=100, observed=datas[0]['p_energy'])
        idata = pm.sample_smc()
    return datas, idata



def fit(datas, reg, lc=False, mcmc_kwargs={}):
    lats = [d['latitude'] for d in datas]
    lons = [d['longitude'] for d in datas]
    min_lat = min(lats)-5
    max_lat = max(lats)+5

    min_lon = min(lons)-5
    max_lon = max(lons)+5
    with pm.Model() as model:
        #vel_var = pm.Uniform('vel_var', upper=4.160167646, lower=-4.160167646, size=3)
        vel_dir = pm.Normal('direction', size=3)
        speed   = pm.Uniform('speed', upper=72, lower=-72)
        a = pm.Deterministic('a', speed * vel_dir/vel_dir.norm(2))
        
        #a = pm.Deterministic('a', vel_var**3)
        #a = pm.Uniform('a', upper=72, lower=-72, size=3)

        # spherical coords for initial location
        r = pm.Uniform('r', lower=6357, upper=6500)
        lam = pm.Uniform('lam', lower=np.radians(min_lat), upper=np.radians(max_lat))
        phi = pm.Uniform('phi', lower=np.radians(min_lon), upper=np.radians(max_lon))
        b = [r*np.cos(phi)*np.cos(lam),r*np.sin(phi)*np.cos(lam),r*np.sin(lam)]
        decel = pm.Uniform('decel', lower=0, upper=0.1) # km/s^2??

        k = pm.Exponential('k', 1)
        w = pm.Exponential('w', 2)
        c = pm.Normal('c', 0, 10, size=len(datas))
        # power = pm.Exponential('power', 1)

        if lc:
            from itertools import chain
            energies = [d['p_energy'] for d in datas]
            energy_mean = np.mean([np.mean(d['p_energy']) for d in datas])
            for d in datas:
                d['p_energy'] /= energy_mean
            #flux = pm.Exponential('flux', 1) # to replace with time-varying function
            #flux0 = pm.Normal('flux0', 0, 1)
            #flux1 = pm.Normal('flux1', 0, 1)
            #flux2 = pm.Normal('flux2', 0, 1)
            #flux3 = pm.Normal('flux3', 0, 20)

            #bg_subtract = pm.Exponential('bg_subtract',100)

        #before = pm.Normal('before', 0, 10)
        #prev = pm.Normal('prev', 0, 10)
        for i,d in enumerate(datas):
            t_shift = d['t_g_shift']
            traj = tt.outer(a, tt.as_tensor_variable(t_shift-(decel*t_shift**2)/2)).T + b
            sat_vec = d['sat_vec']

            p_active = d['p_active']
            p_energy = d['p_energy']

            #p_before = d['p_before']
            p_prev = d['p_prev']

            p_x = d['p_x']
            p_y = d['p_y']

            traj_pixel_x, traj_pixel_y = xyz_to_pixel_tensor(traj, sat_vec, reg=reg)

            if lc:
                #flux = pm.Deterministic('flux', np.exp(pm.Normal('logflux', mu=0, sigma=1, size=len(d['t_g'])))) 
                flux = pm.LogNormal('flux', sigma=1, size=len(d['t_g']))
                pixel_error = pm.Exponential(f'pixel_error{i}', 0.01)#, size=len(d['t_g']))
                #noise_floor = pm.Exponential(f'noise_floor{i}', 1)
                dmax = pm.Exponential('dmax', 1)
            
            for j in range(len(p_active)):

                pixel_x = p_x[j]
                pixel_y = p_y[j]

                # distances are distances from the edge of the pixel.
                # The edge is rectangular with a distance of w from the centroid.
                # w should end up being roughly 0.5 or less.
                d_x = tt.clip(np.abs(pixel_x-traj_pixel_x)-w, 0, np.inf)
                d_y = tt.clip(np.abs(pixel_y-traj_pixel_y)-w, 0, np.inf)
 
                beta = c[i] - k*np.sqrt((d_x**2+d_y**2))# + before*p_before[j]
                scaler = (1/(1+np.exp(-beta)))# ** power
                if lc:
                    #flux = np.exp(flux0 + t_shift*flux1 + t_shift**2 * flux2 + flux_error) #+ t_shift**3 * flux3)
                    #pixel_actual = tt.clip(pm.Normal(f'pixel_actual{i}_{j}', mu=scaler*flux, sigma=pixel_error), 0, np.inf)
                    pixel_actual = scaler*flux
                    #bg = tt.extra_ops.cumsum(pixel_actual)
                    mean = pixel_actual #- bg_subtract*bg
                    noise_floor = 2
 
                    b = 0
                    e = [mean[0]]

                    M = 1/16
                    for k in range(len(d['t_g'])-1):
                        #bnext = M*mean[k+1] + (1-M)*b[k]

                        #b_{k+1} - b_{k}
                        Delta = M*mean[k+1] + (1-M-1)*b
                        Delta_clip = tt.clip(Delta, -dmax, dmax)
                        bnext = b + Delta_clip
                        e.append(mean[k+1]-bnext)
                        b = bnext
                        # Delta = -M/(1-M) * mean[k]
                    pixel_energy = pm.Normal(f'pixel{i}_{j}', mu=e, sigma=pixel_error, observed=p_energy[j])

                    # mean_floored = mean**2 * (mean < noise_floor)/2 + mean * (mean >= noise_floor)
                    #pixel_energy = pm.Normal(f'pixel{i}_{j}', mu=mean_floored, sigma=pixel_error, observed=p_energy[j])
                    #pixel_energy = pm.Normal(f'pixel{i}_{j}', mu=mean, sigma=pixel_error, observed=p_energy[j])

                    #pixel_energy = pm.Normal(f'pixel{i}_{j}', mu=scaler*flux, sigma=pixel_error, observed=p_energy[j])
                else:
                    pixel_active = pm.Bernoulli(f'pixel{i}_{j}', p=scaler, observed=p_active[j])

                 
            # constrain so that traj goes towards center of Earth
            n = p_active.shape[1]
            select_first = np.zeros(n)
            select_first[0] = 1
            select_last = np.zeros(n)
            select_last[-1] = 1

            traj_norms = traj.norm(2,axis=1)
            norm_diff = tt.dot(traj_norms,select_first) - tt.dot(traj_norms,select_last)
            diff_lt0 = pm.Bernoulli(f'diff{i}', p=norm_diff<0, observed=0)

        #max_vel = 75
        #vel_max = pm.Bernoulli(f'vel_max', p=a.norm(2)<max_vel, observed=1)

        max_vel = 42.1
        earth_vel = earth_vel_vec(datas[0]['datetime'])
        rot_vel = rot_vel_vec(datas[0]['latitude'],datas[0]['longitude'])
        vel_heliocentric = a + tt.as_tensor_variable(earth_vel + rot_vel)
        vel_max = pm.Bernoulli(f'vel_max_escape', p=vel_heliocentric.norm(2)<max_vel, observed=1)

        init_lat = np.radians(np.mean(lats))
        init_lon = np.radians(np.mean(lons))
        init = {'a': [0,0,0], 'r': 6450, 'lam': init_lat, 'phi': init_lon}
        idata = sample_numpyro_nuts(initvals=init, **mcmc_kwargs)
        #idata = sample_blackjax_nuts(initvals=init, **mcmc_kwargs)
        return datas, idata
