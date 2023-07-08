import matplotlib.pyplot as plt
import numpy as np
from model import renav

from astropy.coordinates import ICRS, ITRS, SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u

def get_sample(pos):
    chain = np.random.choice(pos.chain)
    draw = np.random.choice(pos.draw)
    return chain,draw


def plot_prf(datas, pos, n=100, prob=True):
    plt.close()
    
    fig, ax = plt.subplots(1,1)
    x_plot = np.linspace(0,2,100)
    for i in range(n):
        chain, draw = get_sample(pos)
        c = pos.c.values[chain,draw]
        k = pos.k.values[chain,draw]
        w = pos.w.values[chain,draw]

        for j,d in enumerate(datas):
            beta = c[j] - k*np.clip(x_plot-w,0,np.inf)
            scaler = (1/(1+np.exp(-beta)))# ** power
            label = None
            label_active = None
            if i==0:
                label = d['detectedBy']
                label_active = label + ' (activated)'
            plt.plot(x_plot, scaler, color=['lightblue','red'][j], alpha=1,linewidth=0.5,label=label)

    plt.xlabel(r'Distance (pixels) to pixel center on one axis')
    if prob:
        plt.ylabel('Probability of event')
    else:
        plt.ylabel('Proportion of energy transmitted')
    plt.xlim(min(x_plot), max(x_plot))
    plt.axvline(0.5,color='white',label='Pixel boundary',linewidth=1,linestyle='--')
    plt.legend(frameon=False)
    plt.ylim(0,1)
    return fig, ax


def plot_prf_3d(datas, pos, prob=True):
    plt.close()
    chain, draw = get_sample(pos)
    c = pos.c.values[chain,draw]
    k = pos.k.values[chain,draw]
    w = pos.w.values[chain,draw]

    x_plot = np.linspace(-0.8,0.8,50)
    y_plot = np.linspace(-0.8,0.8,50)
    xx, yy = np.meshgrid(x_plot,y_plot)

    d_x = np.clip(np.abs(xx)-w, 0, np.inf)**2
    d_y = np.clip(np.abs(yy)-w, 0, np.inf)**2

    beta = c[0] - k*np.sqrt((d_x+d_y))
    p = 1/(1+np.exp(-beta))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_proj_type('ortho')
    ax.set_zlim(0,1)
    fig.tight_layout()
    ax.plot_surface(xx, yy, p, vmin=np.min(p), cmap='inferno', rstride=1, cstride=1,alpha=1)
    zs = [0]
    for i,z in enumerate(zs):
        plt.plot([-0.5,0.5,0.5,-0.5,-0.5],[-0.5,-0.5,0.5,0.5,-0.5],[z,z,z,z,z],
                 color='white',zorder=5,label=('Pixel boundary' if i==0 else None),linestyle='--')

    import matplotlib.cm as cm
    m = cm.ScalarMappable(cmap=cm.inferno)
    m.set_array(p)
    plt.colorbar(m,label='Probability of event',ax=ax)
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

    fig = plt.figure(figsize=(6,6),facecolor='black')
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

    #ax.xaxis.pane.fill = False
    #ax.yaxis.pane.fill = False
    #ax.zaxis.pane.fill = False

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    for i in range(100):
        chain, draw = get_sample(pos)
        a = pos.a[chain,draw,:].values
        r = pos.r[chain,draw].values
        lam = pos.lam[chain,draw].values
        phi = pos.phi[chain,draw].values
        b = [r*np.cos(phi)*np.cos(lam),r*np.sin(phi)*np.cos(lam),r*np.sin(lam)]
        decel = pos.decel[chain,draw].values

        if chain != 2:
            continue
        for j,d in enumerate(datas):
            t_shift = d['t_g_shift']
            traj = np.outer(a, t_shift-(decel*t_shift**2)/2).T + b
            sat_vec = d['sat_vec']
            x,y,z = renav(traj, sat_vec, 14, 6)
            ax.plot(x[:],y[:],z[:],color='xkcd:salmon',alpha=0.1,linewidth=0.5)

            ax.plot(traj[:,0],traj[:,1],traj[:,2],color='pink',alpha=1,linewidth=0.5)
    for i,d in enumerate(datas):
        xyz = d['xyz']
        ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],alpha=0.5,c=d['t_shift'],marker='.',zorder=10)

    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis = 1))
    from matplotlib.ticker import AutoLocator, LinearLocator
    ax.zaxis.set_major_locator(LinearLocator(3))

    plt.title('Trajectories in ECEF',color='white')
    return fig, ax


def plot_sensorplane(datas, pos=None, reg=None):
    from model import xyz_to_pixel
    for j,d in enumerate(datas):
        plt.close()
        plt.rcParams['grid.color'] = "darkgray"
        plt.rcParams['grid.linewidth'] = 0.5
        plt.rcParams['axes.linewidth'] = 0.5
        fig = plt.figure(figsize=(2,2),facecolor='black')
        fig.tight_layout()
        ax = fig.add_subplot()
        ax.set_facecolor('black')
        ax.tick_params(colors='gray')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        
        sat_vec = d['sat_vec']

        
        plt.style.use('dark_background')

        on = np.sum(d['p_active'],axis=1)>0
        plt.scatter(d['p_x'][~on],d['p_y'][~on],color='red',label='no events',s=5)
        #plt.scatter(d['p_x'][on],d['p_y'][on],color='blue',label='events',s=5)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')

        x, y = xyz_to_pixel(d['xyz'], sat_vec, reg=reg)
        ax.scatter(x,y,alpha=0.5,c=d['t_shift'],marker='.',zorder=10,edgecolor='white',linewidth=0.2,s=80)
        
        if pos is not None:
            for i in range(100):
                chain, draw = get_sample(pos)
                a = pos.a[chain,draw,:].values
                r = pos.r[chain,draw].values
                lam = pos.lam[chain,draw].values
                phi = pos.phi[chain,draw].values
                b = [r*np.cos(phi)*np.cos(lam),r*np.sin(phi)*np.cos(lam),r*np.sin(lam)]
                decel = pos.decel[chain,draw].values
                t_shift = d['t_g_shift']
                traj = np.outer(a, t_shift-(decel*t_shift**2)/2).T + b
                
                x, y = xyz_to_pixel(traj, sat_vec, reg=reg)
                ax.scatter(x[:],y[:],c=d['t_g_shift'],alpha=0.2,linewidth=0.5,linestyle='-',s=5,marker='.')
            
        ax.axis('equal')
        ax.invert_yaxis()
        #limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xy'])
        #ax.set_box_aspect(np.ptp(limits, axis = 1))
        from matplotlib.ticker import AutoLocator, LinearLocator
            
        plt.title(f'Approximate {d["detectedBy"]} sensor plane')
        #plt.savefig(f'plots/traj-{d["detectedBy"]}.png', bbox_inches='tight')
        plt.show()


def plot_results(datas, pos, n=1000, truth=None):
    assert truth is not None
    #truth = {'ra': leo['ra'], 'dec': leo['dec'], 'vel': leo['Vg']}
    #truth = {'ra': 271.9, 'dec': 4.4, 'vel': 60.97}

    from model import rot_vel_vec
    ras = []
    decs = []
    vels = []
    alts_start = []
    alts_end = []
    alts = []
    rot_vel = rot_vel_vec(datas[0]['latitude'],datas[0]['longitude'])
    for i in range(n):
        chain, draw = get_sample(pos)
        a = pos.a[chain,draw,:]
        vec = pos.a[chain,draw,:] + rot_vel
        vels.append(np.linalg.norm(vec))

        r = pos.r[chain,draw].values
        lam = pos.lam[chain,draw].values
        phi = pos.phi[chain,draw].values
        b = [r*np.cos(phi)*np.cos(lam),r*np.sin(phi)*np.cos(lam),r*np.sin(lam)]
        decel = pos.decel[chain,draw].values
        t_shift = datas[0]['t_g_shift']
        traj = np.outer(a, t_shift-(decel*t_shift**2)/2).T + b

        traj1_start = traj[0,:]
        traj1_end = traj[-1,:]

        alts_start.append(EarthLocation.from_geocentric(x=traj1_start[0]*u.km, y=traj1_start[1]*u.km, z=traj1_start[2]*u.km).height.value)
        alts_end.append(EarthLocation.from_geocentric(x=traj1_end[0]*u.km, y=traj1_end[1]*u.km, z=traj1_end[2]*u.km).height.value)

        alts.append(EarthLocation.from_geocentric(x=traj[:,0]*u.km, y=traj[:,1]*u.km, z=traj[:,2]*u.km).height.value)
        
        c = SkyCoord(x=-vec[0], y=-vec[1], z=-vec[2], frame='itrs', obstime=Time(datas[0]['datetime']))
        itrs = c.transform_to(ICRS)
        ras.append(itrs.ra.value)
        decs.append(itrs.dec.value)
    plt.close()
    plt.scatter(ras,decs,c=vels, alpha=1,s=3)
    plt.colorbar(label='Geocentric velocity [km/s]')
    plt.scatter(truth['ra'],truth['dec'],color='red')
    plt.xlabel('Right ascension (°)')
    plt.ylabel('Declination (°)')
    #plt.savefig('plots/radec.png', bbox_inches='tight')
    plt.show()
    plt.close()
    plt.hist(vels, color='white',bins=50)
    plt.xlabel('Velocity (km/s)')
    plt.axvline(float(truth['vel']),color='red')
    #plt.savefig('plots/vel.png', bbox_inches='tight')
    plt.show()
    plt.close()
    plt.scatter(alts_start, alts_end, c=vels, s=3)
    plt.xlabel('Initial altitude (km)')
    plt.ylabel('Final altitude (km)')
    #plt.savefig('plots/alt.png', bbox_inches='tight')
    plt.show()
    plt.close()

    import matplotlib as mpl
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=min(vels), vmax=max(vels))

    for i,a in enumerate(alts):
        t_shift = datas[0]['t_g_shift']
        color = mpl.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(vels[i])
        plt.plot(t_shift, a, color=color, linewidth=0.2)
    plt.ylabel('Altitude (km)')
    plt.xlabel('Time')
    #plt.savefig('plots/time-alt.png', bbox_inches='tight')
    plt.show()
