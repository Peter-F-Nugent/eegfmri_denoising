import numpy as np
import matplotlib.pyplot as plt
import mne
import pypulseq as pp

class SphereModel:
    def __init__(self, info=None, bore_offset=0.0, n_wire_points=50):
        self.bore_offset = bore_offset
        self.n_wire_points = n_wire_points
        print("Creating sphere model...")
        self.info = info
        mne_sphere = mne.make_sphere_model("auto", "auto", info)
        self.mne_sphere = mne_sphere
        mne_center = np.array(mne_sphere["r0"])
        self.radius = max(layer["rad"] for layer in mne_sphere["layers"])

        print("Storing electrode positions...")
        eeg_coords_orig = []
        ch_names = []
        for ch in info["chs"]:
            if ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH:
                loc = ch["loc"][:3]
                if not np.allclose(loc, 0):
                    eeg_coords_orig.append(loc)
                    ch_names.append(ch["ch_name"])

        self.ch_names = ch_names
        self.eeg_coords_orig = np.array(eeg_coords_orig)

        # project electrodes onto sphere using mne_center for geometry
        self.vecs = self.eeg_coords_orig - mne_center
        self.vecs /= np.linalg.norm(self.vecs, axis=1, keepdims=True)

        # zero — relative to sphere centre at origin
        self.center = np.array([0.0, 0.0, 0.0])
        self.top    = np.array([0.0, 0.0, self.radius])  # top of head = MNE z
        self.eeg_coords_proj = self.center + self.vecs * self.radius

        # build wires
        self.wires = []
        for ep in self.eeg_coords_proj:
            arc = SphereModel.great_circle_arc(
                self.top, ep, self.center, self.radius, n=self.n_wire_points
            )
            self.wires.append(arc)

        # shift along bore axis (MNE y = head-foot when lying supine)
        offset = np.array([0.0, bore_offset, 0.0])
        self.center          = self.center + offset
        self.top             = self.top    + offset
        self.eeg_coords_proj = self.eeg_coords_proj + offset
        self.wires           = [wire + offset for wire in self.wires]

    def __repr__(self):
        return (f"SphereModel | {len(self.ch_names)} channels | "
            f"radius={self.radius*100:.1f}cm | "
            f"bore_offset={self.bore_offset*100:.1f}cm")

    @staticmethod
    def great_circle_arc(p1, p2, center, radius, n=50):
        p1 = p1 - center
        p2 = p2 - center
        p1 = p1 / np.linalg.norm(p1)
        p2 = p2 / np.linalg.norm(p2)
        omega = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
        if omega < 1e-6:
            return np.tile(center + p1 * radius, (n, 1))
        t = np.linspace(0, 1, n)
        sin_omega = np.sin(omega)
        arc = (
            np.sin((1 - t) * omega)[:, None] * p1 + np.sin(t * omega)[:, None] * p2
        ) / sin_omega
        return center + arc * radius

    def plot(self):
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        x = self.center[0] + self.radius * np.outer(np.cos(u), np.sin(v))
        y = self.center[1] + self.radius * np.outer(np.sin(u), np.sin(v))
        z = self.center[2] + self.radius * np.outer(np.ones_like(u), np.cos(v))
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            self.eeg_coords_proj[:, 0],
            self.eeg_coords_proj[:, 1],
            self.eeg_coords_proj[:, 2],
            s=20,
        )
        ax.plot_surface(x, y, z, alpha=0.1)
        ax.set_box_aspect([1, 1, 1])
        ax.scatter(*self.top, s=50)
        for wire in self.wires:
            ax.plot(wire[:, 0], wire[:, 1], wire[:, 2], linewidth=1)
        ax.set_title("EEG sphere model with geodesic wire paths")
        plt.show()


class PulseSequence:
    """
    Defines gradient waveforms and timing for a simulated pulse sequence. 
    This is used to simulate the gradient artefact and BCG artefact in the EEG data. 
    """
    def __init__(self, sfreq=100000):
        self.sfreq = sfreq

    def __repr__(self) -> str:
        duration = len(self.Gz) / self.sfreq
        return (f"PulseSequence | sfreq={self.sfreq}Hz | "
                f"duration={duration:.2f}s | "
                f"n_samples={len(self.Gz)}")
    
    def load_seq(self, pulse_seq_file = None, GAMMA = 42.576e6):
        seq = pp.Sequence()
        seq.read(pulse_seq_file)
        gw_pp = seq.get_gradients()   # list of 3 PPoly objects [Gx, Gy, Gz]
        duration = seq.duration()[0]
        t = np.arange(0, duration, 1/self.sfreq)

        self.Gx = np.nan_to_num(gw_pp[0](t)) / GAMMA  # Hz/m → T/m
        self.Gy = np.nan_to_num(gw_pp[1](t)) / GAMMA
        self.Gz = np.nan_to_num(gw_pp[2](t)) / GAMMA

    def load_standard_epi(self):
        print('Assigning standard EPI sequence...')
        self.load_seq("../notebooks/epi_pypulseq.seq")

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.Gx, label='Gx')
        plt.plot(self.Gy, label='Gy')
        plt.plot(self.Gz, label='Gz')
        plt.title('Gradient Waveforms')
        plt.xlabel('Sample Index')
        plt.ylabel('Gradient Amplitude (T/m)')
        plt.legend()
        plt.show()

    ### TODO, consider merging this into the artifact simulation...im not convinced this needs to be seperate when we already have seq objects from pypulseseq

       
class GradientArtefact:
    def __init__(self, sphere: SphereModel, pulse_seq: PulseSequence, n_slices=32, tr=2, n_volumes=60):
        self.sphere    = sphere
        self.Gx        = pulse_seq.Gx
        self.Gy        = pulse_seq.Gy
        self.Gz        = pulse_seq.Gz
        self.sfreq     = pulse_seq.sfreq
        self.n_slices  = n_slices
        self.n_volumes = n_volumes
        self.V_t       = None

    def __repr__(self) -> str:
        return (f"GradientArtefact | sfreq={self.sfreq}Hz | "
                f"n_volumes={self.n_volumes} | "
                f"total_duration={len(self.Gz)*self.n_slices*self.n_volumes/self.sfreq:.2f}s")

    # ------------------------------------------------------------------ #
    #  A field equations — for plot_field() only                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def A_z(x, y, z, G):
        """Vector potential for z-gradient. Eq A2."""
        Ax = -0.5 * G * y * z
        Ay =  0.5 * G * x * z
        Az =  np.zeros_like(x)
        return np.column_stack([Ax, Ay, Az])

    @staticmethod
    def A_x(x, y, z, G):
        """Vector potential for x-gradient. Eq A4."""
        Ax = -0.5  * G * x * y
        Ay =  0.25 * G * (x**2 - y**2)
        Az =  G * y * z
        return np.column_stack([Ax, Ay, Az])

    @staticmethod
    def A_y(x, y, z, G, Gx=0):
        """Vector potential for y-gradient. Eq A6.
        Note: k component is cross term with Gx — defaults to 0 for plotting."""
        Ax =  0.25 * G * (x**2 - y**2)
        Ay =  0.5  * G * x * y
        Az = -Gx   * x * z
        return np.column_stack([Ax, Ay, Az])

    # ------------------------------------------------------------------ #
    #  Analytic voltage equations — Yan et al. 2009                       #
    #  Converted to MNE coordinate system:                                #
    #    Paper: x=left, -y=anterior, z=up                                 #
    #    MNE:   x=right, +y=anterior, z=up                                #
    #  Transform: negate cos(beta+phi) and sin(beta+phi)                  #
    #  alpha sign: paper alpha=-15° = MNE alpha=+15°                      #
    #  bore_offset enters as z0 (paper's z = bore = MNE y)               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_Vz(r, theta=0.0, phi=0.0, dG_dt=1.0, x0=0.0, y0=0.0, z0=0.0,
               alpha=np.radians(-15), beta=0.0):
        """
        Full analytic artefact voltage for longitudinal (z) gradient. Eq 6.
        MNE coordinate system. Default alpha=15° for realistic head tilt.
        """
        cot_alpha = (np.cos(alpha) / np.sin(alpha)
                     if np.abs(np.sin(alpha)) > 1e-10 else 0.0)
        cbp = -np.cos(beta + phi)
        sbp = -np.sin(beta + phi)
        V = 0.25 * dG_dt * r**2 * (
            2 * r * cbp * np.sin(alpha) * (
                (1 - np.cos(theta)) * np.sin(alpha) * sbp
                + np.cos(alpha) * np.sin(theta)
            )
            + theta * (
                cbp * np.sin(alpha) * (2*z0 - y0*cot_alpha)
                + x0 * sbp
            )
        )
        return V

    @staticmethod
    def get_Vx(r, theta=0.0, phi=0.0, dG_dt=1.0, x0=0.0, y0=0.0, z0=0.0,
               alpha=np.radians(-15), beta=0.0):
        """
        Full analytic artefact voltage for transverse x-gradient. Eq 7.
        MNE coordinate system.
        """
        cbp = -np.cos(beta + phi)
        sbp = -np.sin(beta + phi)
        V = (1/6) * dG_dt * r**2 * (
            2 * r * (
                (1 - np.cos(theta)) * np.sin(alpha) * cbp**2
                - sbp * np.cos(alpha) * np.sin(theta)
            )
            + 3 * theta * (
                x0 * cbp * np.sin(alpha)
                - z0 * sbp
            )
        )
        return V

    @staticmethod
    def get_Vy(r, theta=0.0, phi=0.0, dG_dt=1.0, x0=0.0, y0=0.0, z0=0.0,
               alpha=np.radians(-15), beta=0.0):
        """
        Full analytic artefact voltage for transverse y-gradient.
        Vy = Vx with beta -> beta + pi/2. MNE coordinate system.
        """
        return GradientArtefact.get_Vx(
            r, theta, phi, dG_dt, x0, y0, z0,
            alpha=alpha, beta=beta + np.pi/2
        )

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    def plot_field(self, axis='z', G=1.0, n_grid=8, show_sphere=True):
        """Plot sphere and A field vectors in 3D — for visual inspection."""
        fig = plt.figure(figsize=(9, 9))
        ax  = fig.add_subplot(111, projection='3d')
        cx, cy, cz = self.sphere.center
        r = self.sphere.radius

        if show_sphere:
            u = np.linspace(0, 2*np.pi, 40)
            v = np.linspace(0, np.pi, 40)
            sx = cx + r * np.outer(np.cos(u), np.sin(v))
            sy = cy + r * np.outer(np.sin(u), np.sin(v))
            sz = cz + r * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(sx, sy, sz, alpha=1.0, color='lightblue', zorder=1)

        for wire in self.sphere.wires:
            ax.plot(wire[:,0], wire[:,1], wire[:,2],
                    linewidth=0.8, alpha=0.6, color='steelblue', zorder=2)

        ep = self.sphere.eeg_coords_proj
        ax.scatter(ep[:,0], ep[:,1], ep[:,2], s=15, color='red', zorder=3)

        coords  = np.linspace(-r*1.5, r*1.5, n_grid)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        x_flat, y_flat, z_flat = X.flatten(), Y.flatten(), Z.flatten()

        A_func   = {'x': self.A_x, 'y': self.A_y, 'z': self.A_z}[axis]
        A_vals   = A_func(x_flat, y_flat, z_flat, G)
        mag      = np.linalg.norm(A_vals, axis=1)
        mag_norm = mag / (mag.max() + 1e-10)
        colors   = plt.cm.viridis(mag_norm)

        ax.quiver(x_flat, y_flat, z_flat,
                  A_vals[:,0], A_vals[:,1], A_vals[:,2],
                  length=r*0.12, normalize=True,
                  color=colors, alpha=0.8, zorder=0)

        ax.set_title(f'A_{axis} field')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
        ax.set_box_aspect([1,1,1])
        plt.show()

    # ------------------------------------------------------------------ #
    #  Simulation                                                          #
    # ------------------------------------------------------------------ #

    def simulate(self, ref_channel_name='Cz', eeg_sfreq=5000,
                 alpha=np.radians(15), beta=0.0):
        """
        Simulate gradient artefact for all channels.

        Parameters
        ----------
        ref_channel_name : str   — reference electrode name
        eeg_sfreq        : float — target EEG sampling frequency (Hz)
        alpha            : float — pitch angle in radians (default 15°)
        beta             : float — yaw angle in radians (default 0°)

        Notes
        -----
        bore_offset (MNE y) enters equations as z0 (paper's bore axis).
        phi uses standard MNE convention arctan2(y, x) — coordinate
        transform is baked into get_Vx/Vy/Vz equations.
        """
        from scipy.signal import resample

        r      = self.sphere.radius
        center = self.sphere.center

        # bore offset = paper's z0 (their z = bore axis = MNE y)
        x0 = 0.0
        y0 = 0.0
        z0 = self.sphere.bore_offset

        dt     = 1 / self.sfreq
        dGx_dt = np.diff(self.Gx, prepend=self.Gx[0]) / dt
        dGy_dt = np.diff(self.Gy, prepend=self.Gy[0]) / dt
        dGz_dt = np.diff(self.Gz, prepend=self.Gz[0]) / dt

        ref_idx   = self.sphere.ch_names.index(ref_channel_name)
        ref_ep    = self.sphere.eeg_coords_proj[ref_idx] - center
        ref_phi   = np.arctan2(ref_ep[1], ref_ep[0])     # standard MNE
        ref_theta = np.arccos(np.clip(ref_ep[2] / r, -1, 1))

        n_ch      = len(self.sphere.ch_names)
        n_samples = len(dGx_dt)

        V_one       = np.zeros((n_ch, n_samples))
        self.Vx_all = np.zeros((n_ch, n_samples))
        self.Vy_all = np.zeros((n_ch, n_samples))
        self.Vz_all = np.zeros((n_ch, n_samples))

        for i, ep in enumerate(self.sphere.eeg_coords_proj):
            ep_rel = ep - center
            phi    = np.arctan2(ep_rel[1], ep_rel[0])    # standard MNE
            theta  = np.arccos(np.clip(ep_rel[2] / r, -1, 1))

            self.Vx_all[i] = (self.get_Vx(r, theta, phi, dGx_dt, x0, y0, z0, alpha, beta) -
                              self.get_Vx(r, ref_theta, ref_phi, dGx_dt, x0, y0, z0, alpha, beta))
            self.Vy_all[i] = (self.get_Vy(r, theta, phi, dGy_dt, x0, y0, z0, alpha, beta) -
                              self.get_Vy(r, ref_theta, ref_phi, dGy_dt, x0, y0, z0, alpha, beta))
            self.Vz_all[i] = (self.get_Vz(r, theta, phi, dGz_dt, x0, y0, z0, alpha, beta) -
                              self.get_Vz(r, ref_theta, ref_phi, dGz_dt, x0, y0, z0, alpha, beta))

            V_one[i] = (self.Vx_all[i] + self.Vy_all[i] + self.Vz_all[i]) * 1e6

        self.Vx_all *= 1e6
        self.Vy_all *= 1e6
        self.Vz_all *= 1e6

        V_full = np.tile(V_one, self.n_slices * self.n_volumes)
        n_eeg_samples = int(V_full.shape[1] * eeg_sfreq / self.sfreq)
        self.V_t      = resample(V_full, n_eeg_samples, axis=1)
        self.eeg_sfreq = eeg_sfreq

        return self


    '''
    def _compute_scaling(self, axis):
        A_func = {'x': self.A_x, 'y': self.A_y, 'z': self.A_z}[axis]
        r = self.sphere.radius
        center = self.sphere.center

        # find reference electrode index — Cz
        ref_idx = self.sphere.ch_names.index('Cz')
        ref_ep  = self.sphere.eeg_coords_proj[ref_idx]

        # compute reference scaling once
        ref_voltage = self._integrate_meridian(A_func, ref_ep, r, center)

        voltages = []
        for ep in self.sphere.eeg_coords_proj:
            # term 1 — active electrode wire
            v_active = self._integrate_meridian(A_func, ep, r, center)
            # term 2 — subtract reference wire
            voltage  = v_active - ref_voltage
            voltages.append(voltage)

        return np.array(voltages) * 1e6

    def _integrate_meridian(self, A_func, ep, r, center):
        """Eq 5 — integrate A_theta along meridian from north pole to electrode."""
        ep_rel  = ep - center
        phi     = np.arctan2(ep_rel[1], ep_rel[0])
        theta_L = np.arccos(np.clip(ep_rel[2] / r, -1, 1))

        n      = 200
        thetas = np.linspace(0, theta_L, n)

        xs = center[0] + r * np.sin(thetas) * np.cos(phi)
        ys = center[1] + r * np.sin(thetas) * np.sin(phi)
        zs = center[2] + r * np.cos(thetas)

        A_vals   = A_func(xs, ys, zs, G=1.0)
        Ax, Ay, Az = A_vals[:,0], A_vals[:,1], A_vals[:,2]

        integrand = (
            (Ax * np.cos(phi) + Ay * np.sin(phi)) * np.cos(thetas)
            - Az * np.sin(thetas)
        )

        return -r * np.trapz(integrand, thetas)
    '''

    '''
    def simulate(self, eeg_sfreq=5000):
        
        from scipy.signal import resample

        # 1. compute scaling factors once per axis
        scaling_x = self._compute_scaling('x')
        scaling_y = self._compute_scaling('y')
        scaling_z = self._compute_scaling('z')

        # 2. dG/dt for one slice
        dt = 1 / self.sfreq
        dGx_dt = np.diff(self.Gx, prepend=self.Gx[0]) / dt
        dGy_dt = np.diff(self.Gy, prepend=self.Gy[0]) / dt
        dGz_dt = np.diff(self.Gz, prepend=self.Gz[0]) / dt

        # 3. V for one slice at full gradient resolution
        V_one = (scaling_x[:, None] * dGx_dt[None, :] +
            scaling_y[:, None] * dGy_dt[None, :] +
            scaling_z[:, None] * dGz_dt[None, :])

        # 4. tile for full scan
        V_full = np.tile(V_one, self.n_slices * self.n_volumes)

        # 5. downsample to EEG sfreq
        n_eeg_samples = int(V_full.shape[1] * eeg_sfreq / self.sfreq)
        self.V_t = resample(V_full, n_eeg_samples, axis=1)
        self.eeg_sfreq = eeg_sfreq

        return self
    '''
    
### Simlate EEG
"""
    def simulate_eeg(self, alpha=True, blinks=True):
        self.src = mne.setup_volume_source_space(
            pos=15.0,
            sphere=self.sphere,
            sphere_units="m",
            verbose=False,
        )
        self.fwd = mne.make_forward_solution(
            self.info,
            trans=None,
            src=self.src,
            bem=self.sphere,
            eeg=True,
            meg=False,
            verbose=False,
        )

        n_dipoles = 1
        rng = np.random.RandomState(0)

        def data_fun(times):
            return 25e-9 * np.sin(2 * np.pi * 10 * times)

        sfreq = self.info["sfreq"]
        duration = 120.0
        n_samp = int(sfreq * duration)
        times = np.arange(n_samp) / sfreq

        stc = simulate_sparse_stc(
            self.src, n_dipoles=n_dipoles, times=times, data_fun=data_fun, random_state=rng
        )

        self.info["dev_head_t"] = self.fwd["info"]["dev_head_t"]
        raw_sim = simulate_raw(self.info, [stc] * 10, forward=self.fwd, verbose=True)
        cov = make_ad_hoc_cov(raw_sim.info)
        add_noise(raw_sim, cov, iir_filter=[0.2, -0.2, 0.04], random_state=rng)
        add_eog(raw_sim, random_state=rng)
        raw_sim.plot()



    def simulate_gradients(self, TR, echo_time, slices=32, multiband=2):
        print()

    def simulate_bcg(self, TR, echo_time, slices=32, multiband=2):
        print()
"""
