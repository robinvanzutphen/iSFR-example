%% ========================================================================
%  Minimal example — iSFR Monte Carlo (MCX) + SIA reflectance
%
%  Goal:
%    Provide a compact coding example that captures the core
%    logic used in the manuscript + supplement:
%
%      Phase 1  (MCX): Run a single absorption-free Monte Carlo simulation
%                      for one (NA, mu_s', phase function) configuration.
%
%      Phase 2  (Post): Reweight detected photons for different mu_a via
%                       Beer–Lambert, compute R(rho) in annuli, and evaluate
%                       the Single-Integral Approximation (SIA) to obtain
%                       R_tot(mu_a, d_f)
%
%  Important notes:
%    - This is an *example*, not the full pipeline:
%        * no parameter sweeps, no batching/continuation, no COV stopping
%        * no file IO, no error handling / fail-safes / diagnostics
%        * no advanced quadrature techniques or diffusion-theory
%        implementations
%    - MCX must be installed and on the MATLAB path (mcxlab callable).
%
% ========================================================================

clear; clc; close all;                       % start clean

%% ----------------------------- User inputs ------------------------------
% Refractive indices
n_tissue = 1.35;                             % refractive index of tissue medium
n_ext    = 1.00;                             % refractive index of external medium (air)

% Reduced scattering (mu_s') for this single example case.
mu_sp = 5;                                   % [1/mm]

% Numerical aperture, gets converted to acceptance angle in tissue.
NA = 0.22;                                   % numerical aperture 
thetaMax = asin(NA / n_tissue);              % acceptance half-angle inside tissue [rad]

% Phase function example: Henyey–Greenstein with anisotropy g.
g_HG = 0.8;                                  % HG anisotropy parameter 

% MCX photon count
nPhoton = 1e5;                               % photons launched in this demo run

% Absorption list 
mu_a_list = linspace(0.1, 1.0, 25);          % [1/mm] absorption values to evaluate

% Disk diameter(s) 
d_f_list  = [0.5, 1.0];                      % [mm] disk diameter(s)

% Number of radial edges for annular binning of R(rho); bins = edges-1.
nRhoEdges = 5001;                            % fine edges for accurate integration

%% ========================================================================
%  Phase-function moments and ICDF table for MCX scattering
% ========================================================================

% Discretize mu = cos(theta) on a dense grid for numerical quadrature.
Nmu = 1e5;                                   % number of mu samples (dense)
mu  = linspace(-1, 1, Nmu).';                % mu grid as column vector in [-1,1]

% Define HG phase function p(mu) 
pHG_mu = @(mu_,g) (1 - g.^2) ./ (4*pi .* (1 + g.^2 - 2*g.*mu_).^(3/2));

% Evaluate HG on the mu grid.
p_mu = pHG_mu(mu, g_HG);                     % unnormalized as PDF over mu

% Normalize p_mu as a PDF over mu so ∫_{-1}^{1} p(mu) dmu = 1.
p_mu = p_mu ./ trapz(mu, p_mu);             

% Compute g1 via the first-order moment: g1 = ∫ mu * p(mu) dmu.
g1 = trapz(mu, mu .* p_mu);                  % first moment (anisotropy)

% Compute g2 via the second-order moment: g2 = ∫ mu^2 * p(mu) dmu.
g2 = trapz(mu, (mu.^2) .* p_mu);             % second moment 

% Convert reduced scattering mu_s' to scattering mu_s using g1:
%   mu_s' = mu_s*(1-g1)  ->  mu_s = mu_s'/(1-g1).
mu_s = mu_sp / max(eps, (1 - g1));           

% Build an inverse-CDF lookup table in mu for MCX scattering sampling.
icdf_mu = make_icdf(mu, p_mu);               % ICDF values in mu = cos(theta)

%% ========================================================================
%  Minimal MCX configuration (homogeneous single-voxel volume)
% ========================================================================

% We do not need spatial fluence, only detected photon statistics.
% Therefore, we can use a minimal homogeneous volume and rely on unit scaling.
volSize_mm = 100;                            % physical edge size [mm] (arbitrary, large)
volN       = 2;                              % single voxel in each dimension
voxSize_mm = volSize_mm / volN;              % voxel size in mm

% Initialize configuration struct required by mcxlabcl.
cfg = struct();                              % MCX configuration container

% Define voxel-to-mm scaling so MCX outputs can be converted to mm.
cfg.unitinmm = voxSize_mm;                   % size of one voxel [mm]

% Define a homogeneous volume with one material label.
cfg.vol = ones(volN, volN, volN, 'uint8');   % uniform tissue label

% Enable Fresnel reflections at boundaries (index mismatch considered).
cfg.isreflect  = 1;                          % include refractive index mismatch effects

% Disable specular reflection at the source boundary.
cfg.isspecular = 0;                          

%% ------------------------------- Source ---------------------------------
% Use a pencil beam (required for SIA)
cfg.srctype = 'pencil';                      % pencil beam illumination

% Place source at the top surface, centered in x-y (grid coordinates).
cfg.srcpos  = [volN/2, volN/2, 0];           % source position in voxel coords

% Launch direction into the tissue (+z in MCX coordinate convention).
cfg.srcdir  = [0, 0, 1];                     % initial launch direction

cfg.issrcfrom0 = 1;                          % MCX source coordinate convention

% Provide a discretized “angleinvcdf” to enforce the launch cone (NA).
% MCX linearly interpolates internally between these points.
cfg.angleinvcdf = linspace(0, thetaMax/pi, 10);

% Number of launched photons for this run.
cfg.nphoton = nPhoton;                       % photons per run/batch

%% ------------------------------ Detector --------------------------------
% Boundary conditions:
%   'a' = absorbing; keep top surface open for detection '1'.
cfg.bc = 'aa_aaa001000';                     % absorbing boundaries except top

% Limit number of stored detected photons (cannot exceed launched photons here).
cfg.maxdetphoton = cfg.nphoton;              % cap stored detections

% Save detected-photon properties:
%   p = pathlength, x = exit position, v = exit direction 
cfg.savedetflag = 'pxv';                     % minimal set for reweighting + trimming
cfg.issave2pt = 0;                           % do not store fluence

%% -------------------------- Optical properties --------------------------

mu_a0 = 0;                                   % absorption during MCX [1/mm] 

% Two “media” rows:
%   row 1: external medium (label 0)
%   row 2: tissue medium (label 1)
cfg.prop = [
    0,     0,    1,  n_ext;                  % external
    mu_a0, mu_s, g1, n_tissue                % tissue
];

% Provide user-defined scattering ICDF so MCX uses the full phase function.
cfg.invcdf = icdf_mu;                        % overrides g usage internally

%% ------------------------------ GPU / misc ------------------------------

cfg.gpuid = 1;                               % GPU selection
cfg.autopilot = 1;                           % MCX autopilot

% Time parameters are not used for steady-state reflectance in this script,
% but MCX requires them.
cfg.tstart = 0;                              % start time [s]
cfg.tend   = 5e-9;                           % end time [s]
cfg.tstep  = 5e-9;                           % time step [s]

% Respins can increase effective photon count in some workflows; kept as 1 here.
cfg.respin = 1;                              % respin count (kept minimal)

%% ------------------------------- Run MCX --------------------------------
% Execute MCX simulation (CUDA; requires mcxlab on path).
[~, det] = mcxlabcl(cfg);                    % det contains detected photon data

%% ========================================================================
%   Map and scale detected-photon outputs to physical units
% ========================================================================

% The row mapping of det.data depends on MCX version/build and savedetflag.
%   det.data(1,:) = pathlength (in voxels)         -> convert to mm
%   det.data(2,:) = x_exit (in voxels)             -> convert to mm, center to source
%   det.data(3,:) = y_exit (in voxels)             -> convert to mm, center to source
%   det.data(7,:) = v_z exit direction vector      -> used for acceptance trimming

L_mm = det.data(1,:) * cfg.unitinmm;                     % pathlength in mm
x_mm = (det.data(2,:) - cfg.srcpos(1)) * cfg.unitinmm;   % centered x in mm
y_mm = (det.data(3,:) - cfg.srcpos(2)) * cfg.unitinmm;   % centered y in mm

% Exit radial distance rho from source position.
rho_exit = hypot(x_mm, y_mm);                % rho = sqrt(x^2 + y^2) in mm

% Exit angle relative to outward surface normal (-z); use acos(-v_z).
vz        = det.data(7,:);                   % exit direction cosine in z
theta_out = acos(-vz);                       % acceptance angle at exit [rad]

%% ========================================================================
%  Trim, reweight, compute R(rho) and SIA reflectance R_tot
% ========================================================================

% Maximum disk diameter used across requested df values.
d_f_max = max(d_f_list);                     % [mm]

% Keep only photons relevant for:
%   (i) spatial radius rho <= d_f_max
%  (ii) angular exit theta_out <= thetaMax
keep = (rho_exit <= d_f_max) & (theta_out <= thetaMax);

% Apply trimming to reduce computation and enforce modality definition.
L_mm     = L_mm(keep);                       % trimmed pathlengths
rho_exit = rho_exit(keep);                   % trimmed radial distances

% Total launched photons used for proper normalization (not detected count).
N_launched = cfg.nphoton * max(1, cfg.respin);  % effective launched photons

% Allocate output reflectance array: rows = mu_a, cols = d_f.
R_tot = nan(numel(mu_a_list), numel(d_f_list));

% Loop over absorption values (post-processing only).
for iA = 1:numel(mu_a_list)

    % Current absorption coefficient.
    mu_a = mu_a_list(iA);                    % [1/mm]

    % Beer–Lambert weight per detected photon: w = exp(-mu_a * pathlength).
    w = exp(-mu_a .* L_mm);                  % absorption reweighting

    % Loop over disk diameters.
    for iD = 1:numel(d_f_list)

        % Current disk diameter.
        d_f = d_f_list(iD);                  % [mm]

        % Build radial edges from 0 to d_f.
        rho_edges = linspace(0, d_f, nRhoEdges);   % fine annular edges

        % Assign each detected photon to a radial annulus.
        binIdx = discretize(rho_exit, rho_edges);  % annulus indices (NaN if outside)
        valid = ~isnan(binIdx);                    % valid annulus assignments

        % Accumulate absorption weights per annulus (sum of w in each ring).
        sumsW = accumarray(binIdx(valid).', double(w(valid)).', ...
                           [nRhoEdges-1, 1], @sum, 0);

        % Compute annular areas A_k = pi(r_k^2 - r_{k-1}^2).
        A_ann = pi * (rho_edges(2:end).^2 - rho_edges(1:end-1).^2);

        % Convert accumulated weight to radial reflectance R(rho) per annulus:
        %   R_k = (sum_w_in_annulus) / (A_k * N_launched).
        R_bin = (sumsW.' ./ (A_ann * N_launched)); 

        % Compute annulus midpoints for quadrature.
        rho_mid = (rho_edges(1:end-1) + rho_edges(2:end)) / 2;

        % Evaluate the disk-distance PDF p(rho; d_f) at midpoints.
        p_rho = disk_distance_pdf(rho_mid, d_f);

        % Evaluate the SIA integral using trapezoidal quadrature:
        %   R_tot = (pi/4)*d_f^2 * ∫_0^{d_f} R(rho) p(rho; d_f) d rho.
        R_tot(iA,iD) = (pi/4) * d_f^2 * trapz(rho_mid, R_bin .* p_rho);

    end
end

%% ========================================================================
%  Plot — Reflectance vs absorption for each disk diameter
% ========================================================================

fig = figure('Color','w');
ax = axes(fig);                               
hold(ax,'on');                                

mk = {'o','s','^','d','v','>'};               % marker set

% Plot each df curve.
for iD = 1:numel(d_f_list)
    m = mk{1 + mod(iD-1, numel(mk))};         
    plot(ax, mu_a_list, R_tot(:,iD), ['-' m])
end
xlabel(ax, '\mu_a [1/mm]');                   % x-axis label
ylabel(ax, 'R_{{tot}}');               % y-axis label
leg = legend(ax, arrayfun(@(d) sprintf('d_f = %.3g mm', d), d_f_list, ...
              'UniformOutput', false), 'Location','best');

%% ========================================================================
%  Local functions
% ========================================================================

function icdf_mu = make_icdf(mu, p_mu)
% Build an inverse-CDF table for mu = cos(theta).
%
% Inputs:
%   mu   : Nx1 grid in [-1,1]
%   p_mu : Nx1 PDF over mu (normalized to 1)
%
% Output:
%   icdf_mu : Mx1 inverse-CDF samples (mu values)


    % Cumulative distribution function over mu.
    cdf = cumtrapz(mu, p_mu);                 % unnormalized CDF

    % Normalize CDF so cdf(end) = 1.
    cdf = cdf ./ cdf(end);                    % normalized CDF

    % Choose uniform u-grid resolution for ICDF sampling.
    du = 5e-4;                                % ICDF resolution (example)
    u  = (du:du:1-du).';                      % avoid endpoints, gets sampled internally by MCX

    % Interpolate mu as a function of CDF to build inverse-CDF lookup.
    icdf_mu = interp1(cdf, mu, u, 'linear', 'extrap');

end


% Circle distance probability
function p = disk_distance_pdf(rho, df)
    p = 16 .* rho / pi / df^2 .* acos(rho / df) - 16 / pi / df .* (rho / df).^2 .* sqrt(1 - (rho / df).^2);
end
