% arb_sensitivity_sweep.m
% =======================
% Phase 3 — Anti-Roll Bar Sensitivity Analysis
%
% Uses a 2-DOF linear bicycle model to quantify:
%   1. How front ARB stiffness changes lateral load transfer distribution (LLTD)
%   2. Effect on understeer gradient (K_us)
%   3. Estimated lap time delta vs baseline setup
%
% Workflow mirrors what a GT3 Performance Engineer does between sessions:
%   "Driver says the car is understeering in slow corners — 
%    how much do I need to soften the front ARB to get neutral?"
%
% Theory:
%   - Lateral Load Transfer Distribution (LLTD_front) = ΔFz_front / ΔFz_total
%   - More front LLTD → more front tyre load → more understeer (and vice versa)
%   - Understeer gradient K_us = (m/L) * (a/C_αr - b/C_αf) [rad/g]
%     where C_αf, C_αr are effective cornering stiffnesses (modified by LLTD)
%
% Reference:
%   Milliken & Milliken, "Race Car Vehicle Dynamics", SAE, 1995
%   Chapter 5 (Steady-State Cornering) + Chapter 18 (Anti-Roll)
%
% -----------------------------------------------------------------------

clear; clc; close all;

fprintf('═══════════════════════════════════════════════════════\n');
fprintf('  ARB SENSITIVITY SWEEP — GT3 Setup Development\n');
fprintf('  2-DOF Bicycle Model | Toyota Racing Portfolio\n');
fprintf('═══════════════════════════════════════════════════════\n\n');


%% ── 1. VEHICLE PARAMETERS (GT3 baseline — Huracán GT3 proxy) ───────────────

% Geometry
L       = 2.62;         % wheelbase [m]
a       = L * 0.585;    % CG to rear axle [m]  (58.5% rear weight)
b       = L * 0.415;    % CG to front axle [m]
h_cg    = 0.44;         % CG height [m]
tf      = 1.64;         % front track [m]
tr      = 1.62;         % rear track [m]

% Mass & inertia
m       = 1350;         % total mass [kg]
Wf      = m * 9.81 * (a/L);   % static front axle load [N]
Wr      = m * 9.81 * (b/L);   % static rear axle load [N]

fprintf('Vehicle geometry:\n');
fprintf('  Wheelbase: %.2f m | CG split: %.1f%% front / %.1f%% rear\n', ...
    L, (b/L)*100, (a/L)*100);
fprintf('  Static front: %.0f N | Static rear: %.0f N\n\n', Wf, Wr);

% Tyre cornering stiffness at nominal load [N/deg]
% Based on Pacejka MF96 slope from Phase 2 (dFy/dα at α=0, Fz=Fz0)
C_alpha_f0 = 1650;      % N/deg (front, from MF96 BCD calculation)
C_alpha_r0 = 1520;      % N/deg (rear, slightly lower due to rear compliance)

% Load sensitivity of cornering stiffness: dC_α/dFz [N/deg per N]
% From Pacejka: cornering stiffness increases with load but degresses
dC_dFz_f   = 0.25;      % N/deg per N
dC_dFz_r   = 0.22;

% Nominal (baseline) ARB stiffnesses [N·m/deg]
% GT3 typical range: front 15–60 N·m/deg, rear 10–50 N·m/deg
ARB_front_base = 35.0;  % N·m/deg
ARB_rear_base  = 25.0;  % N·m/deg

% Spring rates (needed for total roll stiffness)
k_spring_f  = 85000;    % N/m (front spring stiffness)
k_spring_r  = 70000;    % N/m (rear spring stiffness)
k_arb_to_Nm = 1000;     % conversion: N·m/deg to N/m equivalent at wheel

% Baseline lap time reference (from Phase 2 degradation model, fresh tyre)
t_lap_base  = 95.175;   % seconds


%% ── 2. ROLL STIFFNESS & LLTD CALCULATION ────────────────────────────────────

function [LLTD_f, LLTD_r, K_roll_f, K_roll_r] = calc_lltd(ARB_f, ARB_r, params)
    % Returns LLTD_front (fraction), LLTD_rear, and axle roll stiffnesses
    %
    % Roll stiffness per axle [N·m/deg]:
    %   K_roll = K_spring_contribution + K_ARB
    %
    % Spring contribution to roll stiffness [N·m/deg]:
    %   K_spring_roll = k_spring * (track/2)^2 / (180/pi)
    %   (factor converts to N·m/deg)

    deg2rad = pi / 180;

    K_spring_roll_f = params.k_spring_f * (params.tf/2)^2 * deg2rad;
    K_spring_roll_r = params.k_spring_r * (params.tr/2)^2 * deg2rad;

    K_roll_f = K_spring_roll_f + ARB_f;
    K_roll_r = K_spring_roll_r + ARB_r;
    K_roll_total = K_roll_f + K_roll_r;

    LLTD_f = K_roll_f / K_roll_total;
    LLTD_r = K_roll_r / K_roll_total;
end


function K_us = calc_understeer_gradient(ARB_f, ARB_r, ay_g, params)
    % Understeer gradient [deg/g] at lateral acceleration ay_g
    %
    % Steps:
    %   1. Compute lateral load transfer per axle at ay_g
    %   2. Get effective cornering stiffness at loaded condition
    %   3. K_us = (m/L) * (a/C_αr - b/C_αf)  [converted to deg/g]

    g = 9.81;

    % LLTD
    [LLTD_f, LLTD_r, ~, ~] = calc_lltd(ARB_f, ARB_r, params);

    % Total lateral load transfer [N] at ay_g
    delta_Fz_total = m * ay_g * g * params.h_cg / params.tf;   % approx (use avg track)

    % Per-axle load transfer
    delta_Fz_f = LLTD_f * delta_Fz_total;
    delta_Fz_r = LLTD_r * delta_Fz_total;

    % Effective cornering stiffness (load-modified)
    % Loaded tyre gains, unloaded tyre loses — net effect depends on linearity
    C_f_eff = params.C_alpha_f0 + params.dC_dFz_f * delta_Fz_f;
    C_r_eff = params.C_alpha_r0 + params.dC_dFz_r * delta_Fz_r;

    % Understeer gradient [rad/g]
    K_us_rad = (m / params.L) * (params.a / C_r_eff - params.b / C_f_eff);

    % Convert to [deg/g]
    K_us = K_us_rad * (180 / pi);
end


%% ── 3. PARAMETER STRUCT ─────────────────────────────────────────────────────
p.L = L; p.a = a; p.b = b; p.h_cg = h_cg;
p.tf = tf; p.tr = tr; p.m = m;
p.k_spring_f = k_spring_f; p.k_spring_r = k_spring_r;
p.C_alpha_f0 = C_alpha_f0; p.C_alpha_r0 = C_alpha_r0;
p.dC_dFz_f = dC_dFz_f; p.dC_dFz_r = dC_dFz_r;


%% ── 4. PARAMETRIC SWEEP ─────────────────────────────────────────────────────
% Sweep front ARB ±60% of baseline, rear ARB fixed at baseline
ARB_front_sweep = linspace(ARB_front_base * 0.4, ARB_front_base * 1.6, 50);
ay_conditions   = [0.5, 1.0, 1.5, 2.0];   % lateral acc [g] — corner types

fprintf('ARB sweep: %.1f to %.1f N·m/deg  (rear fixed at %.1f)\n\n', ...
    ARB_front_sweep(1), ARB_front_sweep(end), ARB_rear_base);

% Pre-allocate
LLTD_f_arr  = zeros(1, length(ARB_front_sweep));
Kus_arr     = zeros(length(ay_conditions), length(ARB_front_sweep));

for i = 1:length(ARB_front_sweep)
    [lltd_f, ~, ~, ~] = calc_lltd(ARB_front_sweep(i), ARB_rear_base, p);
    LLTD_f_arr(i) = lltd_f * 100;   % convert to %

    for j = 1:length(ay_conditions)
        Kus_arr(j,i) = calc_understeer_gradient(...
            ARB_front_sweep(i), ARB_rear_base, ay_conditions(j), p);
    end
end

% Baseline values
[lltd_base, ~, ~, ~] = calc_lltd(ARB_front_base, ARB_rear_base, p);
K_us_base = calc_understeer_gradient(ARB_front_base, ARB_rear_base, 1.5, p);

fprintf('Baseline LLTD_front: %.1f%%\n', lltd_base * 100);
fprintf('Baseline K_us at 1.5g: %.3f deg/g\n', K_us_base);
fprintf('  Positive K_us = understeer, Negative = oversteer\n\n');


%% ── 5. LAP TIME DELTA ESTIMATION ─────────────────────────────────────────────
% Simplified sensitivity: each deg/g of K_us from neutral costs ~0.03 s/lap
% (empirical GT3 reference: Segers J., "Analysis Techniques for Racecar
%  Data Acquisition", 2nd ed., SAE 2014, p.287)
%
% Neutral K_us = 0 (perfectly balanced car)
% Delta_laptime = sensitivity_factor * K_us²
% Quadratic because both US and OS hurt lap time symmetrically

sensitivity = 0.03;     % s per (deg/g)² — conservative GT3 estimate
K_us_at_1g  = Kus_arr(2,:);   % use 1g as representative corner load
delta_lt    = sensitivity * K_us_at_1g.^2;

% Baseline delta
delta_lt_base = sensitivity * K_us_base^2;

fprintf('Lap time sensitivity:\n');
fprintf('  Neutral setup delta: +%.3f s\n', delta_lt_base);
fprintf('  Softest front ARB  : +%.3f s  (K_us=%.3f)\n', ...
    delta_lt(1), K_us_at_1g(1));
fprintf('  Hardest front ARB  : +%.3f s  (K_us=%.3f)\n\n', ...
    delta_lt(end), K_us_at_1g(end));


%% ── 6. OPTIMAL ARB RANGE ────────────────────────────────────────────────────
% Find front ARB range that keeps K_us within ±0.5 deg/g (near-neutral window)
tolerance  = 0.5;   % deg/g
within_idx = find(abs(K_us_at_1g) <= tolerance);
if ~isempty(within_idx)
    ARB_opt_lo = ARB_front_sweep(within_idx(1));
    ARB_opt_hi = ARB_front_sweep(within_idx(end));
    fprintf('Neutral window (|K_us| ≤ %.1f deg/g): %.1f – %.1f N·m/deg\n\n', ...
        tolerance, ARB_opt_lo, ARB_opt_hi);
else
    ARB_opt_lo = NaN; ARB_opt_hi = NaN;
end


%% ── 7. PLOTS ─────────────────────────────────────────────────────────────────
bg_dark = [0.06 0.06 0.06];
ax_dark = [0.10 0.10 0.10];
txt_col = [0.85 0.85 0.85];
red     = [0.90 0.07 0.07];
cyan    = [0.00 0.71 0.85];
amber   = [0.96 0.64 0.38];
green   = [0.32 0.72 0.53];
ay_cols = [red; cyan; amber; green];

f1 = figure('Color', bg_dark, 'Position', [50 50 1400 900]);
sgtitle('ARB SENSITIVITY SWEEP — GT3 Setup Development', ...
    'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Courier New');

% ── Panel A: LLTD vs ARB ─────────────────────────────────────────────────────
ax1 = subplot(2, 2, 1);
set(ax1, 'Color', ax_dark, 'XColor', txt_col, 'YColor', txt_col, ...
    'GridColor', [0.2 0.2 0.2], 'GridLineStyle', '--', 'FontName', 'Courier New');
hold on; grid on;

plot(ARB_front_sweep, LLTD_f_arr, 'Color', cyan, 'LineWidth', 2);
yline(lltd_base * 100, 'Color', [0.5 0.5 0.5], 'LineWidth', 1, ...
    'LineStyle', '--', 'Label', sprintf('Baseline %.1f%%', lltd_base*100), ...
    'LabelHorizontalAlignment', 'right', 'FontName', 'Courier New');

% Shade neutral zone (45-55% LLTD typical GT3)
patch([ARB_front_sweep(1) ARB_front_sweep(end) ARB_front_sweep(end) ARB_front_sweep(1)], ...
    [45 45 55 55], green, 'FaceAlpha', 0.08, 'EdgeColor', 'none');
yline([45 55], 'Color', [green 0.5], 'LineWidth', 0.6, 'LineStyle', ':');

xline(ARB_front_base, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'LineStyle', ':');
xlabel('Front ARB stiffness (N·m/deg)', 'Color', txt_col, 'FontName', 'Courier New');
ylabel('LLTD front (%)', 'Color', txt_col, 'FontName', 'Courier New');
title('A: Lateral Load Transfer Distribution', 'Color', 'w', 'FontName', 'Courier New', ...
    'FontSize', 9);
text(ARB_front_sweep(3), 56.5, 'Neutral zone (45–55%)', ...
    'Color', green, 'FontSize', 7, 'FontName', 'Courier New');

% ── Panel B: Understeer gradient ────────────────────────────────────────────
ax2 = subplot(2, 2, 2);
set(ax2, 'Color', ax_dark, 'XColor', txt_col, 'YColor', txt_col, ...
    'GridColor', [0.2 0.2 0.2], 'GridLineStyle', '--', 'FontName', 'Courier New');
hold on; grid on;

for j = 1:length(ay_conditions)
    plot(ARB_front_sweep, Kus_arr(j,:), 'Color', ay_cols(j,:), ...
        'LineWidth', 1.8, 'DisplayName', sprintf('%.1fg', ay_conditions(j)));
end
yline(0, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2, 'LineStyle', '-');
patch([ARB_front_sweep(1) ARB_front_sweep(end) ARB_front_sweep(end) ARB_front_sweep(1)], ...
    [-tolerance -tolerance tolerance tolerance], green, ...
    'FaceAlpha', 0.06, 'EdgeColor', 'none');

if ~isnan(ARB_opt_lo)
    patch([ARB_opt_lo ARB_opt_hi ARB_opt_hi ARB_opt_lo], ...
        [min(Kus_arr(:)) min(Kus_arr(:)) max(Kus_arr(:)) max(Kus_arr(:))], ...
        green, 'FaceAlpha', 0.06, 'EdgeColor', green, 'LineStyle', ':');
end

xline(ARB_front_base, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'LineStyle', ':');
xlabel('Front ARB stiffness (N·m/deg)', 'Color', txt_col, 'FontName', 'Courier New');
ylabel('Understeer gradient K_{us} (deg/g)', 'Color', txt_col, 'FontName', 'Courier New');
title('B: Understeer Gradient  (0 = neutral)', 'Color', 'w', ...
    'FontName', 'Courier New', 'FontSize', 9);
legend('show', 'Location', 'northwest', 'TextColor', txt_col, ...
    'Color', ax_dark, 'EdgeColor', [0.3 0.3 0.3], 'FontName', 'Courier New', ...
    'FontSize', 7, 'Title', 'ay [g]');
text(ARB_front_sweep(5), tolerance*1.05, '← Oversteer  |  Understeer →', ...
    'Color', [0.6 0.6 0.6], 'FontSize', 7, 'FontName', 'Courier New');

% ── Panel C: Lap time delta ──────────────────────────────────────────────────
ax3 = subplot(2, 2, 3);
set(ax3, 'Color', ax_dark, 'XColor', txt_col, 'YColor', txt_col, ...
    'GridColor', [0.2 0.2 0.2], 'GridLineStyle', '--', 'FontName', 'Courier New');
hold on; grid on;

fill([ARB_front_sweep fliplr(ARB_front_sweep)], ...
    [delta_lt * 1000 zeros(1,length(ARB_front_sweep))], ...
    red, 'FaceAlpha', 0.25, 'EdgeColor', 'none');
plot(ARB_front_sweep, delta_lt * 1000, 'Color', red, 'LineWidth', 2);

if ~isnan(ARB_opt_lo)
    patch([ARB_opt_lo ARB_opt_hi ARB_opt_hi ARB_opt_lo], ...
        [0 0 max(delta_lt*1000)*1.1 max(delta_lt*1000)*1.1], ...
        green, 'FaceAlpha', 0.12, 'EdgeColor', green, 'LineStyle', ':');
    text(ARB_opt_lo + 0.5, max(delta_lt*1000)*0.6, ...
        sprintf('Optimal\n%.0f–%.0f N·m/deg', ARB_opt_lo, ARB_opt_hi), ...
        'Color', green, 'FontSize', 7, 'FontName', 'Courier New');
end

xline(ARB_front_base, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, 'LineStyle', ':');
xlabel('Front ARB stiffness (N·m/deg)', 'Color', txt_col, 'FontName', 'Courier New');
ylabel('Lap time delta (ms)', 'Color', txt_col, 'FontName', 'Courier New');
title('C: Estimated Lap Time Delta vs Neutral', 'Color', 'w', ...
    'FontName', 'Courier New', 'FontSize', 9);

% ── Panel D: Balance map (front ARB vs rear ARB) ─────────────────────────────
ax4 = subplot(2, 2, 4);
set(ax4, 'Color', ax_dark, 'XColor', txt_col, 'YColor', txt_col, ...
    'GridColor', [0.2 0.2 0.2], 'GridLineStyle', '--', 'FontName', 'Courier New');

ARB_r_sweep = linspace(ARB_rear_base * 0.4, ARB_rear_base * 1.6, 40);
[ARB_F_grid, ARB_R_grid] = meshgrid(ARB_front_sweep(1:40), ARB_r_sweep);
Kus_grid = zeros(size(ARB_F_grid));

for i = 1:size(ARB_F_grid, 1)
    for j = 1:size(ARB_F_grid, 2)
        Kus_grid(i,j) = calc_understeer_gradient(...
            ARB_F_grid(i,j), ARB_R_grid(i,j), 1.5, p);
    end
end

contourf(ARB_F_grid, ARB_R_grid, Kus_grid, 20, 'LineColor', 'none');
colormap(ax4, redblue_cmap());
cb = colorbar(ax4);
cb.Label.String = 'K_{us} (deg/g)';
cb.Label.Color  = txt_col;
cb.Color        = txt_col;
clim([-2 2]);

contour(ARB_F_grid, ARB_R_grid, Kus_grid, [0 0], 'w', 'LineWidth', 2);
plot(ARB_front_base, ARB_rear_base, 'w+', 'MarkerSize', 12, 'LineWidth', 2);
text(ARB_front_base + 0.5, ARB_rear_base + 0.3, 'Baseline', ...
    'Color', 'w', 'FontSize', 7, 'FontName', 'Courier New');

xlabel('Front ARB stiffness (N·m/deg)', 'Color', txt_col, 'FontName', 'Courier New');
ylabel('Rear ARB stiffness (N·m/deg)', 'Color', txt_col, 'FontName', 'Courier New');
title('D: Balance Map  (white line = neutral, K_{us}=0)', 'Color', 'w', ...
    'FontName', 'Courier New', 'FontSize', 9);

saveas(f1, 'output/phase3_09_arb_sweep.png');
fprintf('Saved → output/phase3_09_arb_sweep.png\n');


%% ── HELPER: red-blue colormap ────────────────────────────────────────────────
function cmap = redblue_cmap(n)
    if nargin < 1; n = 256; end
    top    = [linspace(0,1,n/2)', linspace(0,0,n/2)', linspace(1,0,n/2)'];
    bottom = [linspace(1,1,n/2)', linspace(0,0,n/2)', linspace(0,0,n/2)'];
    cmap   = [top; bottom];
end


fprintf('\n✓ ARB sweep complete.\n');
fprintf('  Front ARB optimal range: %.1f – %.1f N·m/deg\n', ARB_opt_lo, ARB_opt_hi);
fprintf('  Use Panel D balance map in driver debrief:\n');
fprintf('    "If you want more rotation in slow corners, we soften front ARB\n');
fprintf('     from %.0f to ~%.0f N·m/deg — that shifts K_us %.2f deg/g\n', ...
    ARB_front_base, ARB_opt_lo, K_us_base - ...
    calc_understeer_gradient(ARB_opt_lo, ARB_rear_base, 1.5, p));
fprintf('     toward neutral. Est. lap time gain: +%.0f ms."\n\n', ...
    (delta_lt_base - min(delta_lt)) * 1000);
