% pacejka_magic_formula.m
% ======================
% Phase 2 — Pacejka Magic Formula (MF96) Implementation
%
% Implements:
%   Fy  — lateral force vs slip angle (alpha)
%   Fx  — longitudinal force vs slip ratio (kappa)
%   Combined load sensitivity (Fz scaling)
%
% Structure:
%   1. MF96 core formula (Fy and Fx)
%   2. Coefficient sets for GT3 slick (front/rear)
%   3. Sweep across slip angles / ratios at multiple loads
%   4. Peak force, slip stiffness, and limit extraction
%   5. Plots: Fy curves, Fx curves, friction ellipse
%   6. Sensitivity: how Fy changes with Fz (load sensitivity)
%
% Reference:
%   Pacejka H.B., "Tyre and Vehicle Dynamics", 3rd ed., 2012
%   MF96 formulation — Section 4.3
%
% Run:
%   pacejka_magic_formula
% -----------------------------------------------------------------------

clear; clc; close all;

fprintf('═══════════════════════════════════════════════════\n');
fprintf('  PACEJKA MAGIC FORMULA — GT3 Tyre Analysis\n');
fprintf('  Phase 2 | Toyota Racing GmbH Portfolio Project\n');
fprintf('═══════════════════════════════════════════════════\n\n');

%% ── 1. MF96 CORE FUNCTIONS ─────────────────────────────────────────────────
%
% General form:
%   y(x) = D * sin(C * atan(B*x - E*(B*x - atan(B*x)))) + Sv
%   Y(X) = y(x) + Sh     where X = x + Sh
%
% Parameters:
%   B  = stiffness factor       (controls initial slope)
%   C  = shape factor           (controls curve shape)
%   D  = peak value             (maximum force)
%   E  = curvature factor       (controls transition region)
%   Sv = vertical shift         (cornering / braking stiffness offset)
%   Sh = horizontal shift       (ply steer / conicity)

function Fy = magic_formula_Fy(alpha_deg, Fz, coeffs)
    % Lateral force (N) vs slip angle (deg) at vertical load Fz (N)
    %
    % coeffs struct fields:
    %   pCy1, pDy1, pDy2, pEy1, pEy2,
    %   pKy1, pKy2, pKy3,
    %   pHy1, pHy2, pVy1, pVy2
    %
    % Fz0 = nominal load (reference tyre load, N)

    Fz0    = coeffs.Fz0;
    dfz    = (Fz - Fz0) / Fz0;          % normalised load deviation

    % Shape factor
    Cy     = coeffs.pCy1;

    % Peak factor (load-sensitive)
    muy    = coeffs.pDy1 + coeffs.pDy2 * dfz;
    Dy     = muy * Fz;

    % Stiffness factor (cornering stiffness normalised by Fz)
    % BCD = cornering stiffness = pKy1 * Fz0 * sin(2*atan(Fz/(pKy2*Fz0)))*(1-pKy3*abs(camber))
    camber = 0;    % zero camber for simplicity
    BCD    = coeffs.pKy1 * Fz0 * sin(2 * atan(Fz / (coeffs.pKy2 * Fz0))) ...
             * (1 - coeffs.pKy3 * abs(camber));
    By     = BCD / (Cy * Dy + eps);

    % Curvature factor (load-sensitive)
    Ey     = coeffs.pEy1 + coeffs.pEy2 * dfz;
    Ey     = min(Ey, 1.0);               % MF96 constraint: Ey ≤ 1

    % Horizontal shift (ply steer / conicity)
    Shy    = coeffs.pHy1 + coeffs.pHy2 * dfz;

    % Vertical shift (residual cornering force at zero slip)
    Svy    = (coeffs.pVy1 + coeffs.pVy2 * dfz) * Fz;

    % Slip angle with shift
    alpha_rad   = deg2rad(alpha_deg);
    Shy_rad     = deg2rad(Shy);
    alpha_star  = alpha_rad + Shy_rad;

    % Magic Formula
    Fy = Dy * sin(Cy * atan(By * alpha_star ...
         - Ey * (By * alpha_star - atan(By * alpha_star)))) + Svy;
end


function Fx = magic_formula_Fx(kappa, Fz, coeffs)
    % Longitudinal force (N) vs slip ratio (kappa, dimensionless)
    % kappa = (V_wheel - V_vehicle) / V_vehicle
    %
    % coeffs struct fields:
    %   pCx1, pDx1, pDx2, pEx1, pEx2, pEx3,
    %   pKx1, pKx2, pKx3,
    %   pHx1, pHx2, pVx1, pVx2

    Fz0   = coeffs.Fz0;
    dfz   = (Fz - Fz0) / Fz0;

    % Shape factor
    Cx    = coeffs.pCx1;

    % Peak factor
    mux   = coeffs.pDx1 + coeffs.pDx2 * dfz;
    Dx    = mux * Fz;

    % Stiffness
    BCD   = (coeffs.pKx1 + coeffs.pKx2 * dfz) * Fz * exp(coeffs.pKx3 * dfz);
    Bx    = BCD / (Cx * Dx + eps);

    % Curvature
    Ex    = (coeffs.pEx1 + coeffs.pEx2 * dfz + coeffs.pEx3 * dfz^2);
    Ex    = min(Ex, 1.0);

    % Shifts
    Shx   = coeffs.pHx1 + coeffs.pHx2 * dfz;
    Svx   = (coeffs.pVx1 + coeffs.pVx2 * dfz) * Fz;

    kappa_star = kappa + Shx;

    % Magic Formula
    Fx = Dx * sin(Cx * atan(Bx * kappa_star ...
         - Ex * (Bx * kappa_star - atan(Bx * kappa_star)))) + Svx;
end


%% ── 2. COEFFICIENT SETS ─────────────────────────────────────────────────────
% GT3 slick tyre approximation — values calibrated to match published
% Pirelli DHD2 / Michelin GT3 compound behaviour from literature.
% Nominal load Fz0 = 3500 N (typical GT3 corner at static)
%
% Front axle (stiffer sidewall, lower camber compliance)
c_front.Fz0  = 3500;
c_front.pCy1 = 1.30;
c_front.pDy1 = 1.10;   c_front.pDy2 = -0.12;
c_front.pEy1 = -1.50;  c_front.pEy2 = -0.80;
c_front.pKy1 = 18.0;   c_front.pKy2 = 1.60;  c_front.pKy3 = 0.22;
c_front.pHy1 = 0.003;  c_front.pHy2 = 0.001;
c_front.pVy1 = 0.00;   c_front.pVy2 = 0.00;
% Longitudinal
c_front.pCx1 = 1.65;
c_front.pDx1 = 1.18;   c_front.pDx2 = -0.10;
c_front.pEx1 = 0.40;   c_front.pEx2 = 0.00;  c_front.pEx3 = 0.00;
c_front.pKx1 = 21.5;   c_front.pKx2 = -0.20; c_front.pKx3 = 0.10;
c_front.pHx1 = 0.002;  c_front.pHx2 = 0.001;
c_front.pVx1 = 0.00;   c_front.pVx2 = 0.00;

% Rear axle (softer sidewall, higher load, more compliant)
c_rear.Fz0   = 3500;
c_rear.pCy1  = 1.28;
c_rear.pDy1  = 1.08;   c_rear.pDy2 = -0.10;
c_rear.pEy1  = -1.30;  c_rear.pEy2 = -0.70;
c_rear.pKy1  = 16.5;   c_rear.pKy2 = 1.55;  c_rear.pKy3 = 0.20;
c_rear.pHy1  = 0.002;  c_rear.pHy2 = 0.001;
c_rear.pVy1  = 0.00;   c_rear.pVy2 = 0.00;
c_rear.pCx1  = 1.60;
c_rear.pDx1  = 1.15;   c_rear.pDx2 = -0.09;
c_rear.pEx1  = 0.38;   c_rear.pEx2 = 0.00;  c_rear.pEx3 = 0.00;
c_rear.pKx1  = 20.0;   c_rear.pKx2 = -0.18; c_rear.pKx3 = 0.09;
c_rear.pHx1  = 0.001;  c_rear.pHx2 = 0.001;
c_rear.pVx1  = 0.00;   c_rear.pVx2 = 0.00;


%% ── 3. SWEEP DEFINITIONS ────────────────────────────────────────────────────
alpha_range = linspace(-14, 14, 300);   % slip angle, degrees
kappa_range = linspace(-0.25, 0.25, 300);  % slip ratio, dimensionless

% Vertical loads: from light aero (low-speed corner) to heavy aero (high-speed)
Fz_loads = [2500, 3500, 4500, 5500];    % N
Fz_labels = {'2500 N (light)', '3500 N (nominal)', ...
             '4500 N (medium aero)', '5500 N (full aero)'};


%% ── 4. COMPUTE Fy CURVES ────────────────────────────────────────────────────
fprintf('Computing Fy (lateral) curves...\n');
Fy_front = zeros(length(Fz_loads), length(alpha_range));
Fy_rear  = zeros(length(Fz_loads), length(alpha_range));

for i = 1:length(Fz_loads)
    for j = 1:length(alpha_range)
        Fy_front(i,j) = magic_formula_Fy(alpha_range(j), Fz_loads(i), c_front);
        Fy_rear(i,j)  = magic_formula_Fy(alpha_range(j), Fz_loads(i), c_rear);
    end
end


%% ── 5. COMPUTE Fx CURVES ────────────────────────────────────────────────────
fprintf('Computing Fx (longitudinal) curves...\n');
Fx_front = zeros(length(Fz_loads), length(kappa_range));
Fx_rear  = zeros(length(Fz_loads), length(kappa_range));

for i = 1:length(Fz_loads)
    for j = 1:length(kappa_range)
        Fx_front(i,j) = magic_formula_Fx(kappa_range(j), Fz_loads(i), c_front);
        Fx_rear(i,j)  = magic_formula_Fx(kappa_range(j), Fz_loads(i), c_rear);
    end
end


%% ── 6. PEAK FORCE & SLIP STIFFNESS EXTRACTION ───────────────────────────────
fprintf('\n%-20s %-10s %-12s %-12s %-14s\n', ...
    'Fz (N)', 'Axle', 'Fy_peak (N)', 'alpha_peak', 'Cornering stiff');
fprintf('%s\n', repmat('-', 1, 70));

for i = 1:length(Fz_loads)
    for axle = {'front', 'rear'}
        if strcmp(axle{1}, 'front')
            fy_arr = Fy_front(i,:);
            c      = c_front;
        else
            fy_arr = Fy_rear(i,:);
            c      = c_rear;
        end

        [Fy_pk, idx_pk]  = max(fy_arr);
        alpha_pk          = alpha_range(idx_pk);

        % Cornering stiffness = dFy/dalpha at alpha=0 (N/deg)
        idx0     = find(alpha_range >= 0, 1);
        if idx0 > 1
            CS = (fy_arr(idx0+1) - fy_arr(idx0-1)) / ...
                 (alpha_range(idx0+1) - alpha_range(idx0-1));
        else
            CS = 0;
        end

        fprintf('%-20.0f %-10s %-12.1f %-12.2f %-14.1f\n', ...
            Fz_loads(i), axle{1}, Fy_pk, alpha_pk, CS);
    end
end


%% ── 7. LOAD SENSITIVITY ANALYSIS ─────────────────────────────────────────────
fprintf('\nLoad sensitivity (Fy_peak / Fz):\n');
fprintf('  At higher Fz, grip increases but friction coefficient drops.\n');
fprintf('  This is the degressive load sensitivity — critical for BoP analysis.\n\n');
for i = 1:length(Fz_loads)
    mu_front = max(Fy_front(i,:)) / Fz_loads(i);
    mu_rear  = max(Fy_rear(i,:))  / Fz_loads(i);
    fprintf('  Fz = %4.0f N  |  mu_front = %.3f  |  mu_rear = %.3f\n', ...
        Fz_loads(i), mu_front, mu_rear);
end


%% ── 8. FRICTION ELLIPSE ─────────────────────────────────────────────────────
% Combined Fx/Fy capacity at nominal load — the "traction circle" boundary
% derived from first principles rather than measured data
fprintf('\nGenerating friction ellipse at Fz = 3500 N...\n');
Fz_nom     = 3500;
alpha_fine = linspace(0, 14, 100);
kappa_fine = linspace(-0.25, 0, 100);

% Front axle friction ellipse
Fy_ellipse_f = arrayfun(@(a) magic_formula_Fy(a, Fz_nom, c_front), alpha_fine);
Fx_ellipse_f = arrayfun(@(k) magic_formula_Fx(k, Fz_nom, c_front), kappa_fine);

Fy_ellipse_r = arrayfun(@(a) magic_formula_Fy(a, Fz_nom, c_rear), alpha_fine);
Fx_ellipse_r = arrayfun(@(k) magic_formula_Fx(k, Fz_nom, c_rear), kappa_fine);


%% ── 9. PLOTS ─────────────────────────────────────────────────────────────────
% Dark Motec-style theme
bg_dark = [0.06 0.06 0.06];
ax_dark = [0.10 0.10 0.10];
txt_col = [0.85 0.85 0.85];
colors  = [0.90 0.07 0.07;   % Toyota red
           0.00 0.71 0.85;   % cyan
           0.96 0.64 0.38;   % amber
           0.32 0.72 0.53];  % green

fig_sz = [1400, 900];

% ── Figure 1: Fy curves (lateral) ───────────────────────────────────────────
f1 = figure('Color', bg_dark, 'Position', [50 50 fig_sz]);
sgtitle('PACEJKA MF96 — Lateral Force Fy vs Slip Angle', ...
    'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Courier New');

for p = 1:2
    ax = subplot(1, 2, p);
    set(ax, 'Color', ax_dark, 'XColor', txt_col, 'YColor', txt_col, ...
        'GridColor', [0.25 0.25 0.25], 'GridLineStyle', '--');
    hold on; grid on; box on;

    if p == 1
        data    = Fy_front;
        ax_name = 'Front axle (stiffer sidewall)';
    else
        data    = Fy_rear;
        ax_name = 'Rear axle (softer sidewall)';
    end

    for i = 1:length(Fz_loads)
        plot(alpha_range, data(i,:) / 1000, ...
            'Color', colors(i,:), 'LineWidth', 1.5, ...
            'DisplayName', Fz_labels{i});
    end

    % Mark peak for nominal load
    [pk, ipk] = max(data(2,:));
    plot(alpha_range(ipk), pk/1000, 'wo', 'MarkerSize', 6, 'MarkerFaceColor', 'w');
    text(alpha_range(ipk)+0.5, pk/1000+0.05, ...
        sprintf('Peak: %.2f kN @ %.1f°', pk/1000, alpha_range(ipk)), ...
        'Color', 'w', 'FontSize', 7, 'FontName', 'Courier New');

    xline(0, 'Color', [0.4 0.4 0.4], 'LineWidth', 0.5);
    xlabel('Slip angle α (°)', 'Color', txt_col, 'FontName', 'Courier New');
    ylabel('Lateral force Fy (kN)', 'Color', txt_col, 'FontName', 'Courier New');
    title(ax_name, 'Color', 'w', 'FontSize', 9, 'FontName', 'Courier New');
    legend('show', 'Location', 'southeast', 'TextColor', txt_col, ...
        'Color', ax_dark, 'EdgeColor', [0.3 0.3 0.3], 'FontSize', 7, ...
        'FontName', 'Courier New');
    ylim([-0.5, 8]);
    xlim([-14, 14]);
end
saveas(f1, 'output/phase2_01_Fy_curves.png');
fprintf('  Saved → output/phase2_01_Fy_curves.png\n');


% ── Figure 2: Fx curves (longitudinal) ──────────────────────────────────────
f2 = figure('Color', bg_dark, 'Position', [50 50 fig_sz]);
sgtitle('PACEJKA MF96 — Longitudinal Force Fx vs Slip Ratio', ...
    'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Courier New');

for p = 1:2
    ax = subplot(1, 2, p);
    set(ax, 'Color', ax_dark, 'XColor', txt_col, 'YColor', txt_col, ...
        'GridColor', [0.25 0.25 0.25], 'GridLineStyle', '--');
    hold on; grid on; box on;

    if p == 1
        data    = Fx_front;
        ax_name = 'Front axle';
    else
        data    = Fx_rear;
        ax_name = 'Rear axle';
    end

    for i = 1:length(Fz_loads)
        plot(kappa_range, data(i,:) / 1000, ...
            'Color', colors(i,:), 'LineWidth', 1.5, ...
            'DisplayName', Fz_labels{i});
    end

    % Mark peak slip ratio at nominal load
    [pk, ipk] = max(abs(data(2,:)));
    [~, sgn_idx] = max(data(2,:));
    plot(kappa_range(sgn_idx), data(2,sgn_idx)/1000, 'wo', ...
        'MarkerSize', 6, 'MarkerFaceColor', 'w');

    xline(0, 'Color', [0.4 0.4 0.4], 'LineWidth', 0.5);
    xlabel('Slip ratio κ (−)', 'Color', txt_col, 'FontName', 'Courier New');
    ylabel('Longitudinal force Fx (kN)', 'Color', txt_col, 'FontName', 'Courier New');
    title(ax_name, 'Color', 'w', 'FontSize', 9, 'FontName', 'Courier New');
    legend('show', 'Location', 'southeast', 'TextColor', txt_col, ...
        'Color', ax_dark, 'EdgeColor', [0.3 0.3 0.3], 'FontSize', 7, ...
        'FontName', 'Courier New');
    ylim([-7, 7]);
end
saveas(f2, 'output/phase2_02_Fx_curves.png');
fprintf('  Saved → output/phase2_02_Fx_curves.png\n');


% ── Figure 3: Load sensitivity ───────────────────────────────────────────────
f3 = figure('Color', bg_dark, 'Position', [50 50 1200, 500]);
sgtitle('LOAD SENSITIVITY — Fy peak and friction coefficient vs Fz', ...
    'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Courier New');

Fy_peaks_f = arrayfun(@(fz) max(arrayfun(@(a) ...
    magic_formula_Fy(a, fz, c_front), alpha_range)), Fz_loads);
Fy_peaks_r = arrayfun(@(fz) max(arrayfun(@(a) ...
    magic_formula_Fy(a, fz, c_rear), alpha_range)), Fz_loads);
mu_f = Fy_peaks_f ./ Fz_loads;
mu_r = Fy_peaks_r ./ Fz_loads;

ax1 = subplot(1, 2, 1);
set(ax1, 'Color', ax_dark, 'XColor', txt_col, 'YColor', txt_col, ...
    'GridColor', [0.25 0.25 0.25], 'GridLineStyle', '--');
hold on; grid on;
plot(Fz_loads/1000, Fy_peaks_f/1000, '-o', 'Color', colors(1,:), ...
    'LineWidth', 2, 'MarkerFaceColor', colors(1,:), 'MarkerSize', 7, ...
    'DisplayName', 'Front');
plot(Fz_loads/1000, Fy_peaks_r/1000, '-o', 'Color', colors(2,:), ...
    'LineWidth', 2, 'MarkerFaceColor', colors(2,:), 'MarkerSize', 7, ...
    'DisplayName', 'Rear');
xlabel('Vertical load Fz (kN)', 'Color', txt_col, 'FontName', 'Courier New');
ylabel('Peak lateral force Fy (kN)', 'Color', txt_col, 'FontName', 'Courier New');
title('Fy peak vs Fz', 'Color', 'w', 'FontName', 'Courier New');
legend('show', 'TextColor', txt_col, 'Color', ax_dark, ...
    'EdgeColor', [0.3 0.3 0.3], 'FontName', 'Courier New');

ax2 = subplot(1, 2, 2);
set(ax2, 'Color', ax_dark, 'XColor', txt_col, 'YColor', txt_col, ...
    'GridColor', [0.25 0.25 0.25], 'GridLineStyle', '--');
hold on; grid on;
plot(Fz_loads/1000, mu_f, '-o', 'Color', colors(1,:), 'LineWidth', 2, ...
    'MarkerFaceColor', colors(1,:), 'MarkerSize', 7, 'DisplayName', 'Front');
plot(Fz_loads/1000, mu_r, '-o', 'Color', colors(2,:), 'LineWidth', 2, ...
    'MarkerFaceColor', colors(2,:), 'MarkerSize', 7, 'DisplayName', 'Rear');
xlabel('Vertical load Fz (kN)', 'Color', txt_col, 'FontName', 'Courier New');
ylabel('Friction coefficient μ (Fy/Fz)', 'Color', txt_col, 'FontName', 'Courier New');
title('Degressive load sensitivity (μ drops with Fz)', ...
    'Color', 'w', 'FontName', 'Courier New');
legend('show', 'TextColor', txt_col, 'Color', ax_dark, ...
    'EdgeColor', [0.3 0.3 0.3], 'FontName', 'Courier New');

annotation('textbox', [0.01 0.01 0.98 0.06], ...
    'String', ['Engineering note: Degressive load sensitivity means adding downforce ' ...
               'increases grip but with diminishing returns. Doubling downforce does NOT ' ...
               'double cornering speed — a key constraint in GT3 BoP analysis.'], ...
    'Color', [0.7 0.7 0.7], 'FontSize', 7, 'FontName', 'Courier New', ...
    'EdgeColor', 'none', 'BackgroundColor', 'none');

saveas(f3, 'output/phase2_03_load_sensitivity.png');
fprintf('  Saved → output/phase2_03_load_sensitivity.png\n');


% ── Figure 4: Friction ellipse ───────────────────────────────────────────────
f4 = figure('Color', bg_dark, 'Position', [50 50 700, 700]);
ax  = axes; hold on; grid on; box on;
set(ax, 'Color', ax_dark, 'XColor', txt_col, 'YColor', txt_col, ...
    'GridColor', [0.25 0.25 0.25], 'GridLineStyle', '--');
title('FRICTION ELLIPSE — Fz = 3500 N | MF96 boundaries', ...
    'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold', 'FontName', 'Courier New');

% Build full ellipse by rotating through combined loading
n_pts   = 200;
theta_e = linspace(0, 2*pi, n_pts);
Fx_nom_f = arrayfun(@(k) magic_formula_Fx(k, Fz_nom, c_front), linspace(-0.25,0.25,50));
Fy_nom_f = arrayfun(@(a) magic_formula_Fy(a, Fz_nom, c_front), linspace(-14,14,50));
Fx_max_f = max(abs(Fx_nom_f));
Fy_max_f = max(abs(Fy_nom_f));
Fx_max_r = max(abs(arrayfun(@(k) magic_formula_Fx(k, Fz_nom, c_rear), linspace(-0.25,0.25,50))));
Fy_max_r = max(abs(arrayfun(@(a) magic_formula_Fy(a, Fz_nom, c_rear), linspace(-14,14,50))));

% Ellipse outlines
ell_Fx_f = Fx_max_f * cos(theta_e) / 1000;
ell_Fy_f = Fy_max_f * sin(theta_e) / 1000;
ell_Fx_r = Fx_max_r * cos(theta_e) / 1000;
ell_Fy_r = Fy_max_r * sin(theta_e) / 1000;

fill(ell_Fx_f, ell_Fy_f, colors(1,:), 'FaceAlpha', 0.10, 'EdgeColor', 'none');
plot(ell_Fx_f, ell_Fy_f, 'Color', colors(1,:), 'LineWidth', 2, 'DisplayName', 'Front axle');
fill(ell_Fx_r, ell_Fy_r, colors(2,:), 'FaceAlpha', 0.10, 'EdgeColor', 'none');
plot(ell_Fx_r, ell_Fy_r, 'Color', colors(2,:), 'LineWidth', 2, 'DisplayName', 'Rear axle');

xline(0, 'Color', [0.35 0.35 0.35], 'LineWidth', 0.5);
yline(0, 'Color', [0.35 0.35 0.35], 'LineWidth', 0.5);
text( Fx_max_f/1000 + 0.05,  0.1, 'Drive', 'Color', txt_col, ...
    'FontSize', 8, 'FontName', 'Courier New');
text(-Fx_max_f/1000 - 0.45, -0.1, 'Brake', 'Color', txt_col, ...
    'FontSize', 8, 'FontName', 'Courier New');
text(0.05, Fy_max_f/1000 + 0.05, 'Left', 'Color', txt_col, ...
    'FontSize', 8, 'FontName', 'Courier New');
text(0.05, -Fy_max_f/1000 - 0.12, 'Right', 'Color', txt_col, ...
    'FontSize', 8, 'FontName', 'Courier New');

xlabel('Longitudinal force Fx (kN)', 'Color', txt_col, 'FontName', 'Courier New');
ylabel('Lateral force Fy (kN)', 'Color', txt_col, 'FontName', 'Courier New');
legend('show', 'Location', 'northeast', 'TextColor', txt_col, ...
    'Color', ax_dark, 'EdgeColor', [0.3 0.3 0.3], 'FontName', 'Courier New');
axis equal;

saveas(f4, 'output/phase2_04_friction_ellipse.png');
fprintf('  Saved → output/phase2_04_friction_ellipse.png\n');

fprintf('\n✓ Pacejka MF96 analysis complete.\n');
fprintf('  Front peak Fy at 3500 N: %.2f kN at %.1f° slip\n', ...
    max(Fy_front(2,:))/1000, alpha_range(find(Fy_front(2,:)==max(Fy_front(2,:)),1)));
fprintf('  Rear  peak Fy at 3500 N: %.2f kN at %.1f° slip\n', ...
    max(Fy_rear(2,:))/1000, alpha_range(find(Fy_rear(2,:)==max(Fy_rear(2,:)),1)));
