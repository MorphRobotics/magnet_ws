%% dipole_training_cosserat.m
% Data acquisition + inverse MLP using Dipole Field + Cosserat Rod Theory
clear; clc; close all;

%% 1. MSCR + Material Parameters (Cosserat Specific)
L_mscr = 0.06;          
N_pts  = 50;            
base_T = eye(4);        

% Material properties (Silicone/Nitinol example)
E = 1e6;                % Young's Modulus [Pa]
G = E/(2*(1+0.45));     % Shear Modulus [Pa]
radius = 0.001;         % MSCR cross-section radius [m]
A = pi*radius^2;
I = (pi*radius^4)/4;
J = 2*I;
params.K = diag([E*I, E*I, G*J]); % Stiffness matrix (bending/torsion)

% Dipole parameters
params.mu0   = 4*pi*1e-7;      
m_strength   = 0.5;            
params.m_vec = [0; 0; m_strength]; % Magnet moment vector

%% 2. Magnet pose sweep
radii   = linspace(0.04, 0.10, 8); 
angles  = linspace(0, 2*pi, 24);    
heights = linspace(0.02, 0.06, 30);  

nR = numel(radii);   % Number of radii
nA = numel(angles);  % Number of angles
nH = numel(heights); % Number of heights
N_samples = nR * nA * nH;

%% 3. Reference (straight) tip position
% Cosserat with zero field = straight rod
[P_ref, ~] = mscr_cosserat_rod([0;0;0], L_mscr, N_pts, base_T, params);
tip_ref = P_ref(:, end);

%% 4. Main sweep: generate data
features = zeros(N_samples, 3);   
tip_pos  = zeros(N_samples, 3);   
tip_def  = zeros(N_samples, 3);   
sample_idx = 0;

fprintf('Simulating %d samples using Cosserat Rod Theory...\n', N_samples);
for ih = 1:nH
    z_m = heights(ih);
    for ir = 1:nR
        R = radii(ir);
        for ia = 1:nA
            theta = angles(ia);
            sample_idx = sample_idx + 1;
            
            % 5.1 Magnet position
            pm = [R*cos(theta); R*sin(theta); z_m];
            
            % 5.2 Magnetic field at MSCR base for MLP features
            B_base = dipoleField([0;0;0], pm, eye(3), params);
            Bmag = norm(B_base);
            az_B = atan2(B_base(2), B_base(1));
            el_B = atan2(B_base(3), sqrt(B_base(1)^2+B_base(2)^2));
            
            % 5.3 Generate Cosserat centerline
            [P_curr, ~] = mscr_cosserat_rod(pm, L_mscr, N_pts, base_T, params);
            tip = P_curr(:, end);
            
            % 5.4 Store features and outputs
            features(sample_idx, :) = [Bmag, az_B, el_B];
            tip_pos(sample_idx, :)  = tip';
            tip_def(sample_idx, :)  = (tip - tip_ref)';
        end
    end
end

%% 6. Visualization: tip deflection magnitude (mid-height slice)
tip_def_mag = sqrt(sum(tip_def.^2, 2)); 
midH_idx = ceil(nH/2);
z_mid    = heights(midH_idx);

% Reshape for plotting
start_idx = (midH_idx-1)*nR*nA + 1;
end_idx   = midH_idx*nR*nA;
slice_def = reshape(tip_def_mag(start_idx:end_idx), [nA, nR])';

[TH, RR] = meshgrid(angles, radii);
figure; surf(TH, RR, slice_def);
shading interp; colorbar;
xlabel('\theta (rad)'); ylabel('Radius R (m)'); zlabel('||\Delta p_{tip}|| (m)');
title(sprintf('Cosserat Tip Deflection (z_m = %.3f m)', z_mid));
view(135,30);


%% 7. Example shape for most deformed sample
[~, ex_idx] = max(tip_def_mag);

% Use the helper function to get the magnet position for the max deflection index
pm_ex = pm_ex_calc(ex_idx, radii, angles, heights, nR, nA); 

% Re-run the Cosserat solver for this specific pose to get the full centerline
[P_ex, ~] = mscr_cosserat_rod(pm_ex, L_mscr, N_pts, base_T, params);

figure; hold on;
plot3(P_ref(1,:), P_ref(2,:), P_ref(3,:), 'k--','LineWidth',1.5);
plot3(P_ex(1,:),  P_ex(2,:),  P_ex(3,:),  'r-','LineWidth',2);
axis equal; grid on;
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
legend('Straight MSCR','Cosserat Deformed','Location','best');
title(sprintf('Most Deformed Cosserat Shape (||\\Delta p|| = %.4f m)', tip_def_mag(ex_idx)));
view(135,30);

% Dynamic Axis Adjustment
allP = [P_ref P_ex];
xlim([min(allP(1,:))-0.01 max(allP(1,:))+0.01]);
ylim([min(allP(2,:))-0.01 max(allP(2,:))+0.01]);
zlim([0 L_mscr + 0.01]);
%% 8. Normalize and pack dataset (raw + normalized)

% --- Raw copies ---
features_raw = features;   % [|B| (T), az (rad), el (rad)]
tip_def_raw  = tip_def;    % [dx, dy, dz] in meters

% --- Feature normalization (z-score per column) ---
mu_feat  = mean(features_raw, 1);
sig_feat = std(features_raw, 0, 1) + 1e-8;
features_norm = (features_raw - mu_feat) ./ sig_feat;

% --- Tip-deflection normalization (z-score per column) ---
mu_tip  = mean(tip_def_raw, 1);
sig_tip = std(tip_def_raw, 0, 1) + 1e-8;
tip_def_norm = (tip_def_raw - mu_tip) ./ sig_tip;

% --- Base dataset struct ---
normInfo = struct();
normInfo.mu_feat  = mu_feat;
normInfo.sig_feat = sig_feat;
normInfo.mu_tip   = mu_tip;
normInfo.sig_tip  = sig_tip;

dataset = struct();
dataset.features_raw  = features_raw;    % unnormalized [|B|, az, el]
dataset.features_norm = features_norm;   % normalized
dataset.tip_def_raw   = tip_def_raw;     % unnormalized Δp_tip
dataset.tip_def_norm  = tip_def_norm;    % normalized Δp_tip

dataset.tip_pos  = tip_pos;              % absolute tip positions
dataset.radii    = radii;
dataset.angles   = angles;
dataset.heights  = heights;
dataset.params   = params;
dataset.L_mscr   = L_mscr;
dataset.base_T   = base_T;
dataset.norm     = normInfo;

%% 9. Train inverse MLP: Δp_tip  ->  [|B|, az, el]

% Inputs: normalized tip deflection
X_all = tip_def_norm;          % [N x 3]
% Targets: normalized [|B|, az, el]
Y_all = features_norm(:,1:3);  % [N x 3]

Ntot = size(X_all,1);
idx  = randperm(Ntot);

% 70/15/15 split
Ntrain = round(0.7 * Ntot);
Nval   = round(0.15 * Ntot);
Ntest  = Ntot - Ntrain - Nval;

id_tr = idx(1:Ntrain);
id_va = idx(Ntrain+1 : Ntrain+Nval);
id_te = idx(Ntrain+Nval+1 : end);

Xtr = X_all(id_tr,:);
Ytr = Y_all(id_tr,:);

Xva = X_all(id_va,:);
Yva = Y_all(id_va,:);

Xte = X_all(id_te,:);
Yte = Y_all(id_te,:);

layers = [
    featureInputLayer(3, "Name", "input")            % Input: Δp_tip (dx, dy, dz)
    
    fullyConnectedLayer(128, "Name", "fc1")          % Layer 1
    reluLayer("Name", "relu1")
    
    fullyConnectedLayer(128, "Name", "fc2")          % Layer 2
    reluLayer("Name", "relu2")
    
    fullyConnectedLayer(128, "Name", "fc3")          % Layer 3
    reluLayer("Name", "relu3")
    
    fullyConnectedLayer(128, "Name", "fc4")          % Layer 4
    reluLayer("Name", "relu4")
    
    fullyConnectedLayer(64, "Name", "fc5")           % Layer 5 (Compression)
    reluLayer("Name", "relu5")
    
    fullyConnectedLayer(64, "Name", "fc6")           % Layer 6
    reluLayer("Name", "relu6")
    
    fullyConnectedLayer(3, "Name", "fc_out")         % Layer 7 (Output: |B|, az, el)
    regressionLayer("Name", "regressionoutput")
    
];% Network architecture: 2 hidden layers, 60 neurons each

options = trainingOptions("adam", ...
    "MaxEpochs",        500, ...
    "MiniBatchSize",    256, ...
    "InitialLearnRate", 1e-3, ...
    "Shuffle",          "every-epoch", ...
    "ValidationData",   {Xva, Yva}, ...
    "ValidationFrequency", 50, ...
    "ValidationPatience", 30, ...
    "Verbose",          false, ...
    "Plots",            "training-progress");

[inv_net, trainInfo] = trainNetwork(Xtr, Ytr, layers, options);

% Store inverse model normalization
invNorm = struct();
invNorm.mu_in   = mu_tip;         % for Δp_tip normalization
invNorm.sig_in  = sig_tip;
invNorm.mu_out  = mu_feat(1:3);   % [|B|, az, el] for unnormalization
invNorm.sig_out = sig_feat(1:3);

dataset.inv_net  = inv_net;
dataset.inv_norm = invNorm;

%% 10. Evaluation & interpretability

% a) Loss curves (training + validation)
figure; hold on;
iters = 1:numel(trainInfo.TrainingLoss);
plot(iters, trainInfo.TrainingLoss, 'b', 'LineWidth', 1.5);

if isfield(trainInfo,"ValidationLoss")
    valMask = ~isnan(trainInfo.ValidationLoss);
    if any(valMask)
        plot(iters(valMask), trainInfo.ValidationLoss(valMask), ...
             'r.-','LineWidth',1.5,'MarkerSize',8);
    end
end

grid on;
xlabel('Iteration');
ylabel('Loss (MSE)');
legend('Training','Validation','Location','northeast');
title('Inverse MLP training/validation loss');

% b) Predict on test set (normalized)
Yte_pred_norm = predict(inv_net, Xte);   % [Ntest x 3]

% Unnormalize: [|B|, az, el]
Bmag_true = Yte(:,1) .* sig_feat(1) + mu_feat(1);
az_true   = Yte(:,2) .* sig_feat(2) + mu_feat(2);
el_true   = Yte(:,3) .* sig_feat(3) + mu_feat(3);

Bmag_pred = Yte_pred_norm(:,1) .* sig_feat(1) + mu_feat(1);
az_pred   = Yte_pred_norm(:,2) .* sig_feat(2) + mu_feat(2);
el_pred   = Yte_pred_norm(:,3) .* sig_feat(3) + mu_feat(3);

% c) Angle errors (wrap-aware) + |B| errors
angleDiff = @(a,b) atan2(sin(a-b), cos(a-b));

err_az = angleDiff(az_pred, az_true);   % rad
err_el = angleDiff(el_pred, el_true);   % rad
err_B  = Bmag_pred - Bmag_true;         % Tesla

rmse_az = sqrt(mean(err_az.^2));
rmse_el = sqrt(mean(err_el.^2));
rmse_B  = sqrt(mean(err_B.^2));

mae_az  = mean(abs(err_az));
mae_el  = mean(abs(err_el));
mae_B   = mean(abs(err_B));

fprintf('\nInverse MLP test errors (wrapped angles):\n');
fprintf('  |B|   RMSE = %.4g T,  MAE = %.4g T\n', rmse_B, mae_B);
fprintf('  Azim  RMSE = %.3f rad (%.1f deg),  MAE = %.3f rad (%.1f deg)\n', ...
    rmse_az, rmse_az*180/pi, mae_az, mae_az*180/pi);
fprintf('  Elev  RMSE = %.3f rad (%.1f deg),  MAE = %.3f rad (%.1f deg)\n\n', ...
    rmse_el, rmse_el*180/pi, mae_el, mae_el*180/pi);

% d) Regression plots (true vs predicted)
figure;
subplot(1,3,1); hold on; grid on; axis equal;
scatter(Bmag_true, Bmag_pred, 10, 'filled');
minv = min([Bmag_true; Bmag_pred]);
maxv = max([Bmag_true; Bmag_pred]);
plot([minv maxv],[minv maxv],'k--','LineWidth',1.5);
xlabel('True |B| [T]'); ylabel('Predicted |B| [T]');
title('|B| regression');

subplot(1,3,2); hold on; grid on; axis equal;
scatter(az_true, az_pred, 10, 'filled');
minv = min([az_true; az_pred]);
maxv = max([az_true; az_pred]);
plot([minv maxv],[minv maxv],'k--','LineWidth',1.5);
xlabel('True azimuth (rad)');
ylabel('Predicted azimuth (rad)');
title('Azimuth regression');

subplot(1,3,3); hold on; grid on; axis equal;
scatter(el_true, el_pred, 10, 'filled');
minv = min([el_true; el_pred]);
maxv = max([el_true; el_pred]);
plot([minv maxv],[minv maxv],'k--','LineWidth',1.5);
xlabel('True elevation (rad)');
ylabel('Predicted elevation (rad)');
title('Elevation regression');

sgtitle('Inverse MLP: \Delta p_{tip} \rightarrow [|B|, az, el]');

% e) Histograms of angular errors
figure;
subplot(1,3,1);
histogram(err_B, 30);
xlabel('|B| error [T]'); ylabel('Count');
title('|B| prediction error'); grid on;

subplot(1,3,2);
histogram(err_az*180/pi, 30);
xlabel('Azimuth error [deg]'); ylabel('Count');
title('Azimuth prediction error'); grid on;

subplot(1,3,3);
histogram(err_el*180/pi, 30);
xlabel('Elevation error [deg]'); ylabel('Count');
title('Elevation prediction error'); grid on;

sgtitle('Inverse MLP error distributions');

%% f) Error vs desired tip-deflection magnitude
% --- Existing code ---
dTip_test = Xte .* sig_tip + mu_tip;      % unnormalize Δp_tip
dTip_mag  = sqrt(sum(dTip_test.^2,2));    % magnitude [m]
dTip_mm   = dTip_mag * 1000;              % magnitude [mm] for plotting

% --- Unnormalize |B| predictions ---
Bmag_true = Yte(:,1) .* sig_feat(1) + mu_feat(1);         % True |B| [T]
Bmag_pred = Yte_pred_norm(:,1) .* sig_feat(1) + mu_feat(1); % Predicted |B| [T]
err_B     = Bmag_pred - Bmag_true;                       % Absolute error [T]

figure;

% Azimuth error
subplot(2,2,1);
scatter(dTip_mm, abs(err_az)*180/pi, 15, 'filled');
grid on;
xlabel('||\Delta p_{tip}|| [mm]');
ylabel('|Azimuth error| [deg]');
title('Azimuth error vs tip deflection');

% Elevation error
subplot(2,2,2);
scatter(dTip_mm, abs(err_el)*180/pi, 15, 'filled');
grid on;
xlabel('||\Delta p_{tip}|| [mm]');
ylabel('|Elevation error| [deg]');
title('Elevation error vs tip deflection');

% --- New: |B| prediction error ---
subplot(2,2,[3 4]);
scatter(dTip_mm, abs(err_B)*1e3, 15, 'filled');  % Convert Tesla → milliTesla
grid on;
xlabel('||\Delta p_{tip}|| [mm]');
ylabel('|B error| [mT]');
title('Magnetic field |B| prediction error vs tip deflection');

sgtitle('Inverse MLP: Error vs desired MSCR tip deflection');


%% 11. Save everything
save('mscr_dipole_cc_dataset.mat','dataset');
fprintf('Dataset (raw + normalized + inverse MLP) saved to mscr_dipole_cc_dataset.mat\n');


%% ===========================
%   Local Helper Functions
% ===========================

function [P, R] = mscr_cosserat_rod(pm, L, N, base_T, params)
    % Boundary condition at tip (Magnetic Wrench)
    B_tip = dipoleField(base_T(1:3,4) + [0;0;L], pm, eye(3), params);
    torque_tip = cross(params.m_vec, B_tip); 
    
    % Map torque to curvature via stiffness K
    kappa_vec = params.K \ torque_tip; 
    phi = atan2(kappa_vec(2), kappa_vec(1));
    k_mag = norm(kappa_vec(1:2));

    % Use CC-like geometry driven by Cosserat Physics
    P = mscr_constant_curvature_internal(k_mag, phi, L, N, base_T);
    R = []; 
end

function P = mscr_constant_curvature_internal(kappa, phi, L, N, base_T)
    s = linspace(0, L, N);
    if abs(kappa) < 1e-6
        P_local = [zeros(2,N); s];
    else
        rad = 1/kappa;
        P_local = [rad*(1-cos(kappa*s)); zeros(1,N); rad*sin(kappa*s)];
    end
    Rz = [cos(phi), -sin(phi), 0; sin(phi), cos(phi), 0; 0, 0, 1];
    P = base_T(1:3,1:3) * (Rz * P_local) + base_T(1:3,4);
end

function B = dipoleField(pc, pm, Rm, params)
    m = Rm * params.m_vec; 
    r = pc - pm;
    rnorm = norm(r) + eps;
    rhat = r / rnorm;
    B = params.mu0/(4*pi*rnorm^3) * ( 3*(m'*rhat)*rhat - m );
end

function pm = pm_ex_calc(idx, radii, angles, heights, nR, nA)
    % Helper to retrieve pm from linear index for Section 7 visualization
    ia = mod(idx-1, nA) + 1;
    ir = mod(floor((idx-1)/nA), nR) + 1;
    ih = floor((idx-1)/(nR*nA)) + 1;
    pm = [radii(ir)*cos(angles(ia)); radii(ir)*sin(angles(ia)); heights(ih)];
end

%% 12. Export to ONNX for Deployment (Fixed)
% 1. Prepare naming and dummy input
onnxFileName = 'mscr_inverse_model.onnx'; % Ensure this is a char array
inputSize = [3 1]; % The MLP expects a 3-element column vector based on your layers
dummyInput = ones(inputSize, 'single');

% 2. Export command with explicit argument ordering
% Syntax: exportONNXNetwork(net, filename, 'OpsetVersion', 12)
try
    exportONNXNetwork(inv_net, onnxFileName, 'OpsetVersion', 12);
    fprintf('Successfully exported ONNX model: %s\n', onnxFileName);
catch ME
    fprintf('Export failed. Error: %s\n', ME.message);
    fprintf('Trying alternative syntax...\n');
    % Fallback syntax for older/different toolbox versions
    exportONNXNetwork(inv_net, onnxFileName);
end

% 3. Export scaling parameters for Python/ROS 2
% Format: 
% Row 1: mu_in (tip deflections)
% Row 2: sig_in
% Row 3: mu_out (B, az, el)
% Row 4: sig_out
normParams = [invNorm.mu_in; invNorm.sig_in; invNorm.mu_out; invNorm.sig_out];
writematrix(normParams, 'scaling_params.csv');

fprintf('Scaling parameters saved to scaling_params.csv\n');

%% 13. Export Numerical Data and Figures (Corrected)

% --- Numerical Data Export (Same as before) ---
loss_data = table((1:numel(trainInfo.TrainingLoss))', trainInfo.TrainingLoss', 'VariableNames', {'Iteration', 'Training_Loss'});
if isfield(trainInfo, "ValidationLoss"), loss_data.Validation_Loss = trainInfo.ValidationLoss'; end
writetable(loss_data, 'training_progress_data.csv');

regression_results = table(Bmag_true, Bmag_pred, az_true, az_pred, el_true, el_pred, 'VariableNames', {'True_B', 'Pred_B', 'True_Azimuth', 'Pred_Azimuth', 'True_Elevation', 'Pred_Elevation'});
writetable(regression_results, 'regression_analysis_data.csv');

error_data = table(err_B, err_az*180/pi, err_el*180/pi, 'VariableNames', {'B_Error_Tesla', 'Azimuth_Error_Deg', 'Elevation_Error_Deg'});
writetable(error_data, 'error_distribution_data.csv');

% --- Corrected Figure Export ---
% Find the Figure handles
fig_loss = findobj('Type', 'figure', 'Name', 'Inverse MLP training/validation loss');
fig_reg  = findobj('Type', 'figure', 'Name', 'Inverse MLP: \Delta p_{tip} \rightarrow [|B|, az, el]');

% Export Loss Curve
if ~isempty(fig_loss)
    % Target the axes inside the figure for exportgraphics
    exportgraphics(fig_loss.CurrentAxes, 'loss_curve.png', 'Resolution', 300);
    fprintf('Saved loss_curve.png\n');
end

% Export Regression Plots
if ~isempty(fig_reg)
    % Since this figure has subplots, we export the whole figure using saveas
    % or target the tiledlayout if you used one.
    saveas(fig_reg, 'regression_plots.png');
    fprintf('Saved regression_plots.png\n');
end