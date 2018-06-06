clear all;
close all;
addpath '/Users/alexandre.bone/Desktop/2017_NIPS/98_visualization_tools'

%% Parameters.
maxIt = 20000;
totalNumberOfTrajectoryPoints = 1000;

truthPath = '/Users/alexandre.bone/Desktop/2017_NIPS/33_simulation_4sources_psi_easier/';

deformetricaPath = '/Users/alexandre.bone/Workspace/deformetrica_build/utilities/c++/deformetrica-compute-varifold-distance';
refTemplatePath = 'data/ForInitialization_Template.vtk';
varifoldKernelSize = 1;

initialReferenceTime = 1;
varianceReferenceTimePrior = 2;
initialControlPointsPath = 'ForInitialization_ControlPoints_0.1.txt';
initialMomentaPath = 'ForInitialization_Momenta_0.05.txt';
timeShiftVariancePrior = 5;
logAccelerationVariancePrior = 0.5;
noiseVariancePrior = 1;
nbtp = 5;
Lambda = 32;

defKernelSize = 1;

step = 0.3;
[X, Y] = meshgrid(-3:step:3);
gridsize = size(X,1);


%% Load reference template.
[refTempPts,refTempEdges] = VTKPolyDataReader(refTemplatePath);
nbpts = size(refTempPts,1);

%% Load the true parameters.
truthTemplatePath = [truthPath,'template_shape.vtk'];
[TemplatePts,TemplateEdges] = VTKPolyDataReader(truthTemplatePath);

cp = load([truthPath,'template_cp.txt']);
nbcp = size(cp, 1);

mom = load([truthPath,'output/SimulatedData__Parameters__Momenta.txt']);
modmat = load([truthPath,'output/SimulatedData__Parameters__ProjectedModulationMatrix.txt']);
nbSources = size(modmat,2);

referenceTime_truth = 0;
timeShiftVariance_truth = 1^2;
logAccelerationVariance_truth = 0.1^2;
noiseVariance_truth = 0^2;

% Template case.
cmd = [deformetricaPath, ' ', num2str(2), ' ', num2str(varifoldKernelSize), ' ', refTemplatePath, ' ', truthPath, 'template_shape.vtk'];
[~, out] = system(cmd);
templateDataVarifoldDistanceToRef_truth = str2double(out);

% Momenta case.
CP = cp;
MOM = mom;
vfield_MOM_truth = cell(2,1);
vfield_MOM_truth{1} = zeros(gridsize);
vfield_MOM_truth{2} = zeros(gridsize);
for i=1:gridsize
    for j=1:gridsize
        for p=1:nbcp
            dsq = (CP(p,1)-X(i,j))^2 + (CP(p,2)-Y(i,j))^2;
            ker = exp(-dsq / defKernelSize^2);
            vfield_MOM_truth{1}(i,j) = vfield_MOM_truth{1}(i,j) + ker*MOM(p,1); 
            vfield_MOM_truth{2}(i,j) = vfield_MOM_truth{2}(i,j) + ker*MOM(p,2); 
        end
    end
end 
momVelocityFieldNorm_truth = norm(vfield_MOM_truth{1}(:)) + norm(vfield_MOM_truth{2}(:));

% Modulation matrix case. Solution 1.
modVelocityFieldNorm_truth = cell(1,nbSources);
% for m=1:nbSources
%     modVelocityFieldNorm_truth{m} = zeros(nbSamples, 1);
% end

vfields_MOD_truth = cell(nbSources,2);
for m=1:nbSources
    CP = cp;
    MOM = reshape(modmat(:, m), [2, nbcp])';
    vfields_MOD_truth{m,1} = zeros(gridsize);
    vfields_MOD_truth{m,2} = zeros(gridsize);
    for i=1:gridsize
        for j=1:gridsize
            for p=1:nbcp
                dsq = (CP(p,1)-X(i,j))^2 + (CP(p,2)-Y(i,j))^2;
                ker = exp(-dsq / defKernelSize^2);
                vfields_MOD_truth{m,1}(i,j) = vfields_MOD_truth{m,1}(i,j) + ker*MOM(p,1); 
                vfields_MOD_truth{m,2}(i,j) = vfields_MOD_truth{m,2}(i,j) + ker*MOM(p,2); 
            end
        end
    end 
    modVelocityFieldNorm_truth{m} = norm(vfields_MOD_truth{m,1}(:)) + norm(vfields_MOD_truth{m,2}(:));
end

% Modulation matrix case. Solution 2.
K_truth = zeros(nbcp);
for i=1:nbcp
    for j=1:(i-1)
       dsq = (cp(i,1)-cp(j,1))^2 + (cp(i,2)-cp(j,2))^2;
       ker = exp(-dsq / defKernelSize^2);
       K_truth(i, j) = ker;
       K_truth(j, i) = ker;
    end
    K_truth(i,i) = 1;
end
ortho_truth = zeros(nbcp*2);
orthoResidual_truth = zeros(nbcp*2,1);
for k=1:nbcp*2
    e = zeros(nbcp*2,1);
    e(k) = 1;
    e_mom = reshape(e, [2, nbcp])';
    Km = K_truth * e_mom;
    
    projection = zeros(nbcp*2,1);
    for m=1:nbSources
        c = reshape(modmat(:,m), [2, nbcp])';
        cKm = sum(sum(c .* Km));
        projection = projection + cKm * modmat(:,m); 
    end
    ortho_truth(k,:) = e - projection;
    orthoResidual_truth(k) = norm(ortho_truth(k,:));
end

% Modulation matrix case. Solution 3.
W_ref = modmat;
Kdouble_truth = kron(K_truth, eye(2));
Kaug_ref = Kdouble_truth;
G_ref = W_ref' * Kaug_ref * W_ref;
P_ref = sqrtm(Kaug_ref) * (W_ref / G_ref) * W_ref' * sqrtm(Kaug_ref);
Pnorm_ref = norm(P_ref);

%% Compute the empirical truth. 
% Template.
templateData_statTruth = (defKernelSize^2 * TemplatePts + (defKernelSize/50)^2 * refTempPts) / (defKernelSize^2 + (defKernelSize/50)^2);
lines = zeros(nbpts,2);
for m=1:nbpts
    lines(m,1)=m;
    lines(m,2)=m+1;
end
lines(nbpts,2)=1;
VTKPolyDataWriter(templateData_statTruth, lines, [], [], [], 'template_statTruth.vtk')
statTruthTemplatePath = 'template_statTruth.vtk';

cmd = [deformetricaPath, ' ', num2str(2), ' ', num2str(varifoldKernelSize), ' ', statTruthTemplatePath, ' ',  refTemplatePath];
[~, out] = system(cmd);
templateDataVarifoldDistanceToRef_statTruth(k) = str2double(out);

% Control points. 
initialControlPoints = load(['data/', initialControlPointsPath]);
controlPoints_statTruth = (defKernelSize^2 * cp + (1/50)^2 * initialControlPoints) / (defKernelSize^2 + (defKernelSize/50)^2);

% Momenta. 
initialMomenta = load(['data/', initialMomentaPath]);
momenta_statTruth = (defKernelSize^2 * mom + (1/50)^2 * initialMomenta) / (defKernelSize^2 + (defKernelSize/50)^2);

% Geodesic velocity field. 
vfield_MOM_statTruth = cell(2,1);
vfield_MOM_statTruth{1} = zeros(gridsize);
vfield_MOM_statTruth{2} = zeros(gridsize);
for i=1:gridsize
    for j=1:gridsize
        for p=1:nbcp
            dsq = (controlPoints_statTruth(p,1)-X(i,j))^2 + (controlPoints_statTruth(p,2)-Y(i,j))^2;
            ker = exp(-dsq / defKernelSize^2);
            vfield_MOM_statTruth{1}(i,j) = vfield_MOM_statTruth{1}(i,j) + ker * momenta_statTruth(p,1); 
            vfield_MOM_statTruth{2}(i,j) = vfield_MOM_statTruth{2}(i,j) + ker * momenta_statTruth(p,2); 
        end
    end
end 
momVelocityFieldNorm_statTruth = norm(vfield_MOM_statTruth{1}(:)) + norm(vfield_MOM_statTruth{2}(:));

% Exp-parallelization velocity fields. 
modulationMatrix_statTruth = defKernelSize^2 * modmat / (defKernelSize^2 + (defKernelSize/50)^2);

vfields_MOD_statTruth = cell(nbSources,2);
for m=1:nbSources
    MOM = reshape(modulationMatrix_statTruth(:, m), [2, nbcp])';
    vfields_MOD_statTruth{m,1} = zeros(gridsize);
    vfields_MOD_statTruth{m,2} = zeros(gridsize);
    for i=1:gridsize
        for j=1:gridsize
            for p=1:nbcp
                dsq = (controlPoints_statTruth(p,1)-X(i,j))^2 + (controlPoints_statTruth(p,2)-Y(i,j))^2;
                ker = exp(-dsq / defKernelSize^2);
                vfields_MOD_statTruth{m,1}(i,j) = vfields_MOD_statTruth{m,1}(i,j) + ker*MOM(p,1); 
                vfields_MOD_statTruth{m,2}(i,j) = vfields_MOD_statTruth{m,2}(i,j) + ker*MOM(p,2); 
            end
        end
    end 
end

% Reference time. 
referenceTime_statTruth = (varianceReferenceTimePrior * referenceTime_truth + varianceReferenceTimePrior/50)^2 * initialReferenceTime / (varianceReferenceTimePrior + (varianceReferenceTimePrior/50)^2);

% Time-shift variance. 
timeShifts_truth = load([truthPath,'output/SimulatedData__Parameters__TimeShifts.txt']);
nbSubjects = size(timeShifts_truth,1);
S6 = sum(timeShifts_truth.^2);
timeShiftVariance_statTruth = (S6 + 1 * timeShiftVariancePrior) / (nbSubjects + 1);

% Log-acceleration variance. 
logAcceleration_truth = load([truthPath,'output/SimulatedData__Parameters__LogAccelerations.txt']);
S7 = sum(logAcceleration_truth.^2);
logAccelerationVariance_statTruth = (S7 + 1 * logAccelerationVariancePrior) / (nbSubjects + 1);

% Noise variance. 
noiseVariance_statTruth = 1 * noiseVariancePrior / (nbSubjects * nbtp * Lambda + 1);

%% Load the parameters trajectory matrix.
pt = load('output/LongitudinalAtlas__ParametersTrajectory.txt');
nbSamples = size(pt,1);
saveModelParametersEveryNIters = floor(maxIt/1000);

dim = 2;
controlPoints = pt(:,1:nbcp*dim);
logAccelerationVariance = pt(:,1+nbcp*dim);
modulationMatrix = pt(:,2+nbcp*dim:2+nbcp*dim+dim*nbcp*nbSources-1);
momenta = pt(:,2+nbcp*dim+dim*nbcp*nbSources:2+nbcp*dim+dim*nbcp*nbSources+nbcp*dim-1);
noiseVariance = pt(:,2+2*nbcp*dim+dim*nbcp*nbSources);
referenceTime = pt(:,23+dim*nbcp*nbSources);
templateData = pt(:,24+dim*nbcp*nbSources:24+dim*nbcp*nbSources+2*nbpts-1);
timeShiftVariance = pt(:,24+dim*nbcp*nbSources+2*nbpts);

%% Pre-process the parameters trajectory matrix
% Initialization. 
cpDistanceToTruth = zeros(nbSamples,1);
cpRepojectionDistanceToTruth = zeros(nbSamples,1);

momVelocityFieldNorm = zeros(nbSamples,1);
momVelocityFieldDistanceToStatTruth = zeros(nbSamples,1);

modVelocityFieldNorm = cell(1,nbSources);
modVelocityFieldDistanceToTruth = cell(1,nbSources);
for m=1:nbSources
    modVelocityFieldNorm{m} = zeros(nbSamples, 1);
    modVelocityFieldDistanceToTruth{m} = zeros(nbSamples, 1);
end
modSubspaceDistanceToTruth = zeros(nbSamples,1);
modDetDistanceToTruth = zeros(nbSamples,1);
modProjectorDistanceToTruth = zeros(nbSamples,1);

% % Find the best independent component ordering. 
% orderings = perms([1 2 3 4]);
% nbOrders = size(orderings,1);
% signs = [-1 -1 -1 -1 ; -1 -1 -1 1 ; -1 -1 1 -1 ; -1 -1 1 1 ; -1 1 -1 -1 ; -1 1 -1 1 ; -1 1 1 -1 ; -1 1 1 1 ; 1 -1 -1 -1 ;  1 -1 -1 1 ; 1 -1 1 -1 ; 1 -1 1 1 ; 1 1 -1 -1 ; 1 1 -1 1; 1 1 1 -1 ; 1 1 1 1];
% nbSigns = size(signs,1);
% cost = zeros(nbOrders,nbSigns);
% 
% CP = reshape(controlPoints(nbSamples,:), [2, nbcp])';
% MOM = reshape(momenta(nbSamples,:), [2, nbcp])';
% 
% % Pre-compute the metric matrix K. 
% K = zeros(nbcp, nbcp);
% for i=1:nbcp
%     for j=1:(i-1)
%        dsq = (CP(i,1)-CP(j,1))^2 + (CP(i,2)-CP(j,2))^2;
%        ker = exp(-dsq / defKernelSize^2);
%        K(i, j) = ker;
%        K(j, i) = ker;
%     end
%     K(i,i) = 1;
% end
% vfields = cell(nbSources,2);
% modmatProjected = zeros(nbcp*2, nbSources);
% for m=1:nbSources
%     % Project on the hyperspace defined by v = (CP, MOM).
%     MOD = reshape(modulationMatrix(nbSamples, (1+(m-1)*2*nbcp):m*2*nbcp), [2, nbcp])';
%     Km = K * MOM;
%     mKm = sum(sum(MOM .* Km));
%     cKm = sum(sum(MOD .* Km));
%     MOD = MOD - cKm * MOM / mKm;  
%     
%     aux = MOD';
%     modmatProjected(:,m) = MOD(:);
%     
%     % Compute the velocity field. 
%     vfields{m,1} = zeros(gridsize);
%     vfields{m,2} = zeros(gridsize);
%     for i=1:gridsize
%         for j=1:gridsize
%             for p=1:nbcp
%                 dsq = (CP(p,1)-X(i,j))^2 + (CP(p,2)-Y(i,j))^2;
%                 ker = exp(-dsq / defKernelSize^2);
%                 vfields{m,1}(i,j) = vfields{m,1}(i,j) + ker*MOD(p,1); 
%                 vfields{m,2}(i,j) = vfields{m,2}(i,j) + ker*MOD(p,2); 
%             end
%         end
%     end 
% end
% for o=1:nbOrders
%     for s=1:nbSigns
%         order = orderings(o,:);
%         sign = signs(s,:);
%         
%         diff_1 = sign(m) * vfields{order(m),1} - vfields_MOD_truth{m,1};
%         diff_2 = sign(m) * vfields{order(m),2} - vfields_MOD_truth{m,2};
%         cost(o,s) = norm(diff_1(:)) + norm(diff_2(:));
%     end
% end
% [~,idx] = min(cost(:));
% [idx_o, idx_s] = ind2sub(size(cost),idx);
% order = orderings(idx_o, :);
% sign = signs(idx_s, :);
order=1:nbSources;
sign=ones(1,nbSources);

%% Loop over all sample points. 
templateDataVarifoldDistanceToRef = zeros(nbSamples,1);
templateDataVarifoldDistanceToStatTruth = zeros(nbSamples,1);
for k=1:nbSamples
    % Control points. 
    CP_k = reshape(controlPoints(k,:), [2, nbcp])';
    diff = CP_k - cp;
    cpDistanceToTruth(k) = norm(diff(:));
    
    % Momenta-induced velocity field. 
    MOM_k = reshape(momenta(k,:), [2, nbcp])';
    vfield = cell(2,1);
    vfield{1} = zeros(gridsize);
    vfield{2} = zeros(gridsize);
    for i=1:gridsize
        for j=1:gridsize
            for p=1:nbcp
                dsq = (CP_k(p,1)-X(i,j))^2 + (CP_k(p,2)-Y(i,j))^2;
                ker = exp(-dsq / defKernelSize^2);
                vfield{1}(i,j) = vfield{1}(i,j) + ker*MOM_k(p,1); 
                vfield{2}(i,j) = vfield{2}(i,j) + ker*MOM_k(p,2); 
            end
        end
    end 
    momVelocityFieldNorm(k) = norm(vfield{1}(:)) + norm(vfield{2}(:));
    
    diff_1 = vfield{1} - vfield_MOM_statTruth{1};
    diff_2 = vfield{2} - vfield_MOM_statTruth{2};
    momVelocityFieldDistanceToStatTruth(k) = norm(diff_1(:)) + norm(diff_2(:));
    
    % Modulation matrix-induced velocity fields.
    % Pre-computation of the metric matrix K. 
    K_k = zeros(nbcp, nbcp);
    for i=1:nbcp
        for j=1:(i-1)
           dsq = (CP_k(i,1)-CP_k(j,1))^2 + (CP_k(i,2)-CP_k(j,2))^2;
           ker = exp(-dsq / defKernelSize^2);
           K_k(i, j) = ker;
           K_k(j, i) = ker;
        end
        K_k(i,i) = 1;
    end
    vfields = cell(nbSources,2);
    modmatProjected_k = zeros(nbcp*2, nbSources);
    for m=1:nbSources
        MOD = reshape(modulationMatrix(k, (1+(m-1)*2*nbcp):m*2*nbcp), [2, nbcp])';
        
        % Project on the hyperspace defined by v_k = (CP_k, MOM_k). 
        Km = K_k * MOM_k;
        mKm = sum(sum(MOM_k .* Km));
        cKm = sum(sum(MOD .* Km));
        MOD = MOD - cKm * MOM_k / mKm; 
        
        aux = MOD';
        modmatProjected_k(:,m) = aux(:);
        
        % Compute velocity field. 
        vfields{m,1} = zeros(gridsize);
        vfields{m,2} = zeros(gridsize);
        for i=1:gridsize
            for j=1:gridsize
                for p=1:nbcp
                    dsq = (CP_k(p,1)-X(i,j))^2 + (CP_k(p,2)-Y(i,j))^2;
                    ker = exp(-dsq / defKernelSize^2);
                    vfields{m,1}(i,j) = vfields{m,1}(i,j) + ker*MOD(p,1); 
                    vfields{m,2}(i,j) = vfields{m,2}(i,j) + ker*MOD(p,2); 
                end
            end
        end 
        modVelocityFieldNorm{m}(k) = norm(vfields{m,1}(:)) + norm(vfields{m,2}(:));
    end
    for m=1:nbSources
        diff_1 = sign(m) * vfields{order(m),1} - vfields_MOD_truth{m,1};
        diff_2 = sign(m) * vfields{order(m),2} - vfields_MOD_truth{m,2};
        modVelocityFieldDistanceToTruth{m}(k) = norm(diff_1(:)) + norm(diff_2(:));
    end
    
    % Projector-based metric. 
    % Reproject modmatProjected_k onto the true control points.
    modmatReprojected_k = zeros(nbcp * 2, nbSources);
    for m=1:nbSources
        MOD = reshape(modmatProjected_k(:,m), [2, nbcp])';
        vfield_cp = zeros(nbcp,2);
        for i=1:nbcp
            for j=1:nbcp
                dsq = (cp(i,1)-CP_k(j,1))^2 + (cp(i,2)-CP_k(j,2))^2;
                ker = exp(-dsq / defKernelSize^2);
                vfield_cp(i,1) = vfield_cp(i,1) + ker * MOD(j,1); 
                vfield_cp(i,2) = vfield_cp(i,2) + ker * MOD(j,2); 
            end
        end 
        x = K_truth \ vfield_cp;
        aux = x';
        modmatReprojected_k(:,m) = aux(:);
    end
    aux = modmatReprojected_k - modmatProjected_k;
    cpRepojectionDistanceToTruth(k) = norm(aux(:));
    
    % Project a basis of the embedding space to the subspace defined by
    % modmatReprojected_k (and the true control points cp), and find the orthogonal.  
    ortho_k = zeros(nbcp*2);
    orthoResidual_k = zeros(nbcp*2,1);
    for p=1:nbcp*2
        e = zeros(nbcp*2,1);
        e(p) = 1;
        e_mom = reshape(e, [2, nbcp])';
        Km = K_truth * e_mom;

        projection = zeros(nbcp*2,1);
        for m=1:nbSources
            c = reshape(modmatReprojected_k(:,m), [2, nbcp])';
            cKm = sum(sum(c .* Km));
            projection = projection + cKm * modmatReprojected_k(:,m); 
        end
        ortho_k(p,:) = e - projection;
        orthoResidual_k(p) = norm(ortho_k(p,:));
    end
    modSubspaceDistanceToTruth(k) = norm(orthoResidual_k - orthoResidual_truth);
    modDetDistanceToTruth(k) = abs(det(ortho_k) / det(ortho_truth) - 1);
    
    % Modulation matrix case. Solution 3.
    Kcross_k = zeros(nbcp, nbcp);
    for i=1:nbcp
        for j=1:i
           dsq = (CP_k(i,1)-cp(j,1))^2 + (CP_k(i,2)-cp(j,2))^2;
           ker = exp(-dsq / defKernelSize^2);
           Kcross_k(i, j) = ker;
           Kcross_k(j, i) = ker;
        end
    end
    KcrossDouble_k = kron(Kcross_k, eye(2));
    Kdouble_k = kron(K_k, eye(2));
    Kaug = [Kdouble_truth, KcrossDouble_k ; KcrossDouble_k, Kdouble_k];
    
    W_truth = [modmat ; zeros(nbcp*2,nbSources)];
    G_truth = W_truth' * Kaug * W_truth;
    P_truth = sqrtm(Kaug) * (W_truth / G_truth) * W_truth' * sqrtm(Kaug);
    
    W_k = [zeros(nbcp*2,nbSources) ; modmatProjected_k];
    G_k = W_k' * Kaug * W_k;
    
    if (k>1)
        P_k = sqrtm(Kaug) * (W_k / G_k) * W_k' * sqrtm(Kaug);
        modProjectorDistanceToTruth(k) = norm(P_k - P_truth);
    else
        modProjectorDistanceToTruth(k) = norm(P_truth);
    end
       
    % Case of the template data. 
    pts = [reshape(templateData(k,:),[2, nbpts])', zeros(nbpts, 1)];
    lines = zeros(nbpts,2);
    for m=1:nbpts
        lines(m,1)=m;
        lines(m,2)=m+1;
    end
    lines(nbpts,2)=1;
    VTKPolyDataWriter(pts, lines, [], [], [], 'template_estimated.vtk')
    
    cmd = [deformetricaPath, ' ', num2str(2), ' ', num2str(varifoldKernelSize), ' ', refTemplatePath, ' ', 'template_estimated.vtk'];
    [~, out] = system(cmd);
    templateDataVarifoldDistanceToRef(k) = str2double(out);
    
    cmd = [deformetricaPath, ' ', num2str(2), ' ', num2str(varifoldKernelSize), ' ', statTruthTemplatePath, ' ', 'template_estimated.vtk'];
    [~, out] = system(cmd);
    templateDataVarifoldDistanceToStatTruth(k) = str2double(out);
end
%delete 'tmp.vtk';

% Other scalar parameters. 
referenceTimeDistanceToStatTruth = abs(referenceTime - referenceTime_statTruth);

timeShiftStdDistanceToStatTruth = abs(sqrt(timeShiftVariance) - sqrt(timeShiftVariance_statTruth));
timeShiftVarianceDistanceToStatTruth = abs(timeShiftVariance - timeShiftVariance_statTruth);

logAccelerationStdDistanceToStatTruth = abs(sqrt(logAccelerationVariance) - sqrt(logAccelerationVariance_statTruth));
logAccelerationVarianceDistanceToStatTruth = abs(logAccelerationVariance - logAccelerationVariance_statTruth);

noiseStdDistanceToStatTruth = abs(sqrt(noiseVariance) - sqrt(noiseVariance_statTruth));
noiseVarianceDistanceToStatTruth = abs(noiseVariance - noiseVariance_statTruth);

%% Plot the parameters trajectory.
colors_data = [0 0.5 1 ; 0 1 1 ; 0.4 0.7 0.4 ; 1 0 1 ; 1 0.5 0];
color_target = [0 0 0];

linewidth_data = 2;
linewidth_target = 2;

quiver_linewidth = 1.5;
quiver_linewidth_projector = 4;
quiver_mult = 0.9;

x = 1:saveModelParametersEveryNIters:saveModelParametersEveryNIters*nbSamples;
x = x - 1;
y = ones(nbSamples,1);

fig1 = figure(1);
set(fig1, 'Visible', 'off');
set(gcf,'OuterPosition',[-1500 1200 2000 1000]);

subplot(2,4,7);
plot(x, logAccelerationVariance, 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
plot(x, y * logAccelerationVariance_truth, 'linestyle', '--', 'linewidth' , linewidth_target,'color', color_target);
title('log-acceleration variance');

subplot(2,4,3:4);
for m=1:nbSources
    plot(x, modVelocityFieldNorm{order(m)}, 'linewidth', linewidth_data, 'color', colors_data(1+m,:)); hold on;
    plot(x, y * modVelocityFieldNorm_truth{m}, 'linestyle', '--', 'linewidth' , linewidth_target,'color', color_target); 
end
title('projected modulation matrix-induced velocity fields norm');

subplot(2,4,2);
plot(x, momVelocityFieldNorm, 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
plot(x, y * momVelocityFieldNorm_truth, 'linestyle', '--', 'linewidth' , linewidth_target,'color', color_target);
title('momenta-induced velocity field norm');

subplot(2,4,8);
plot(x, noiseVariance, 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
plot(x, y * noiseVariance_truth, 'linestyle', '--', 'linewidth' , linewidth_target,'color', color_target);
title('noise variance');

subplot(2,4,5);
plot(x, referenceTime, 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
plot(x, y * referenceTime_truth, 'linestyle', '--', 'linewidth' , linewidth_target,'color', color_target);
title('time-shift mean');

subplot(2,4,1);
plot(x, templateDataVarifoldDistanceToRef, 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
plot(x, y * templateDataVarifoldDistanceToRef_truth, 'linestyle', '--', 'linewidth' , linewidth_target,'color', color_target);
title('template data varifold distance to initialization');

subplot(2,4,6);
plot(x, timeShiftVariance, 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
plot(x, y * timeShiftVariance_truth, 'linestyle', '--', 'linewidth' , linewidth_target,'color', color_target);
title('time-shift variance');

saveas(fig1,'Parameters_1_Trajectory','png');

%% Plot the parameters distance to truth.
textSize = 20;

fig2 = figure(2);
set(fig2, 'Visible', 'off');
set(gcf,'OuterPosition',[-1500 1200 2000 1000]);

subplot(2,4,6);
%plot(x, 100 * logAccelerationStdDistanceToStatTruth / sqrt(logAccelerationVariance_statTruth), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
%plot(x, 100 * logAccelerationVarianceDistanceToStatTruth / logAccelerationVariance_statTruth, 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
%plot(x, 100 * (logAccelerationVariance - logAccelerationVariance_statTruth) / logAccelerationVariance_statTruth, 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
plot(x, 100 * (sqrt(logAccelerationVariance) - sqrt(logAccelerationVariance_statTruth)) / sqrt(logAccelerationVariance_statTruth), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('(\sigma_\xi^k - \sigma_\xi^{MAP}) / \sigma_\xi^{MAP}');
xlim([0,x(end)]);
%ylim([0, 100]);
ylim([-50, 50]);
set(gca,'fontsize', textSize)

% subplot(2,4,3);
% plot(x, 100 * modSubspaceDistanceToTruth / modSubspaceDistanceToTruth(1), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
% title('mod matrix subspace dst to truth (% initial)');
% xlim([0,x(end)]);

% subplot(2,4,3);
% plot(x, 100 * cpDistanceToTruth / (1 * 5), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
% title('control points dst to truth (% of kernel width)');
% xlim([0,x(end)]);
% ylim([0, 100]);

% subplot(2,4,3);
% plot(x, 100 * cpRepojectionDistanceToTruth / (1 * nbcp), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
% title('control points reprojection dst to truth (% of kernel width)');
% xlim([0,x(end)]);
% ylim([0, 100]);

% subplot(2,4,4);
% plot(x, 100 * modProjectorDistanceToTruth / modProjectorDistanceToTruth(1), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
% title('mod matrix projector dst to truth (% truth)');
% xlim([0,x(end)]);
% ylim([0, 200]);

subplot(2,4,2);
plot(x, 100 * momVelocityFieldDistanceToStatTruth / momVelocityFieldDistanceToStatTruth(1), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('||v_0^k - v_0^{MAP}|| / ||v_0^0 - v_0^{MAP}||');
xlim([0,x(end)]);
ylim([0, 100]);
set(gca,'fontsize', textSize)

subplot(2,4,7);
%plot(x, 100 * noiseStdDistanceToStatTruth / noiseStdDistanceToStatTruth(1), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
%plot(x, 100 * noiseVarianceDistanceToStatTruth / noiseVarianceDistanceToStatTruth(1), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
plot(x, 100 * (noiseVariance - noiseVariance_statTruth) / (noiseVariance(1) - noiseVariance_statTruth), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
%title('||(\sigma_{\epsilon}^2)^k - (\sigma_{\epsilon}^2)^{MAP}|| / ||(\sigma_{\epsilon}^2)^0 - (\sigma_{\epsilon}^2)^{MAP}||');
title('Unexplained variance (% initial).')
xlim([0,x(end)]);
ylim([0, 100]);
set(gca,'fontsize', textSize)

subplot(2,4,3);
%plot(x, 100 * referenceTimeDistanceToStatTruth / sqrt(timeShiftVariance_statTruth), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
plot(x, 100 * (referenceTime - referenceTime_statTruth) / 4, 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('(t_0^k - t_0^{MAP}) / time span');
xlim([0,x(end)]);
ylim([-50, 50]);
set(gca,'fontsize', textSize)

subplot(2,4,1);
plot(x, 100 * templateDataVarifoldDistanceToStatTruth / templateDataVarifoldDistanceToRef_statTruth(end), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('||y_0^k - y_0^{MAP}||_{varifold} / ||y_0^0 - y_0^{MAP}||_{varifold}');
xlim([0,x(end)]);
ylim([0, 100]);
set(gca,'fontsize', textSize)

subplot(2,4,5);
%plot(x, 100 * timeShiftStdDistanceToStatTruth / sqrt(timeShiftVariance_statTruth), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
%plot(x, 100 * timeShiftVarianceDistanceToStatTruth / timeShiftVariance_statTruth, 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
%plot(x, 100 * (timeShiftVariance - timeShiftVariance_statTruth) / timeShiftVariance_statTruth, 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
plot(x, 100 * (sqrt(timeShiftVariance) - sqrt(timeShiftVariance_statTruth)) / sqrt(timeShiftVariance_statTruth), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('(\sigma_\tau^k - \sigma_\tau^{MAP}) / \sigma_\tau^{MAP}');
xlim([0,x(end)]);
%ylim([0, 100]);
ylim([-50, 50]);
set(gca,'fontsize', textSize)

% Adding graphical representations. 
% Truth. 
subplot(2,4,4);
for k=1:1:size(TemplateEdges,1)
    hold on;
    plot(TemplatePts(TemplateEdges(k,:),1),TemplatePts(TemplateEdges(k,:),2),'-k','LineWidth',2);
end
quiver(X(:),Y(:), vfield_MOM_statTruth{1}(:), vfield_MOM_statTruth{2}(:),0,'LineWidth',quiver_linewidth,'color',colors_data(1,:)); 
for m=1:nbSources
    %quiver(X(:),Y(:), vfields_MOD_statTruth{m,1}(:), vfields_MOD_statTruth{m,2}(:),0,'LineWidth',quiver_linewidth,'color',colors_data(1+m,:)); hold on;
    quiver(X(:),Y(:), vfields_MOD_statTruth{m,1}(:), vfields_MOD_statTruth{m,2}(:),0,'LineWidth',quiver_linewidth,'color',[0.9 0 0]); 
    axis([-2.5 2.5 -2.5 2.5]); 
end
title('(y_0, v_0, v_{A})^{MAP}')
axis([-2.5 2.5 -2.5 2.5]); 
axis off;
set(gca,'fontsize', textSize)

% Last iteration. 
subplot(2,4,8);
for k=1:1:size(TemplateEdges,1)
    hold on;
    plot(pts(TemplateEdges(k,:),1),pts(TemplateEdges(k,:),2),'-k','LineWidth',2);
end
quiver(X(:),Y(:), vfield{1}(:), vfield{2}(:),0,'LineWidth',quiver_linewidth,'color',colors_data(1,:)); 
for m=1:nbSources
    %quiver(X(:),Y(:), vfields_MOD_statTruth{m,1}(:), vfields_MOD_statTruth{m,2}(:),0,'LineWidth',quiver_linewidth,'color',colors_data(1+m,:)); hold on;
    quiver(X(:),Y(:), vfields{m,1}(:), vfields{m,2}(:),0,'LineWidth',quiver_linewidth,'color',[0.9 0 0]); 
    axis([-2.5 2.5 -2.5 2.5]); 
end
title('(y_0, v_0, v_{A})^{k_{max}}')
axis([-2.5 2.5 -2.5 2.5]); 
axis off;
set(gca,'fontsize', textSize)

saveas(fig2,'Parameters_2_DistanceToTruth','png');

%% Plot the parameters distance to truth : variation. 
textSize = 20;

fig2bis = figure(99);
set(fig2bis, 'Visible', 'off');
set(gcf,'OuterPosition',[-1500 1200 1000 1000]);

subplot(2,2,1);
plot(x, 100 * (noiseVariance - noiseVariance_statTruth) / (noiseVariance(1) - noiseVariance_statTruth), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('Unexplained \sigma_\epsilon^2  (% initial).')
xlim([0,x(end)]);
ylim([0, 100]);
set(gca,'fontsize', textSize)

subplot(2,2,2);
plot(x, 100 * (referenceTime - referenceTime_statTruth) / 4, 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('100 \times (t_0^k - t_0^{MAP}) / \Delta_t^{obs}');
xlim([0,x(end)]);
ylim([-50, 50]);
set(gca,'fontsize', textSize)

subplot(2,2,3);
plot(x, 100 * (sqrt(timeShiftVariance) - sqrt(timeShiftVariance_statTruth)) / sqrt(timeShiftVariance_statTruth), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('100 \times (\sigma_\tau^k - \sigma_\tau^{MAP}) / \sigma_\tau^{MAP}');
xlim([0,x(end)]);
ylim([-50, 50]);
set(gca,'fontsize', textSize)

subplot(2,2,4);
plot(x, 100 * (sqrt(logAccelerationVariance) - sqrt(logAccelerationVariance_statTruth)) / sqrt(logAccelerationVariance_statTruth), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('100 \times (\sigma_\xi^k - \sigma_\xi^{MAP}) / \sigma_\xi^{MAP}');
xlim([0,x(end)]);
ylim([-50, 50]);
set(gca,'fontsize', textSize)

saveas(fig2bis,'Parameters_2_DistanceToTruth_bis','png');

%% Plot the parameters distance to truth : variation 2. 
textSize = 20;

fig2ter = figure(98);
set(fig2ter, 'Visible', 'off');
set(gcf,'OuterPosition',[-1500 1200 1000 1000]);

subplot(2,2,1);
plot(x, 100 * noiseVariance / noiseVariance(1), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('Unexplained \sigma_\epsilon^2  (% initial).')
xlim([0,x(end)]);
ylim([0, 100]);
set(gca,'fontsize', textSize)

subplot(2,2,2);
plot(x, 100 * (referenceTime - referenceTime_truth) / 4, 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('100 \times (t_0^k - t_0^{true}) / \Delta_t^{obs}');
xlim([0,x(end)]);
ylim([-50, 50]);
set(gca,'fontsize', textSize)

subplot(2,2,3);
plot(x, 100 * (sqrt(timeShiftVariance) - sqrt(timeShiftVariance_truth)) / sqrt(timeShiftVariance_truth), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('100 \times (\sigma_\tau^k - \sigma_\tau^{true}) / \sigma_\tau^{true}');
xlim([0,x(end)]);
ylim([-50, 50]);
set(gca,'fontsize', textSize)

subplot(2,2,4);
plot(x, 100 * (sqrt(logAccelerationVariance) - sqrt(logAccelerationVariance_truth)) / sqrt(logAccelerationVariance_truth), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('100 \times (\sigma_\xi^k - \sigma_\xi^{true}) / \sigma_\xi^{true}');
xlim([0,x(end)]);
ylim([-50, 50]);
set(gca,'fontsize', textSize)

saveas(fig2ter,'Parameters_2_DistanceToTruth_ter','png');

%% Plot the parameters distance to truth : variation 3. 
textSize = 20;

fig2tetra = figure(97);
set(fig2tetra, 'Visible', 'off');
set(gcf,'OuterPosition',[-1500 1200 1000 1000]);

subplot(2,2,1);
plot(x, 100 * (noiseVariance - noiseVariance_statTruth) / (noiseVariance(1) - noiseVariance_statTruth), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('Unexplained \sigma_\epsilon^2  (% initial).')
xlim([0,x(end)]);
ylim([0, 100]);
set(gca,'fontsize', textSize)

subplot(2,2,2);
plot(x, 100 * (referenceTime - referenceTime_statTruth) / 4, 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('100 \times (t_0^k - t_0^{MAP}) / \Delta_t^{obs}');
xlim([0,x(end)]);
ylim([-50, 50]);
set(gca,'fontsize', textSize)

subplot(2,2,3);
plot(x, 100 * (sqrt(timeShiftVariance) - sqrt(timeShiftVariance_statTruth)) / (sqrt(timeShiftVariance(1)) - sqrt(timeShiftVariance_statTruth)), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('\sigma_\tau^k - \sigma_\tau^{MAP}  (% initial)');
xlim([0,x(end)]);
ylim([-50, 50]);
set(gca,'fontsize', textSize)

subplot(2,2,4);
plot(x, 100 * (sqrt(logAccelerationVariance) - sqrt(logAccelerationVariance_statTruth)) / (sqrt(logAccelerationVariance(1)) - sqrt(logAccelerationVariance_statTruth)), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
title('\sigma_\xi^k - \sigma_\xi^{MAP}  (% initial)');
xlim([0,x(end)]);
ylim([-50, 50]);
set(gca,'fontsize', textSize)

saveas(fig2tetra,'Parameters_2_DistanceToTruth_tetra','png');

%% Plot the parameters distance to truth : variation 4. 
textSize = 21;

fig2quinta = figure(96);
set(fig2quinta, 'Visible', 'off');
set(gcf,'OuterPosition',[-1500 1200 1500 1000]);

subplot(2,3,1);
plot(x/1000, 100 * (noiseVariance - noiseVariance_statTruth) / (noiseVariance(1) - noiseVariance_statTruth), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
ylabel('Unexplained \sigma_\epsilon^2   (% initial)')
xlabel('Number of iterations (\times 1000)')
xlim([0,x(end)/1000]);
ylim([0, 100]);
set(gca,'fontsize', textSize)

subplot(2,3,2);
plot(x/1000, 100 * abs(referenceTime - referenceTime_statTruth) / abs(referenceTime(1) - referenceTime_statTruth), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
ylabel('|t_0^k - t_0^{MAP}|   (% initial)');
xlabel('Number of iterations (\times 1000)')
xlim([0,x(end)/1000]);
ylim([0, 100]);
set(gca,'fontsize', textSize)

subplot(2,3,4);
plot(x/1000, 100 * abs(sqrt(timeShiftVariance) - sqrt(timeShiftVariance_statTruth)) / abs(sqrt(timeShiftVariance(1)) - sqrt(timeShiftVariance_statTruth)), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
ylabel('|\sigma_\tau^k - \sigma_\tau^{MAP}|   (% initial)');
xlabel('Number of iterations (\times 1000)')
xlim([0,x(end)/1000]);
ylim([0, 100]);
set(gca,'fontsize', textSize)

subplot(2,3,5);
plot(x/1000, 100 * abs(sqrt(logAccelerationVariance) - sqrt(logAccelerationVariance_statTruth)) / abs(sqrt(logAccelerationVariance(1)) - sqrt(logAccelerationVariance_statTruth)), 'linewidth', linewidth_data, 'color', colors_data(1,:)); hold on;
ylabel('|\sigma_\xi^k - \sigma_\xi^{MAP}|   (% initial)');
xlabel('Number of iterations (\times 1000)')
xlim([0,x(end)/1000]);
ylim([0, 100]);
set(gca,'fontsize', textSize)

% Adding graphical representations. 
% Truth. 
subplot(2,3,3);
for k=1:1:size(TemplateEdges,1)
    hold on;
    plot(TemplatePts(TemplateEdges(k,:),1),TemplatePts(TemplateEdges(k,:),2),'-k','LineWidth',2);
end
quiver(X(:),Y(:), vfield_MOM_statTruth{1}(:), vfield_MOM_statTruth{2}(:),0,'LineWidth',quiver_linewidth,'color',colors_data(1,:)); 
axis([-2.5 2.5 -2.5 2.5]); 
axis off;
title('Template y_0^{MAP} and geodesic velocity field v_0^{MAP}', 'FontWeight','Normal')
set(gca,'fontsize', textSize)

% Last iteration. 
scale = 10;
subplot(2,3,6);
for k=1:1:size(TemplateEdges,1)
    hold on;
    plot(pts(TemplateEdges(k,:),1),pts(TemplateEdges(k,:),2),'-k','LineWidth',2);
end
quiver(X(:),Y(:), scale*(vfield{1}(:)-vfield_MOM_statTruth{1}(:)), scale*(vfield{2}(:)-vfield_MOM_statTruth{2}(:)),0,'LineWidth',quiver_linewidth,'color','r'); 
axis([-2.5 2.5 -2.5 2.5]); 
axis off;
title('Estimated y_0^{k_{max}} and error v_0^{k_{max}} - v_0^{MAP} (\times 10)', 'FontWeight','Normal');
set(gca,'fontsize', textSize)

saveas(fig2quinta,'Parameters_2_DistanceToTruth_quinta','png');


%% Plot the template and velocity fields. 

fig3 = figure(3);
set(fig3, 'Visible', 'off');
set(gcf,'OuterPosition',[-1500 1200 2000 1200]);

% Truth. 
subplot(3,5,1);
for k=1:2:size(TemplateEdges,1)
    plot(TemplatePts(TemplateEdges(k,:),1),TemplatePts(TemplateEdges(k,:),2),'-k','LineWidth',2); hold on;
end
scatter(cp(:,1),cp(:,2),'r','+'); hold on;
quiver(X(:),Y(:), vfield_MOM_truth{1}(:), vfield_MOM_truth{2}(:),0,'LineWidth',quiver_linewidth,'color',colors_data(1,:)); hold on;
axis([-2.5 2.5 -2.5 2.5]); hold on;
title('Truth : template and v(momenta)'); hold on;
grid on; hold on;

for m=1:nbSources
    subplot(3,5,1+m);
    for k=1:2:size(TemplateEdges,1)
        plot(TemplatePts(TemplateEdges(k,:),1),TemplatePts(TemplateEdges(k,:),2),'-k','LineWidth',2); hold on;
    end
    scatter(cp(:,1),cp(:,2),'r','+'); hold on;
    quiver(X(:),Y(:), vfields_MOD_truth{m,1}(:), vfields_MOD_truth{m,2}(:),0,'LineWidth',quiver_linewidth,'color',colors_data(1+m,:)); hold on;
%     quiver(cp(:,1),cp(:,2), projector_truth(1:2:end,m), projector_truth(2:2:end,m),0,'LineWidth',quiver_linewidth_projector,'color',colors_data(1+m,:)*quiver_mult);
    axis([-2.5 2.5 -2.5 2.5]); hold on;
    title(['Truth : template and v(mod_ ',num2str(m),')']); hold on;
    grid on; hold on;
end

% Estimation. 
subplot(3,5,6);
for k=1:size(TemplateEdges,1)
    plot(pts(TemplateEdges(k,:),1),pts(TemplateEdges(k,:),2),'-k','LineWidth',2);
    hold on;
end
scatter(CP_k(:,1),CP_k(:,2),'r','x');
quiver(X(:),Y(:), vfield{1}(:), vfield{2}(:),0,'LineWidth',quiver_linewidth,'color',colors_data(1,:));
axis([-2.5 2.5 -2.5 2.5]); hold on;
title('Estimation : template and v(momenta)');
grid on;

for m=1:nbSources
    subplot(3,5,6+m);
    for k=1:size(TemplateEdges,1)
        plot(pts(TemplateEdges(k,:),1),pts(TemplateEdges(k,:),2),'-k','LineWidth',2);
        hold on;
    end
    scatter(CP_k(:,1),CP_k(:,2),'r','x');
    quiver(X(:),Y(:), sign(m) * vfields{order(m),1}(:), sign(m) * vfields{order(m),2}(:),0,'LineWidth',quiver_linewidth,'color',colors_data(1+m,:));
%     quiver(cp(:,1),cp(:,2), sign(m) * projector_k(1:2:end,order(m)), sign(m) * projector_k(2:2:end,order(m)),0,'LineWidth',quiver_linewidth_projector,'color',colors_data(1+m,:)/2);
%     quiver(cp(:,1),cp(:,2), sign(m) * modmatReprojected_k(1:2:end,order(m)), sign(m) * modmatReprojected_k(2:2:end,order(m)),0,'LineWidth',quiver_linewidth_projector,'color',colors_data(1+m,:)*quiver_mult);    
    axis([-2.5 2.5 -2.5 2.5]); hold on;
    title(['Estimation : template and v(mod_ ',num2str(m),')']);
    grid on; 
end

% Superposition. 
subplot(3,5,11);
for k=1:2:size(TemplateEdges,1)
    plot(TemplatePts(TemplateEdges(k,:),1),TemplatePts(TemplateEdges(k,:),2),'-k','LineWidth',2); hold on;
end
for k=1:size(TemplateEdges,1)
    plot(pts(TemplateEdges(k,:),1),pts(TemplateEdges(k,:),2),'-k','LineWidth',2); hold on;
end
scatter(cp(:,1),cp(:,2),'r','+'); hold on;
scatter(CP_k(:,1),CP_k(:,2),'r','x'); hold on;
quiver(X(:),Y(:), vfield{1}(:)-vfield_MOM_truth{1}(:), vfield{2}(:)-vfield_MOM_truth{2}(:),0,'LineWidth',quiver_linewidth,'color',colors_data(1,:)); hold on;
axis([-2.5 2.5 -2.5 2.5]); hold on;
title('Difference : v(estimation) - v(truth)');
grid on;

% for m=1:nbSources
%     subplot(3,5,11+m);
%     for k=1:2:size(TemplateEdges,1)
%         plot(TemplatePts(TemplateEdges(k,:),1),TemplatePts(TemplateEdges(k,:),2),'-k','LineWidth',2); hold on;
%     end
%     for k=1:size(TemplateEdges,1)
%         plot(pts(TemplateEdges(k,:),1),pts(TemplateEdges(k,:),2),'-k','LineWidth',2); hold on;
%     end
%     scatter(cp(:,1),cp(:,2),'r','+');
%     scatter(CP_k(:,1),CP_k(:,2),'r','x');
%     quiver(X(:),Y(:), sign(m) * vfields{order(m),1}(:)-vfields_MOD_truth{m,1}(:), sign(m) * vfields{order(m),2}(:)-vfields_MOD_truth{m,2}(:),0,'LineWidth',quiver_linewidth,'color',colors_data(1+m,:));
% %     quiver(cp(:,1),cp(:,2), sign(m) * projector_k(1:2:end,order(m))-projector_truth(1:2:end,m), sign(m) * projector_k(2:2:end,order(m))-projector_truth(2:2:end,m),0,'LineWidth',quiver_linewidth_projector,'color',colors_data(1+m,:)*quiver_mult);
%     axis([-2.5 2.5 -2.5 2.5]); hold on;
%     title('Difference : v(estimation) - v(truth)');
%     grid on; 
% end

saveas(fig3,'Parameters_3_GraphicalRepresentation','png');

% %% Additional figure, with the truth only. 
% 
% fig4=figure(4);
% set(fig4, 'Visible', 'off');
% set(gcf,'OuterPosition',[-1500 1200 2000 400]);
% 
% boundx = 2;
% boundy = 2;
% 
% subplot(1,5,1);
% for k=1:1:size(TemplateEdges,1)
%     plot(TemplatePts(TemplateEdges(k,:),1),TemplatePts(TemplateEdges(k,:),2),'-k','LineWidth',3); hold on;
% end
% scatter(cp(:,1),cp(:,2),'r','+'); hold on;
% quiver(X(:),Y(:), vfield_MOM_truth{1}(:), vfield_MOM_truth{2}(:),0,'LineWidth',quiver_linewidth,'color',colors_data(1,:)); hold on;
% 
% axis([-boundx boundx -boundy boundy]);
% axis off;
% 
% for m=1:nbSources
%     subplot(1,5,1+m);   
%     for k=1:1:size(TemplateEdges,1)
%         plot(TemplatePts(TemplateEdges(k,:),1),TemplatePts(TemplateEdges(k,:),2),'-k','LineWidth',3); hold on;
%     end
%     scatter(cp(:,1),cp(:,2),'r','+'); hold on;
%     quiver(X(:),Y(:), vfields_MOD_truth{m,1}(:), vfields_MOD_truth{m,2}(:),0,'LineWidth',quiver_linewidth,'color',colors_data(1+m,:)); hold on;
%     
%     axis([-boundx boundx -boundy boundy]);
%     axis off;
% end
% 
% saveas(fig4,'Parameters_4_GraphicalRepresentationTruth','png');
