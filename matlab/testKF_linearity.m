%% Testing the kalman linearity

%% setting up the neural data

% bmiUnits = decoder.predInds;  % changes for decoder
% bmiInds = bmiDecoderInd;
% sessID = sessID_bmi;
% N_bmi = N_norm_bmi;
% 
% sess_ind = find(bmiInds == decInd);
% 
% dec_sess = [];
% for iS = sess_ind
%     dec_sess = cat(1, dec_sess, find(sessID == iS));
% end
% dec_sess = sort(dec_sess);
% n_data = N_bmi(align:end, bmiUnits, dec_sess)/.1;
% 
% size(n_data)

%% load data

clear;
dir = '/Volumes/GoogleDrive/Shared drives/aoLab/Data/bmiLearning_jeev/jeev072312_080412/';

% this is NOT normalized 
nData = load(strcat(dir, 'catNeuralDat_jeev072312_080412_trE5_B100.mat'), 'N_norm_bmi', 'bmiDecoderInd', 'sessID_bmi', 'trEs_bmi');
% have to access by nData.N_bmi

% decoder information
decData = load(strcat(dir, 'decoderParams_jeev072312_080412.mat'), 'bmiUnitInds_all', 'usedUnit_decoder');

% kinematic/cursor information
kinPath = load(strcat(dir, 'catBehaviorDat_jeev072312_080412.mat'), 'kinSegs', 'segE');

decPath = '/Volumes/GoogleDrive/Shared drives/aoLab/Data/bmiLearning_jeev/jeev072312/jeev072312_VFB_Kawf_B100_NS5_NU16_Z1_smoothbatch.mat';
dec = load(decPath,'decoder');



%% sort ndata to only the decoder neurons, time bins to care about and trials we care about

% to sort time
alignCue = 21; % where binvector = 0
endCue = 36;

% to sort neuron units that we care about
dNum = 2;
usedUnits = decData.bmiUnitInds_all(decData.usedUnit_decoder(dNum,:))


% to sort the trials that we care about
sess_ind = find(nData.bmiDecoderInd == dNum);

dec_sess = [];
for iS = sess_ind
    dec_sess = cat(1, dec_sess, find(nData.sessID_bmi == iS));
end
dec_sess = sort(dec_sess);
dec_start = dec_sess(1);
dec_end = dec_sess(end);
succIdx = find(nData.trEs_bmi(dec_start:dec_end, 7) == 8 | nData.trEs_bmi(dec_start:dec_end, 7) == 9) + (dec_start-1);
% succLog = (nData.trEs_bmi(:, 7) == 8 |nData.trEs_bmi(:, 7) == 9);

%%
% neural data: time bins since go cue x neurons used in decoder x trials in
% decoder
n_data = nData.N_norm_bmi(alignCue:endCue, usedUnits, succIdx)/.1;

size(succIdx)
size(n_data)
%%


% setting up sizes
b_num = size(n_data, 1); % number of timebins (2) 
tr_num = size(succIdx, 1); % number of trials 
n_num = size(n_data, 2); % number of neurons (16)
s_num = 5; % number of states
%% setting up kalman parameters

dec_A = dec.decoder.A;
dec_W = dec.decoder.W;
dec_H = dec.decoder.H;
dec_Q = dec.decoder.Q;

% initial state from kinematic data
% dec_x0 =  cat(1, kinPath.kinSegs{dec_sess(1)}(1, :)', 1);
dec_x0 = [0 0 0 0 1]';

%% run kalman filter

X_hat = zeros(5, b_num, tr_num);
X_hat_flat = zeros(s_num*b_num, tr_num);
ndata_flat = zeros(s_num*n_num, tr_num);
for iT = 2:tr_num
    X_hat(:, :, iT) = runKalmanForward(dec_A, dec_W, dec_H, dec_Q, n_data(:, :, iT)', dec_x0);
    for iB = 1:b_num
        idx_x = (iB - 1)*s_num + 1;
        X_hat_flat(idx_x:idx_x+s_num-1, iT) = X_hat(:, iB, iT);
        
        idx_n = (iB - 1)*n_num + 1;
        ndata_flat(idx_n:idx_n+n_num-1, iT) = n_data(iB, :, iT);
    end
    dec_x0 = [0 0 0 0 1]';
%     dec_x0 = X_hat(:, 1, iT);
end    

% X_hat_flat is the x_hat estimate from running the kalman filter

%% find k_estimate

d_est = X_hat_flat*pinv(ndata_flat);
X_est = zeros(s_num*b_num, tr_num);
for iT = 2:tr_num
    X_est(:, iT) = d_est*ndata_flat(:, iT);
end

%% plotting

% X_est is te 
X_est_pos = X_est;
X_act_pos = X_hat_flat;

scatter(X_act_pos, X_est_pos)
xlabel('position (actual)')
ylabel('position (estimated)')

%%
n = 5;
X_act_pos(n:n:end, :) = [];
X_est_pos(n:n:end, :) = [];
n = 4;
X_act_pos(n:n:end, :) = [];
X_est_pos(n:n:end, :) = [];
n = 3;
X_act_pos(n:n:end, :) = [];
X_est_pos(n:n:end, :) = [];

 
% % plot only the x's
% figure;
% scatter(X_act_pos(:, 2:end), X_est_pos(:, 2:end))
% title('back calculating kalman states: actual x,y pos vs estimate x,y pos')
% xlabel('position (actual)')
% ylabel('position (estimated)')
% 
% 
% 
% % plot only the x's
% figure;
% scatter(X_act_pos(1:2:end, 2:end), X_est_pos(1:2:end, 2:end))
% title('back calculating kalman states: actual x pos vs estimate x pos')
% xlabel('position (actual)')
% ylabel('position (estimated)')
% 
% % plot only the y's
% figure;
% scatter(X_act_pos(2:2:end, 2:end), X_est_pos(2:2:end, 2:end))
% title('back calculating kalman states: actual y pos vs estimate y pos')
% xlabel('position (actual)')
% ylabel('position (estimated)')

%% commented here

%% What does the actual state information look like?


x_data = kinPath.kinSegs(succIdx);
x_data_sub = zeros(b_num, tr_num, 4);
for iX = 1:tr_num
%     if (size(x_data{iX}, 1) >= 20 )
    lin_v = int64(linspace(1, size(x_data{iX}, 1), 20));
    x_data_sub(:, iX, :) = x_data{iX}([lin_v], :); 
end
% x_lin = int64(linspace(0, size(k1, 1), 20))


%%

% plot only the x's
figure;
scatter(X_act_pos(1:2:end, :), x_data_sub(:, :, 1))
title('back calculating kalman states: actual x,y pos vs estimate x,y pos')
xlabel('position (actual)')
ylabel('position (estimated)')



%% Plot the kinSegs data and plot the x_act data


d_start = dec_sess(1);
d_end = dec_sess(end);
t_kin = kinPath.segE(d_start:d_end, 2) - 63;
t_cat = nData.trEs_bmi(d_start:d_end, 2) - 63;
t_diff = t_kin - t_cat;

%%
figure;

for iD = 1:3
%     x_kin = x_data{iD}(:, 1);
%     y_kin = x_data{iD}(:, 2);
    
    x_kin = x_data_sub(:, iD, 1);
    y_kin = x_data_sub(:, iD, 2);
    
%     x_est = X_act_pos(1:2:end, iD);
%     y_est = X_act_pos(2:2:end, iD);
    
    plot(x_kin, y_kin, '-o', 'DisplayName', strcat("t: ", num2str(t_kin(iD))))
%     hold on
%     plot(x_est, y_est,'-o', 'DisplayName', strcat("est: ", num2str(iD)))
    legend()
    hold on
end

%%
figure;
for iD = 1:3
%     x_kin = x_data{iD}(:, 1);
%     y_kin = x_data{iD}(:, 2);
    
    x_est = X_act_pos(1:2:end, iD);
    y_est = X_act_pos(2:2:end, iD);
    
%     plot(x_kin, y_kin, '-o', 'DisplayName', strcat("act: ", num2str(iD)))
%     hold on
    plot(x_est, y_est,'-o', 'DisplayName', strcat("est: ", num2str(iD)))
    legend()
    hold on
end
