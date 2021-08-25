function [nData_dec] = processdata()

%% load data

dir = '/Volumes/GoogleDrive/Shared drives/aoLab/Data/bmiLearning_jeev/jeev072312_080412/';

% this is NOT normalized 
nData = load(strcat(dir, 'catNeuralDat_jeev072312_080412_trE5_B100.mat'), 'N_bmi', 'bmiDecoderInd', 'sessID_bmi');
% have to access by nData.N_bmi

decData = load(strcat(dir, 'decoderParams_jeev072312_080412.mat'), 'bmiUnitInds_all', 'usedUnit_decoder');


%% sort ndata to only the decoder neurons, time bins to care about and trials we care about

% to sort time
alignCue = 21; % where binvector = 0

% to sort neuron units that we care about
dNum = 2;
usedUnits = decData.bmiUnitInds_all(decData.usedUnit_decoder(dNum,:));

% to sort the trials that we care about

sess_ind = find(nData.bmiDecoderInd == dNum);

dec_sess = [];
for iS = sess_ind
    dec_sess = cat(1, dec_sess, find(nData.sessID_bmi == iS));
end
dec_sess = sort(dec_sess);

nData_dec = nData.N_bmi(alignCue:end, usedUnits, dec_sess)/.1;

size(nData_dec)

%%



