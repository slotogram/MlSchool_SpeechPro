
fea_dir =  'C:\Samples\data_v_7_stc\audio'; %training directory
test_dir = 'C:\Samples\data_v_7_stc\test'; %test directory
test_meta = 'C:\Samples\data_v_7_stc\meta\meta.txt';

ubmFile = 'gmms_MFC20EDVZ.mat';
bwFile = 'bw.mat';
tFile = 'T.mat';
pldaFile = 'plda.mat';


%��������
featCol=[1:42,64,65];

%������� ������ � ������� �� ��������
fid = fopen(test_meta, 'rt');
meta = textscan(fid, '%q %q %q %q %q','Delimiter',{'	'});
fclose(fid);
filenames = meta{1};
filenames = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepare path to files
                       filenames, 'UniformOutput', false);
filenames = cellfun(@(x) x(1:end-3),filenames, 'un',0);

%filenames_i = extractBefore(filenames,'_time_stretch');

filenames = cellfun(@(x) strcat(x,'htk'),filenames, 'un',0);

model_ids = unique(meta{5}, 'stable'); %�������� ��� ������������ ������



%% Step0: Opening MATLAB pool
nworkers = 4;
nworkers = min(nworkers, feature('NumCores'));
p = gcp('nocreate'); 
if isempty(p), parpool(nworkers); end 

%% Step1: Training the UBM

%Check if UBM is already trained
loadMem = true; %% load all files into the memory
loadUBM = true; %% load UBM from disk
loadBW = false; %% load bw from disk
loadT = false; %% load T from disk
loadP = false; %% load PLDA from disk

if ~exist('dataCut','var')
    dataCut = load_data(filenames,featCol);
end
if loadUBM&&exist(ubmFile,'file')
  
    load(ubmFile);
    clear('gmm_models');

else    
nmix = 256;
final_niter = 10;
ds_factor = 1;

ubm = gmm_em(dataCut, nmix, final_niter, ds_factor, nworkers,'',featCol);%,vadCol,vadThr);

end


%% Step2: Learning the total variability subspace from background data
tv_dim = 400; 
niter  = 5;

if loadBW&&exist(bwFile,'file')
      
    load(bwFile);

else    
stats = cell(length(dataCut), 1);

parfor file = 1 : length(dataCut)
    [N, F] = compute_bw_stats(dataCut{file}, ubm, featCol,'','');
    stats{file} = [N; F];
end
save(bwFile,'stats');

end
if loadT&&exist(tFile,'file')
      
    load(tFile);

else    
T = train_tv_space(stats, ubm, tv_dim, niter, nworkers);
save(tFile,'T');
end

%% Step3: Training the Gaussian PLDA model with development i-vectors

if loadP&&exist(pldaFile,'file')
      
    load(pldaFile);

else
lda_dim = 50;

nphi    = 200;
niter   = 10;

dev_ivs = zeros(tv_dim, length(filenames));
parfor file = 1 : length(filenames)
    dev_ivs(:, file) = extract_ivector(stats{file}, ubm, T);
end
% reduce the dimensionality with LDA
spk_labs = meta{5};
nSpeakers = size(model_ids,1);

%lda_dim = min(lda_dim, nSpeakers-1);
V = lda(dev_ivs, spk_labs);
dev_ivs = V(:, 1 : lda_dim)' * dev_ivs;
%------------------------------------
plda = gplda_em(dev_ivs, spk_labs, nphi, niter);

save(pldaFile,'plda','V','lda_dim');
end

%% Step4: Scoring the verification trials

%clear dataCut;
nspks = length(model_ids);
model_ivs1 = zeros(tv_dim, nspks);
model_ivs2 = model_ivs1;
parfor spk = 1 : nspks
    ids = find(ismember(spk_labs, model_ids{spk}));
    spk_files = filenames(ids);
    
    N = 0; F = 0; 
    for ix = 1 : length(spk_files)
        [n, f] = compute_bw_stats(spk_files{ix}, ubm, featCol, '', '');
        N = N + n; F = f + F; 
        model_ivs1(:, spk) = model_ivs1(:, spk) + extract_ivector([n; f], ubm, T);
    end
    model_ivs2(:, spk) = extract_ivector([N; F]/length(spk_files), ubm, T); % stats averaging!
    model_ivs1(:, spk) = model_ivs1(:, spk)/length(spk_files); % i-vector averaging!
end

%��������� �������� ������ � ���������� ������ ���������
test_files = struct2cell(dir(strcat(test_dir,'\*.htk')));
test_files =  test_files(1,1:473)'; %������ unknown, ��� ��� ��� ��� ���������� ������

test_Ids = strtok(test_files,'_');
model_ids = strtok(model_ids,'_'); % knocking fix

trials = zeros(nspks*length(test_files),2); %������� ���� �� ������ ������ ��� ������� �����
labels = zeros(nspks*length(test_files),1);
Kmodel = zeros(nspks*length(test_files),1);
Ktest = zeros(nspks*length(test_files),1);
for i=1:length(test_files)
    for j=1:nspks
        trials(((i-1)*nspks)+j,:)=[j,i];
        labels(((i-1)*nspks)+j) = startsWith(test_files(i),model_ids(j));
        Kmodel(((i-1)*nspks)+j) = j;
        Ktest(((i-1)*nspks)+j) = i;
    end
end

test_files = cellfun(@(x) fullfile(test_dir, x),...  %# Prepare path to files
                       test_files, 'UniformOutput', false);

test_ivs = zeros(tv_dim, length(test_files));
parfor tst = 1 : length(test_files)
    [N, F] = compute_bw_stats(test_files{tst}, ubm, featCol, '', '');
    test_ivs(:, tst) = extract_ivector([N; F], ubm, T);
end
% reduce the dimensionality with LDA
model_ivs1 = V(:, 1 : lda_dim)' * model_ivs1;
model_ivs2 = V(:, 1 : lda_dim)' * model_ivs2;
test_ivs = V(:, 1 : lda_dim)' * test_ivs;
%------------------------------------
scores1 = score_gplda_trials(plda, model_ivs1, test_ivs);
linearInd =sub2ind([nspks, length(test_files)], Kmodel, Ktest);
scores1 = scores1(linearInd); % select the valid trials

scores2 = score_gplda_trials(plda, model_ivs2, test_ivs);
scores2 = scores2(linearInd); % select the valid trials

%% Step5: Computing the EER and plotting the DET curve

eer1 = compute_eer(scores1(linearInd), labels, true); % IV averaging
hold on
eer2 = compute_eer(scores2(linearInd), labels, true); % stats averaging

%[eer, dcf1, dcf2] = compute_eer(scores2(linearInd), labels, false);
%[eer, dcf1, dcf2] = compute_eer(scores1(linearInd), labels, false);

%% Computing classification error
scores = scores1(linearInd);
TP=0;
for i=1:length(test_files)
    %������� ������������ �������
    [mx,ind] = max(scores(((i-1)*nspks)+1:(i-1)*nspks+nspks));
    %���������� � ���, ������� ������ ����.
    if (labels((i-1)*nspks+ind))
        TP = TP+1;
    end
   
end
TP/length(test_files)
plot_cm;
