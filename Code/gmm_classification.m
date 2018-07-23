%GMM-UBM model sound classification
fea_dir =  'C:\Samples\data_v_7_stc\audio'; %training directory
test_dir = 'C:\Samples\data_v_7_stc\test'; %test directory
test_meta = 'C:\Samples\data_v_7_stc\meta\meta.txt';
gmm_file = 'gmms_.mat';

%�������� ������ ������ ���������
featCol=[1:42,64,65];

%������� ������ � ������� �� ��������
fid = fopen(test_meta, 'rt');
meta = textscan(fid, '%q %q %q %q %q','Delimiter',{'	'});
fclose(fid);
filenames = meta{1};
filenames = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepare path to files
                       filenames, 'UniformOutput', false);
filenames = cellfun(@(x) x(1:end-3),filenames, 'un',0);
filenames = cellfun(@(x) strcat(x,'htk'),filenames, 'un',0);

model_ids = unique(meta{5}, 'stable'); %�������� ��� ������������ ������


%��� ������������� ����������
nworkers = 4;
nworkers = min(nworkers, feature('NumCores'));
p = gcp('nocreate'); 
if isempty(p), parpool(nworkers); end 
%% Step1: Training the UBM

loadMem = true; %% load all files into the memory
loadUBM = false; %% load UBM from disk

if loadUBM&&exist(ubmFile,'file')
  
load(ubmFile);
ubm = gmm;
clear('gmm');

else    
nmix = 1024; %���������� ����������� UBM
final_niter = 10;
ds_factor = 1;
if  ~loadMem || ~exist('dataUBM','var') 
    % ������� ��������� �� ����� ����������
	dataUBM = load_data(filenames);
    
end

%%removing not needed features
dataCut = dataUBM;
nfiles = size(dataCut, 1);
for ix = 1 : nfiles
    dataCut{ix} = dataCut{ix}(featCol,:);
end

ubm = gmm_em(dataCut, nmix, final_niter, ds_factor, nworkers,'',featCol);% ������� UBM

end

%% Step2: Adapting the speaker models from UBM

map_tau = 6.0; 
config = 'mvw';
model_ids = unique(meta{5}, 'stable'); %�������� ��� ������������ ������

nspks = length(model_ids); 
gmm_models = cell(nspks, 1); 
dataTrain = cell(nspks,1);

for spk = 1 : nspks %�������� ������ �� ������ ������
    ids = find(ismember(meta{5}, model_ids{spk}));
    dataTrain{spk} = dataCut(ids);            
end


%%��� ��� � ������������ ������� �������� ��������� UBM
for spk = 1 : nspks
    gmm_models{spk} = mapAdapt(dataTrain{spk}, ubm, map_tau, config,'',featCol,'','');
end

save(gmm_file,'ubm','gmm_models','featCol');
%% Step3: Scoring the classification trials

%��������� �������� ������ � ���������� ������ ���������
test_files = struct2cell(dir(strcat(test_dir,'\*.htk')));
test_files =  test_files(1,1:473)'; %������ unknown, ��� ��� ��� ��� ���������� ������

test_Ids = strtok(test_files,'_');
model_ids = strtok(model_ids,'_'); % knocking fix

trials = zeros(nspks*length(test_files),2); %������� ���� �� ������ ������ ��� ������� �����
labels = zeros(nspks*length(test_files),1); %����� ���������� �����
for i=1:length(test_files)
    for j=1:nspks
        trials(((i-1)*nspks)+j,:)=[j,i];
        labels(((i-1)*nspks)+j) = startsWith(test_files(i),model_ids(j));
    end
end

test_files = cellfun(@(x) fullfile(test_dir, x),...  %# Prepare path to files
                       test_files, 'UniformOutput', false);
                   
if  ~loadMem || ~exist('dataTest','var') 
    % ������� ��������� �� ����� ����������
    nfiles = length(test_files);
    dataTest = cell(nfiles, 1);
    
    for ix = 1 : nfiles
         dataTest{ix} = htkread(test_files{ix});
    end

    
end

%removing not needed features
dataCut1 = dataTest;
nfiles = length(dataCut1);
for ix = 1 : nfiles
    dataCut1{ix} = dataCut1{ix}(featCol,:);
end


%�������� ���������� �������������
scores = score_gmm_trials2(gmm_models, dataCut1, trials, '',featCol,'','');
save('scores_ubm','scores');

%% Save scores to file

test_files2 = struct2cell(dir(strcat(test_dir,'\*.wav')));
test_files2 =  test_files2(1,1:473)'; %������ unknown, ��� ��� ��� ��� ���������� ������
model_ids = unique(meta{5}, 'stable'); %�������� ��� ������������ ������
mx = zeros(nfiles,1);
ind = zeros(nfiles,1);
for i=1:nfiles
	[mx(i),ind(i)] = max(scores(((i-1)*nspks)+1:(i-1)*nspks+nspks)); %����� ��������� ������
end
max_score=max(mx);

min_score=min(mx);

scores_p=(0.5*(scores-min_score)/(max_score-min_score))+0.5; %��������� ����������� (��-�� GMM)

fid = fopen('result.txt','w');
for i=1:nfiles
    [mx,ind] = max(scores_p(((i-1)*nspks)+1:(i-1)*nspks+nspks));
	fprintf(fid,'%s	%.3f	%s\n',test_files2{i},mx,model_ids{ind});
end
fclose(fid);

%% Step4: Computing the EER and plotting the DET curve

[eer, dcf1, dcf2] = compute_eer(scores, labels, true); % ������ ���� ��������.

%% Computing classification error
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
plot_cm; %������ confusion matrix. �������� ���� ���� NN Toolbox


