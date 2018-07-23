fea_dir =  'C:\Samples\data_v_7_stc\audio'; %training directory
test_dir = 'C:\Samples\data_v_7_stc\test'; %test directory
test_meta = 'C:\Samples\data_v_7_stc\meta\meta.txt';
gmm_file = 'svm_.mat';

%выбираем нужный вектор признаков
featCol=[1:42,64,65];

%Создаем список с файлами на обучение
fid = fopen(test_meta, 'rt');
meta = textscan(fid, '%q %q %q %q %q','Delimiter',{'	'});
fclose(fid);
filenames = meta{1};
filenames = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepare path to files
                       filenames, 'UniformOutput', false);
filenames = cellfun(@(x) x(1:end-3),filenames, 'un',0);
filenames = cellfun(@(x) strcat(x,'htk'),filenames, 'un',0);

model_ids = unique(meta{5}, 'stable'); %выбираем все существующие классы


%для параллельного исполнения
nworkers = 4;
nworkers = min(nworkers, feature('NumCores'));
p = gcp('nocreate'); 
if isempty(p), parpool(nworkers); end 
rng(1);

%% Step1: Loading and computing features

loadMem = true; %% load all files into the memory


if  ~loadMem || ~exist('dataUBM','var') 
    % сначала загружаем со всеми признаками
	dataUBM = load_data(filenames);
    
end
%%removing not needed features
dataCut = dataUBM;
nfiles = size(dataCut, 1);
for ix = 1 : nfiles
    dataCut{ix} = dataCut{ix}(featCol,:);
end

%%get new features
%dataCut = cell2mat(cellfun(@(x) [mean(x,2);std(x,0,2)],dataCut, 'un',0));
meanData = cellfun(@(x) mean(x,2),dataCut, 'un',0);
stdData = cellfun(@(x) std(x,0,2),dataCut, 'un',0);

dataCut = [meanData{:};stdData{:}]';
%dataCut = [meanData{:}]';
nfiles = size(dataCut, 1);

nspks = length(model_ids); 
models = cell(nspks, 1); 

%%обучаем SVM модели
for spk = 1 : nspks
        fprintf('.');
        %set labels for speaker
        labels = zeros(size(dataCut,1),1);
        ids = find(ismember(meta{5}, model_ids{spk}));
        labels (ids) = 1;
    
        %train model
        models{spk} = fitcsvm(dataCut,labels,'KernelFunction','linear' );
end

save(gmm_file,'models','featCol');
%% Step3: Scoring the classification trials

%загружаем тестовые данные и составляем список испытаний
test_files = struct2cell(dir(strcat(test_dir,'\*.htk')));
test_files =  test_files(1,1:473)'; %убрали unknown, так как для них неизвестны классы

test_Ids = strtok(test_files,'_');
model_ids = strtok(model_ids,'_'); % knocking fix
%test_files1=strtok(test_files,'_');
trials = zeros(nspks*length(test_files),2); %создаем тест на каждую модель для каждого файла
labels = zeros(nspks*length(test_files),1); %метки истинности теста
for i=1:length(test_files)
    for j=1:nspks
        trials(((i-1)*nspks)+j,:)=[j,i];
        labels(((i-1)*nspks)+j) = startsWith(test_files(i),model_ids(j));
    end
end

test_files = cellfun(@(x) fullfile(test_dir, x),...  %# Prepare path to files
                       test_files, 'UniformOutput', false);
                   
if  ~loadMem || ~exist('dataTest','var') 
    % сначала загружаем со всеми признаками
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

%%get new features
meanData = cellfun(@(x) mean(x,2),dataCut1, 'un',0);
stdData = cellfun(@(x) std(x,0,2),dataCut1, 'un',0);
dataCut1 = [meanData{:};stdData{:}]';
%dataCut1 = [meanData{:}]';

%получаем результаты классификации
scores = zeros(size(trials,1),1);
for i=1:size(trials,1)
    [~,score] = predict(models{trials(i,1)},dataCut1(trials(i,2),:));
    scores(i)=sum(score(:,2))/size(score,1);
end

save('scores_svm','scores');

%% Save scores to file

test_files2 = struct2cell(dir(strcat(test_dir,'\*.wav')));
test_files2 =  test_files2(1,1:473)'; %убрали unknown, так как для них неизвестны классы
model_ids = unique(meta{5}, 'stable'); %выбираем все существующие классы
mx = zeros(nfiles,1);
ind = zeros(nfiles,1);
for i=1:nfiles
	[mx(i),ind(i)] = max(scores(((i-1)*nspks)+1:(i-1)*nspks+nspks)); %берем максимумы оценок
end
max_score=max(mx);
%min_score=median(mx);
min_score=min(mx);

scores_p=(0.5*(scores-min_score)/(max_score-min_score))+0.5; %вычисляем вероятность (из-за GMM)

fid = fopen('result.txt','w');
for i=1:nfiles
    [mx,ind] = max(scores_p(((i-1)*nspks)+1:(i-1)*nspks+nspks));
	fprintf(fid,'%s	%.3f	%s\n',test_files2{i},mx,model_ids{ind});
end
fclose(fid);

%% Step4: Computing the EER and plotting the DET curve

[eer, dcf1, dcf2] = compute_eer(scores, labels, true); % просто ради интереса.

%% Computing classification error
TP=0;
for i=1:length(test_files)
    %находим максимальный вариант
    [mx,ind] = max(scores(((i-1)*nspks)+1:(i-1)*nspks+nspks));
    %сравниваем с тем, который должен быть.
    if (labels((i-1)*nspks+ind))
        TP = TP+1;
    end
   
end
TP/length(test_files)
plot_cm; %рисуем confusion matrix. Работает если есть NN Toolbox
