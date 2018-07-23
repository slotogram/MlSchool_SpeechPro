test_files2 = struct2cell(dir(strcat(test_dir,'\*.wav')));
test_files2 =  test_files2(1,:)';
%model_ids = unique(meta{5}, 'stable'); %выбираем все существующие классы
mx = zeros(nfiles,1);
mn  = zeros(nfiles,1);
scnd =  zeros(nfiles,1);
ind = zeros(nfiles,1);
for i=1:nfiles
    A = sort(scores(((i-1)*nspks)+1:(i-1)*nspks+nspks),'descend');
    mn(i) = A(nspks);
    scnd(i) = A(2);
	[mx(i),ind(i)] = max(scores(((i-1)*nspks)+1:(i-1)*nspks+nspks)); %берем максимумы оценок

end

fid = fopen('result.txt','w');
for i=1:nfiles
    score_p =((mx(i)-scnd(i))/(scnd(i)-mn(i))); 
    if score_p >0.9 
        score_p=0.9;
    end
	fprintf(fid,'%s	%.3f	%s\n',test_files2{i}, score_p  ,model_ids{ind(i)});
end
fclose(fid);