
predicted = zeros(nspks,length(test_files));
class=zeros(nspks,length(test_files));
for i=1:length(test_files)
    %������� ������������ �������
    predicted(:,i) = scores(((i-1)*nspks)+1:(i-1)*nspks+nspks);% ������������� �����
    %[~,predicted(i)] = max(scores(((i-1)*nspks)+1:(i-1)*nspks+nspks)); 
    class(:,i) = labels(((i-1)*nspks)+1:(i-1)*nspks+nspks);% �������� �����
    %[~,class(i)] = max(labels(((i-1)*nspks)+1:(i-1)*nspks+nspks)); 
    %���������� � ���, ������� ������ ����.
end

plotconfusion(class,predicted)