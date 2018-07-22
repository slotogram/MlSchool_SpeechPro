function [ data ] = load_data( datalist,featCol,vadCol,vadThr )
% load all data into memory
if ~iscellstr(datalist)
    fid = fopen(datalist, 'rt');
    filenames = textscan(fid, '%q');
    fclose(fid);
    filenames = filenames{1};
else
    filenames = datalist;
end
nfiles = size(filenames, 1);
data = cell(nfiles, 1);
if nargin == 2
for ix = 1 : nfiles,
    data{ix} = htkread(filenames{ix},featCol);
end
else
        if nargin == 1
    for ix = 1 : nfiles,
    data{ix} = htkread(filenames{ix});
    end
        else % all parameters
            if ~isempty(vadCol)
                for ix = 1 : nfiles,
                data{ix} = htkread(filenames{ix},featCol,vadCol,vadThr);
                end
            else
                for ix = 1 : nfiles,
                data{ix} = htkread(filenames{ix},featCol);
                end    
            end
        end


end

