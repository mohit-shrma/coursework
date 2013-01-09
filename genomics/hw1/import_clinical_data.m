function[clinical_data] = import_clinical_data(fileToRead1)

newData1 = importdata(fileToRead1);

j = 1;
for i = 9:length(newData1.textdata)
    newData1.textdata{i,7} = num2str(newData1.data(j));
    j = j + 1;
end

[n d] = size(newData1.textdata);

k = 1
for i = 8:n
    for j = 1:d
        clinical_data{k,j} = newData1.textdata{i,j};
    end
    k = k + 1;
end
