fid=fopen('GSE2034-22071.txt');
numCols=287;
numHeaders=288;
format1=repmat('%s ',1,numCols);
myHeader=textscan(fid,format1,numHeaders);
myData=textscan(fid,format1,22571);
fclose(fid);

[n d] = size(myHeader);
j = 1;
for i=2:d
    tmp = myHeader{1,i}{numHeaders,1};
    PatientID{j,1} = tmp;
    j = j + 1;
end

[n d] = size(myData);
Data = [];

for i=1:d
    if(i == 1)
        ProbeID = myData{i,1};
    else
        tmp=str2num(char(myData{1,i}));
        Data = [Data, tmp];
    end
end

[clinical_data] = import_clinical_data('clinical_data.txt');
    
save Wang_data Data PatientID ProbeID clinical_data
clear fid numCols numHeaders format1 myHeader myData n d j i tmp 
