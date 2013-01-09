import java.util.*
m = HashMap

%{to create a map for data}%
for i=2:clinical_dat_siz(1)
	m.put(clinical_data{i,2}, clinical_data{i,5})
end

%{to know whether relapse or not for particular col}%
m.get(PatientID{1})

relapseList = []
nonrelapseList = []

for iter=1:size(PatientID,1)
    if m.get(PatientID{iter}) == '1'
	   relapseList = [relapseList, iter]
      else
	nonrelapseList = [nonrelapseList, iter]
    end
end

numRelapses = length(relapseList)
numNonRelapses = length(nonrelapseList)

for probeIter=1:size(Data,1)
    for iter=1:numRelapses
        {%do computation on Data(1, relapseList(iter))%}
         end
         for iter=1:numNonRelapses
            {%do computation on Data(1, nonrelapseList(iter))%}
         end
end

	      
