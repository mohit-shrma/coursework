% map probe ids to gene names
load Wang_data
load gene_name

tmp_idx = [];
for i = 1:length(ProbeID)
    [idx] = find(strcmp(ProbeID{i},gene_name));
    tmp_idx = [tmp_idx ; [i idx]];
end

GeneID = gene_name(tmp_idx(:,2),3);
empty_geneID = find(strcmp('', GeneID));

GeneID(empty_geneID) = [];
ProbeID(empty_geneID) = [];
Data(empty_geneID,:) = [];

save Wang_data_named GeneID ProbeID Data PatientID 
clear empty_geneID tmp_idx idx i
