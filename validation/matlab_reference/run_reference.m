% Runs the reference MATLAB/Octave implementation on every CSV in in/ and writes results to out/.
addpath(getenv('REFDIR'));
files = dir('in/*.csv');
for f = 1:numel(files)
  name = files(f).name(1:end-4);
  I = dlmread(fullfile('in', files(f).name));
  dlmwrite(fullfile('out', [name '_Zh.csv']), visibilityPatches(I, 1, 'horizontal'), 'precision', 17);
  dlmwrite(fullfile('out', [name '_Zn.csv']), visibilityPatches(I, 1, 'natural'), 'precision', 17);
  dlmwrite(fullfile('out', [name '_Eh_T.csv']), imageVisibilityGraph(I, 'horizontal', true));
  dlmwrite(fullfile('out', [name '_Eh_F.csv']), imageVisibilityGraph(I, 'horizontal', false));
  dlmwrite(fullfile('out', [name '_En_T.csv']), imageVisibilityGraph(I, 'natural', true));
  dlmwrite(fullfile('out', [name '_En_F.csv']), imageVisibilityGraph(I, 'natural', false));
  printf('%s done\n', name);
end
% Also the Lena image shipped with the reference repository (patches only)
S = load(fullfile(getenv('REFDIR'), 'lena512.mat')); fn = fieldnames(S); L = double(S.(fn{1}));
dlmwrite('in/lena512.csv', L, 'precision', 17);
dlmwrite('out/lena512_Zh.csv', visibilityPatches(L, 1, 'horizontal'), 'precision', 17);
dlmwrite('out/lena512_Zn.csv', visibilityPatches(L, 1, 'natural'), 'precision', 17);
printf('lena512 done\n');
