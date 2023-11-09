@echo off & setlocal
set batchPath=%~dp0
set destination="%batchPath%\..\pomdp-baselines.zip"
if exist %destination% del %destination%
"C:\Program Files\7-zip\7z.exe" a -tzip %destination% "%batchPath%\.." "%batchPath%\..\deployment" -mx0 -xr!logs* -xr!__pycache__ -xr!data -xr!.git -xr!wandb -xr!.idea

scp -i C:\Users\Sander\.ssh\delftblue_rsa %destination% sajvanleeuwen@login.delftblue.tudelft.nl:/scratch/sajvanleeuwen/pomdp-baselines.zip

set target_dir=positional-encodings
ssh delftblue "rm -rf /scratch/sajvanleeuwen/%target_dir%; unzip /scratch/sajvanleeuwen/pomdp-baselines.zip -d /scratch/sajvanleeuwen/%target_dir%; rm /scratch/sajvanleeuwen/pomdp-baselines.zip; exit"
