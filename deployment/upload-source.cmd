@echo off & setlocal
set batchPath=%~dp0
set destination="%batchPath%\..\pomdp-baselines.zip"
if exist %destination% del %destination%
"C:\Program Files\7-zip\7z.exe" a -tzip %destination% "%batchPath%\.." "%batchPath%\..\deployment" -mx0 -xr!logs* -xr!__pycache__ -xr!data -xr!.git -xr!wandb

scp -i C:\Users\Sander\.ssh\delftblue_rsa %destination% sajvanleeuwen@login.delftblue.tudelft.nl:/home/sajvanleeuwen/pomdp-baselines.zip

ssh delftblue "rm -rf /home/sajvanleeuwen/distance-test; unzip /home/sajvanleeuwen/pomdp-baselines.zip -d /home/sajvanleeuwen/distance-test; rm /home/sajvanleeuwen/pomdp-baselines.zip; exit"
