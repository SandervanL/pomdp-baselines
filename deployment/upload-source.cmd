@echo off & setlocal
set batchPath=%~dp0
set destination="%batchPath%\..\pomdp-baselines.zip"
if exist %destination% del %destination%
"C:\Program Files\7-zip\7z.exe" a -tzip %destination% "%batchPath%\.." "%batchPath%\..\deployment" -mx0 -xr!logs* -xr!__pycache__ -xr!data -xr!.git

scp -i C:\Users\Sander\.ssh\delftblue_rsa %destination% sajvanleeuwen@login.delftblue.tudelft.nl:/home/sajvanleeuwen/pomdp-baselines.zip

ssh delftblue "rm -rf /home/sajvanleeuwen/pomdp-baselines; unzip /home/sajvanleeuwen/pomdp-baselines.zip -d /home/sajvanleeuwen/pomdp-baselines; rm /home/sajvanleeuwen/pomdp-baselines.zip; exit"
