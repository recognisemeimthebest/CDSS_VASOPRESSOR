# Launcher: activates the conda env and runs the given Python command.
# Usage:  .\scripts\run.ps1 scripts\verify_env.py
#         .\scripts\run.ps1 -m streamlit run app\main.py
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$activate = "C:\Users\dwd00\anaconda3\Scripts\activate.bat"
$envPath = "G:\anaconda_envs\cdss_vasopressor"
$pyArgs = ($args | ForEach-Object { "`"$_`"" }) -join " "
cmd /c "call `"$activate`" `"$envPath`" && python $pyArgs"
