@echo off
REM Launcher: activates the conda env and runs the given Python command.
REM Usage:  scripts\run.bat <python args...>
REM Examples:
REM   scripts\run.bat scripts\verify_env.py
REM   scripts\run.bat -m streamlit run app\main.py
REM   scripts\run.bat -m jupyter lab
set KMP_DUPLICATE_LIB_OK=TRUE
call C:\Users\dwd00\anaconda3\Scripts\activate.bat G:\anaconda_envs\cdss_vasopressor
python %*
