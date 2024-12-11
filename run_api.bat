@echo off
if "%1" == "h" goto begin
mshta vbscript:createobject("wscript.shell").run("""%~nx0"" h",0)(window.close)&&exit
:begin
REM 启动ComfyUI 端口：8188
rem activate flux &D:& cd  D:/project/ComfyUI &start "" /b "D:/software/anaconda3/envs/flux/python.exe" "D:/project/ComfyUI/main.py"
REM 启动去背景api 端口：8001
start "" /b "D:/software/anaconda3/envs/BiRefNet/python.exe" "D:\project\BiRefNet\main.py"
REM 启动npc
start "" /b "C:\Users\beall\windows_amd64_client\npc.exe" -server=47.243.201.190:8024 -vkey=uz38ccio49m22t1r -type=tcp
rem 启动gpt 端口：7860
"D:\project\pythonProject\venv\Scripts\python.exe" "D:\project\pythonProject\main.py"