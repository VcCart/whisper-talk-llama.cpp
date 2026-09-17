@echo off
set "filename=struktura_%DATE:~-4,4%-%DATE:~-7,2%-%DATE:~-10,2%.txt"

:: Запускаем tree и правильно конвертируем в UTF-8
tree /F > "%temp%\tree_temp.txt"
powershell -Command "& { $c = Get-Content -Path '%temp%\tree_temp.txt' -Encoding OEM; $c | Out-File -FilePath '%filename%' -Encoding utf8 }"
del "%temp%\tree_temp.txt" 2>nul

echo File "%filename%" created in %cd%
pause