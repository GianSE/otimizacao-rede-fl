@echo off
title Launcher do Projeto FL
echo --- Iniciando Projeto de Federated Learning (PyTorch) ---

:: 1. Entra na pasta do script (garante que estamos no lugar certo)
cd /d "%~dp0"

:: 2. Ativa o Ambiente Virtual
echo Ativando ambiente virtual...
call venv\Scripts\activate.bat

:: 3. Inicia o Servidor em uma nova janela
echo Iniciando o Servidor...
start "Servidor FL" cmd /k "venv\Scripts\activate.bat && python server.py"

:: 4. Espera 5 segundos para o servidor carregar
echo Aguardando o servidor iniciar (5s)...
timeout /t 5 /nobreak >nul

:: 5. Inicia o Cliente 0
echo Iniciando Cliente 0...
start "Cliente 0" cmd /k "venv\Scripts\activate.bat && python client_runner.py 0"

:: --- A MÁGICA ESTÁ AQUI: PAUSA DE 15 SEGUNDOS ---
echo Aguardando 15 segundos para liberar a rede...
timeout /t 15 /nobreak >nul
:: ------------------------------------------------

:: 6. Inicia o Cliente 1
echo Iniciando Cliente 1...
start "Cliente 1" cmd /k "venv\Scripts\activate.bat && python client_runner.py 1"

echo.
echo --- Tudo iniciado! Verifique as 3 janelas abertas. ---
pause