@echo off
title Launcher do Projeto FL
echo --- Iniciando Projeto de Federated Learning (PyTorch) ---

:: 1. Entra na pasta do script (garante que estamos no lugar certo)
cd /d "%~dp0"

:: 2. Ativa o Ambiente Virtual
:: O comando 'call' é necessário para não fechar o script ao chamar outro bat
echo Ativando ambiente virtual...
call venv\Scripts\activate.bat

:: 3. Inicia o Servidor em uma nova janela
echo Iniciando o Servidor...
start "Servidor FL" cmd /k "venv\Scripts\activate.bat && python server.py"

:: 4. Espera 5 segundos para o servidor carregar
echo Aguardando o servidor iniciar...
timeout /t 5 /nobreak >nul

:: 5. Inicia os 2 Clientes em janelas separadas
echo Iniciando Cliente 0...
start "Cliente 0" cmd /k "venv\Scripts\activate.bat && python client_runner.py 0"

echo Iniciando Cliente 1...
start "Cliente 1" cmd /k "venv\Scripts\activate.bat && python client_runner.py 1"

echo.
echo --- Tudo iniciado! Verifique as 3 janelas abertas. ---
pause