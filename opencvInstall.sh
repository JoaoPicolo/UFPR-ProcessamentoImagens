#!/bin/bash

# Script feito por Gabriel Pimentel Dolzan
# Data: 12/07/2022

function installOpenCV() {
    # Clone do repositorio do github
    git clone https://github.com/opencv/opencv.git
    # Mudando para a branch atual, atualmente 4.x
    git -C opencv checkout 4.x
}

function compileOpenCV() {
    cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local ../opencv
    make -j4
}

echo Navegando para a home do usuario
cd
echo instalando OpenCV
installOpenCV
echo Criando a pasta build
mkdir -p build && cd build
echo Compilando o OpenCV
compileOpenCV
echo Fim do script

