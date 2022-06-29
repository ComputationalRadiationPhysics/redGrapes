#!/usr/bin/env bash
#
# Copyright 2022 The RedGrapes Community
#
# License: MPL-2.0
# Authors: Axel Huebl

set -eu -o pipefail

sudo apt-get -qqq update
sudo apt-get install -y  \
    build-essential      \
    ca-certificates      \
    ccache               \
    cmake                \
    gnupg                \
    libboost-context-dev \
    libfmt-dev           \
    libfreetype-dev      \
    liblapack-dev        \
    liblapacke-dev       \
    libopenmpi-dev       \
    libpng-dev           \
    libspdlog-dev        \
    ninja-build

# cmake-easyinstall
#
sudo curl -L -o /usr/local/bin/cmake-easyinstall https://git.io/JvLxY
sudo chmod a+x /usr/local/bin/cmake-easyinstall
export CEI_SUDO="sudo"
export CEI_TMP="/tmp/cei"

# PNGwriter
#
CXXFLAGS="" cmake-easyinstall --prefix=/usr/local  \
    git+https://github.com/pngwriter/pngwriter.git \
    -DCMAKE_BUILD_TYPE=Release
