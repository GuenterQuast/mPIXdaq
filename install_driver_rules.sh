#!/bin/bash

USERID=$(id -u)

UDEVADM=$(which udevadm)

if [ -z "$UDEVADM" ]; then
       echo "Can not find udevadm. Pleas check if udev is installed."
       exit -1
fi

if [ "$USERID" -ne 0 ]; then
	echo "Please run as root or with sudo !!!"
	exit -1
fi

# Install script for Linux distributions
# This is a basic installer that merely copies the include files and
# libraries to the system-wide directories.

# Copy the udev rules file and reload all rules
cp ./60-opalkelly.rules /etc/udev/rules.d
cp ./60-pixet.rules /etc/udev/rules.d
$UDEVADM control --reload-rules

# create symlink to libudev.0
LIBUDEV=`ldconfig -p | grep libudev | grep -oP "/.*" | sed -n 1p`
LIBDIR=$(dirname "${LIBUDEV}")

if [ ! -f "$LIBDIR"/libudev.so.0 ]; then
    sudo ln -s "$LIBUDEV" "$LIBDIR"/libudev.so.0
fi



