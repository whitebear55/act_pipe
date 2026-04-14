#!/usr/bin/env bash

# Run this script to get the serial number of an X-Series arm's U2D2 serial converter at /dev/ttyUSB0

udevadm info --name=/dev/ttyUSB0 --attribute-walk | grep ATTRS{serial} | head -n 1 | cut -d '"' -f2
