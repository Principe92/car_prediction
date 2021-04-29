#!/usr/bin/env bash

# download 
mkdir data
zip -F data.zip --out combined.zip
unzip -P d89551fd190e38 combined.zip -d data
