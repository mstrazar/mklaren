#!/usr/bin/env bash
# Clean redundant files
rm ./.DS_Store
rm `find * | grep "DS_Store"`
rm `find * | grep "octave-workspace"`
rm `find * | grep ".pyc$"`