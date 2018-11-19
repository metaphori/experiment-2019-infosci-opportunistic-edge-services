#!/bin/bash
git add data/*
git commit -m "[AUTOMATIC ${USER}@${HOSTNAME} ] - Data generated up to `date`"
git pull --no-edit
git push
