#!/bin/bash
rm -rf ./docs
mkdir -p docs/html
pdoc3 --force --html -o docs/html chatbot
mkdir -p docs/pdf
pdoc3 --force --pdf -o docs/pdf chatbot | pandoc  -o docs/pdf/chatbot.pdf
