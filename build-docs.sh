#!/bin/bash

pdoc3 --force --html -o docs/html src/chatbot/
mkdir -p docs/pdf
pdoc3 --force --pdf -o docs/pdf src/chatbot/ | pandoc  -o docs/pdf/chatbot.pdf
