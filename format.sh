#!/usr/bin/env bash

# Format code with yapf.
yapf -i -r -p -vv chatbot/

# Format docstrings with docformatter.
docformatter -i -r chatbot/
