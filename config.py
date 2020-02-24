import json

configFile = "config.json"

with open(configFile) as json_file:
    configData = json.load(json_file)

