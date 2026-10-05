#!/bin/bash
rsync -avP /p/scratch/hclimrep/shared/weather_generator_data/ifs-fesom/ /e/data1/climateai/hclimrep/data/ifs-fesom/ > /e/scratch/hclimrep/nowak2/WeatherGenerator2/logs/rsync_fesom.log 2>&1
