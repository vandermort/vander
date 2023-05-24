#!/usr/bin/env bash

for blueprint in experiments/*/blueprint.yaml
do
	echo $blueprint
	vndr.eval  --blueprint $blueprint
done
