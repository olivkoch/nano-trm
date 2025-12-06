#!/bin/bash

# Lambda Labs instance termination script
# Usage: ./terminate_instance.sh <instance_id>

if [ $# -eq 0 ]; then
    echo "Error: Instance ID required"
    echo "Usage: $0 <instance_id>"
    exit 1
fi

INSTANCE_ID=$1

# Check if API key is set
if [ -z "$LAMBDA_API_KEY" ]; then
    echo "Error: LAMBDA_API_KEY environment variable not set"
    echo "Set it with: export LAMBDA_API_KEY='your_api_key'"
    exit 1
fi

echo "Terminating instance: $INSTANCE_ID"

curl -u "${LAMBDA_API_KEY}:" \
     -X POST \
     "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate" \
     -H "Content-Type: application/json" \
     -d "{\"instance_ids\": [\"${INSTANCE_ID}\"]}"

echo ""
echo "Termination request sent"