import boto3

# Create a Bedrock Runtime client in the AWS Region of your choice.
bedrock = boto3.client("bedrock", region_name="us-east-1")

response = bedrock.list_foundation_models()

for m in response["modelSummaries"]:
        print(
            m["modelId"],
            "| status =", m["modelLifecycle"]["status"],
            "| provider =", m["providerName"]
        )


