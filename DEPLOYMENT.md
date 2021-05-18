# Deployment and Usage instruction

## docker build and run
- docker build from project root: docker build -t tweet-classifier .
- docker run -dp 8000:8000 tweet-classifier

## Deploying to AWS
### Create EC2 Instance
- connect via terminal after downloading .pem key
- update and install docker
  - sudo yum update -y
  - amazon-linux-extras install docker
  - sudo service docker start
### Create docker image to ECR
- create new repository
- need to install [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
- 'view push commands' from created repository
- follow [AWS ECR authentication instructions](https://docs.aws.amazon.com/AmazonECR/latest/userguide/Registries.html#registry_auth)
- build docker image in local and push to ECR

### Pull docker image into EC2 instance
- connect to EC2 instance
- start docker service
- docker pull <image_uri>(copied from ECR repository)
- run docker image

### Configure EC2 network setting
- EC2 instance -> security -> security group then 'Edit inbound rules'

### Use of service
POST http://{EC2 instance URL}/classify

curl -X 'POST' \
  'http://34.217.127.191/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data": [
    "tweet1",
    "tweet2",
    ...
  ]
}'

response 200 (0: not an actual disaster, 1: disaster)
{
  "data": [
    0,
    0,
    1,
    ...
  ]
}
