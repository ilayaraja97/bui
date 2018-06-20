# bui
sentiment analysis in business intelligence

## Instalation
`$ sudo apt install python3-pip`

`$ sudo pip3 install -r requirements.txt`

## Data cleaning
`$ python3 preProcessing.py -[sml]`

## Training
`$ python3 train.py -[sml] -e epochs -o model -a activation`

## Testing
`$ python3 test.py -[k(i <input>)] -[sml] -o model+activation`
