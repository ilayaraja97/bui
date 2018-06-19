while read ip; do
    echo $ip
    mkdir data/$ip
    pscp -batch -pw iiitkota iiitk@$ip:/home/iiitk/bui/log.txt $PWD/data/$ip/log.txt
    pscp -batch -pw iiitkota iiitk@$ip:/home/iiitk/bui/data/model-amazon-lstm_500k.h5 $PWD/data/$ip/model-amazon-lstm_500k.h5
    pscp -batch -pw iiitkota iiitk@$ip:/home/iiitk/bui/data/model-amazon-lstm_500k.json $PWD/data/$ip/model-amazon-lstm_500k.json
done < iplist.txt
