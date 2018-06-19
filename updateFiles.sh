while read ip; do
    echo $ip
    pscp -pw iiitkota $PWD/environment.py iiitk@$ip:/home/iiitk/bui/environment.py
    # pscp -pw iiitkota $PWD/train.py iiitk@$ip:/home/iiitk/bui/train.py
    pscp -pw iiitkota $PWD/requirements.txt iiitk@$ip:/home/iiitk/bui/requirements.txt
done < iplist.txt
