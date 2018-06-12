cat log.txt |grep -i "val_acc:.*"|awk 'BEGIN{i=1}{print $14*100; i=i+1;}'|head -n 50 > rnn-train.txt
cat log.txt |grep -i "val_acc:.*"|awk 'BEGIN{i=-49}{print $14*100; i=i+1;}'|head -n 100|tail -n 50 > gru-train.txt
cat log.txt |grep -i "val_acc:.*"|awk 'BEGIN{i=-49}{print $14*100; i=i+1;}'|head -n 150|tail -n 50 > lstm-train.txt
cat log.txt |grep -i "val_acc:.*"|awk 'BEGIN{i=-99}{print $14*100; i=i+1;}'|tail -n 50 > cnn-train.txt
#echo "set title \"RNN : Epochs vs Validation accuracy\";\
set xlabel \"Epochs --->\";\
set ylabel \"Accuracy(%) --->\";\
set xrange [0:50];\
set yrange [0:100];\
set terminal png;\
set output \"data/rnn-train.png\";\
set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5;\
plot \"rnn-train.txt\" with linespoints ls 1;\
set title \"GRU : Epochs vs Validation accuracy\";\
set xlabel \"Epochs --->\";\
set ylabel \"Accuracy(%) --->\";\
set xrange [0:50];\
set yrange [0:100];\
set terminal png;\
set output \"data/gru-train.png\";\
set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5;\
plot \"gru-train.txt\" with linespoints ls 1;\
set title \"LSTM : Epochs vs Validation accuracy\";\
set xlabel \"Epochs --->\";\
set ylabel \"Accuracy(%) --->\";\
set xrange [0:50];\
set yrange [0:100];\
set terminal png;\
set output \"data/lstm-train.png\";\
set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5;\
plot \"lstm-train.txt\" with linespoints ls 1;"\
|gnuplot
#rm *-train.txt
