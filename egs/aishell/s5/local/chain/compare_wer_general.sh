
# this script is used for comparing decoding results between systems.
# e.g. local/chain/compare_wer_general.sh tdnn_1a_sp

echo "# $0 $*";  # print command line.


set_names() {
  if [ $# != 1 ]; then
    echo "compare_wer_general.sh: internal error"
    exit 1  # exit the program
  fi
  name=$(echo $1 | cut -d: -f1)
  epoch=$(echo $1 | cut -s -d: -f2)
  dirname=exp/chain/$name
  if [ -z $epoch ]; then
    epoch_suffix=""
  else
    used_epochs=true
    epoch_suffix=_epoch${epoch}
  fi
}

echo -n "# System                     "
for x in $*; do   printf " % 9s" $x;   done
echo

echo -n "# WER on dev                 "
for x in $*; do
  set_names $x
  wer=$(grep WER $dirname/decode_dev/wer_* | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "# WER on test                "
for x in $*; do
  set_names $x
  wer=$(grep WER $dirname/decode_test/wer_* | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "# CER on dev                 "
for x in $*; do
  set_names $x
  wer=$(grep WER $dirname/decode_dev/cer_* | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "# CER on test                "
for x in $*; do
  set_names $x
  wer=$(grep WER $dirname/decode_test/cer_* | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo


echo -n "# Final train prob           "
for x in $*; do
  prob=$(grep Overall exp/chain/${x}/log/compute_prob_train.final.log | grep -v xent | awk '{print $8}')
  printf "% 10.3f" $prob
done
echo

echo -n "# Final valid prob           "
for x in $*; do
  prob=$(grep Overall exp/chain/${x}/log/compute_prob_valid.final.log | grep -v xent | awk '{print $8}')
  printf "% 10.3f" $prob
done
echo

echo -n "# Final train prob (xent)    "
for x in $*; do
  prob=$(grep Overall exp/chain/${x}/log/compute_prob_train.final.log | grep -w xent | awk '{print $8}')
  printf "% 10.3f" $prob
done
echo

echo -n "# Final valid prob (xent)    "
for x in $*; do
  prob=$(grep Overall exp/chain/${x}/log/compute_prob_valid.final.log | grep -w xent | awk '{print $8}')
  printf "% 10.4f" $prob
done
echo

echo -n "# Num-parameters             "
for x in $*; do
  num_params=$(grep num-parameters exp/chain/${x}/log/progress.1.log | awk '{print $2}')
  printf "% 10d" $num_params
done
echo
