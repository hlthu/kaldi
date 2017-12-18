#!/bin/bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2015  Guoguo Chen
#           2017  Hainan Xu
#           2017  Lu Hunag (THU)

# This script trains LMs on the aishell LM-training data and some other data from 
# People's Daily 1998 and Sougou
# you can get them from the internet
# the vocabulary size is about 137k, and the text from the training data is about 120k lines
# and the text from People's Daily 1998 and Sougou is about 1150k lines.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 102.2 / 249.4.
# Train objf: -5.84 -5.52 -5.03 -4.99 -5.08 -5.04 -4.76 -4.87 -4.65 -4.83 -4.58 -4.75 -4.52 -4.71 -4.45 -4.67 -4.62 -4.45 -4.31 -4.51
# Dev objf:   -11.83 -6.73 -6.36 -6.17 -6.04 -5.96 -5.85 -5.79 -5.75 -5.72 -5.69 -5.67 -5.64 -5.62 -5.60 -5.58 -5.56 -5.54 -5.53 -5.52
# Also this is the PPL for the original 3-gram LM:
# Perplexity over 99496.000000 words is 567.320537

# Begin configuration section.
dir=exp/rnnlm_gru_1b
embedding_dim=800
gru_dim=500
stage=0
train_stage=-10

# variables for lattice rescoring
run_rescore=true
ac_model_dirs="exp/chain/tdnn_opgru_1a_sp"
decode_dir_suffix=rnnlm_gru_1b
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially

. path.sh
. cmd.sh
. utils/parse_options.sh

text=data/train/text
other_text=data/text/other.txt      # This is the data from People's Daily 1998 and Sougou
lexicon=data/local/dict/lexiconp.txt
text_dir=data/rnnlm/text_gru_1b

mkdir -p $dir/config
set -e

for f in $text $lexicon; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for local/wsj_extend_dict.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  echo -n >$text_dir/dev.txt
  # hold out one in every 30 lines as dev data.
  cat $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%30 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/train.txt
  cat $other_text > $text_dir/other.txt
fi

if [ $stage -le 1 ]; then
  cp data/lang/words.txt $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <SPOKEN_NOISE> during training.
  echo "<SPOKEN_NOISE>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
train   3   1.0
other 1   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<SPOKEN_NOISE>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --special-words='<s>,</s>,<SPOKEN_NOISE>,<brk>' \
                           $dir/config/words.txt > $dir/config/features.txt

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
relu-renorm-layer name=tdnn1 dim=$embedding_dim input=Append(0, IfDefined(-1))
gru-layer name=lstm1 cell-dim=$gru_dim  
relu-renorm-layer name=tdnn2 dim=$embedding_dim input=Append(0, IfDefined(-3))
gru-layer name=lstm2 cell-dim=$gru_dim  
relu-renorm-layer name=tdnn3 dim=$embedding_dim input=Append(0, IfDefined(-3))
output-layer name=output include-log-softmax=false dim=$embedding_dim
EOF
  rnnlm/validate_config_dir.sh $text_dir $dir/config
fi

if [ $stage -le 2 ]; then
  rnnlm/prepare_rnnlm_dir.sh $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 3 \
                  --stage $train_stage --num-epochs 10 --cmd "$train_cmd" $dir
fi

if [ $stage -le 4 ] && $run_rescore; then
for ac_model_dir in $ac_model_dirs; do
  echo "$0: Perform lattice-rescoring on $ac_model_dir"
  LM=test
  for decode_set in dev test; do
    (
    decode_dir=${ac_model_dir}/decode_${decode_set}

    # Lattice rescoring
    rnnlm/lmrescore.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.5 --max-ngram-order $ngram_order \
      data/lang_$LM $dir \
      data/${decode_set}_hires ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix}
    ) &
  done
done
fi

exit 0
