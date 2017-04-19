#!/usr/bin/env sh

DIR=../..
SOLVER=$DIR/examples/dream/net/solver.prototxt
NET=$DIR/examples/dream/net/ResNet-50-softmax.prototxt
WEIGHTS=$DIR/examples/dream/snapshot/ResNet_iter_6000.caffemodel

$DIR/build/tools/caffe test -model $NET -weights $WEIGHTS -iterations 100 -gpu 0
