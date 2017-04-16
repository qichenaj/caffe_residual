DIR=../..
SOLVER=$DIR/examples/dream/net/solver.prototxt
NET=$DIR/examples/dream/net/ResNet-50-softmax.prototxt
WEIGHTS=$DIR/examples/dream/ResNet-50-model.caffemodel

VIEW=$DIR/examples/dream/model_png
rm -rf $VIEW/*
python $DIR/python/draw_net.py $NET $VIEW/pp.png --rankdir=LR