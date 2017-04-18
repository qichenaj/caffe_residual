DIR=../..
SOLVER=$DIR/examples/dream/net/solver.prototxt
NET=$DIR/examples/dream/net/ResNet-50-softmax.prototxt
#NET=$DIR/examples/dream/net/mcc_hdf5_train_test.prototxt
VIEW=$DIR/examples/dream/model_png

rm -rf $VIEW/*
python $DIR/python/draw_net.py $NET $VIEW/model.png --rankdir=LR